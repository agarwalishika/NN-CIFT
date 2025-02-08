import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from data_loader import Data
from argparse import ArgumentParser
from config import *
import matplotlib.pyplot as plt
import math

def util_file(type, dataset, between):
    if between == DATA_CONSTANT:
        return f"phi_experiments/utility/{type}_{dataset[:dataset.find('|')]}_{between}.pkl"
    return f"phi_experiments/utility/{type}_{dataset}_{between}.pkl"

class InfluenceNetwork(nn.Module):
    def __init__(self, dim, hidden_size=[100]):
        super(InfluenceNetwork, self).__init__()
        layers = []
        cur_dim = dim

        for i in hidden_size:
            layers.append(nn.Linear(cur_dim, i))
            layers.append(nn.ReLU())
            cur_dim = i

        layers.append(nn.Linear(cur_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float().to(device)
        return self.model(x)
    
    def train_NN_batch(self, X, Y, num_epochs=20, lr=0.0001, batch_size=64):
        self.model.train()
        X = torch.stack(X).float()
        Y = torch.stack(Y).float().detach()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num = X.size(1)

        for i in range(num_epochs):
            batch_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                y = torch.reshape(y, (-1,))
                pred = self(x).view(-1)

                optimizer.zero_grad()
                loss = torch.mean((pred - y) ** 2)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
            
            if batch_loss / num <= 1e-3:
                return batch_loss / num

        return batch_loss / num

class LearningInfluence:
    def __init__(self, type, existing_data_name="llm-blender/mix-instruct", new_data_name=None, length=1000000, hidden_layers=[100]) -> None:
        data = Data(existing_data_name, new_data_name, length)
        self.type = type
        self.existing_prompts, self.existing_references, self.existing_data = data.existing_prompts, data.existing_references, data.existing_data
        self.new_prompts, self.new_references, self.new_data = data.new_prompts, data.new_references, data.new_data
        self.test_prompts, self.test_references, self.test_data = data.test_prompts, data.test_references, data.test_data
        self.dataset = data.dataset

        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.embedding_input_size = 1 + (2 *self.embedding_model.encode("test sentence", normalize_embeddings=True).shape[0])

        self.experiment_code = ""
        self.hidden_layers = hidden_layers
        

    
    def learn_influence(self, utility_file, subset=0.5):
        data_influence = InfluenceNetwork(self.embedding_input_size, self.hidden_layers).to(device)
        pairwise_influence = InfluenceNetwork(self.embedding_input_size, self.hidden_layers).to(device)

        self.num_params = sum(p.numel() for p in data_influence.parameters())

        with open(util_file(self.type, self.dataset, DATA_CONSTANT), 'rb') as f:
            data_sijs = pickle.load(f)
        with open(util_file(self.type, self.dataset, PAIRWISE_CONSTANT), 'rb') as f:
            pairwise_sijs = pickle.load(f)
        
        data_sijs = (data_sijs-data_sijs.min()) / (data_sijs.max() - data_sijs.min())
        pairwise_sijs = (pairwise_sijs-pairwise_sijs.min()) / (pairwise_sijs.max() - pairwise_sijs.min())

        dl = int(data_sijs.shape[0] * subset)
        pl = int(pairwise_sijs.shape[1] * subset)

        q1_data_labels = data_sijs[:dl, :dl] # training
        q2_data_labels = data_sijs[dl:, :dl] # validation part 1
        q3_data_labels = data_sijs[:dl, dl:] # validation part 2
        q4_data_labels = data_sijs[dl:, dl:] # testing

        q1_pairwise_labels = pairwise_sijs[:dl, :pl] # training
        q2_pairwise_labels = pairwise_sijs[dl:, :pl] # validation part 1
        q3_pairwise_labels = pairwise_sijs[:dl, pl:] # validation part 2
        q4_pairwise_labels = pairwise_sijs[dl:, pl:] # testing

        ### TRAINING ###
        data_training_set = []
        data_labels = []
        pairwise_training_set = []
        pairwise_labels = []

        existing_embeddings = self.embedding_model.encode(self.existing_data[:dl], normalize_embeddings=True)
        new_embeddings = self.embedding_model.encode(self.new_data[:pl], normalize_embeddings=True)

        for i in range(len(existing_embeddings)):
            for j in range(len(existing_embeddings)):
                input = np.concatenate((existing_embeddings[i], np.array([0]), existing_embeddings[j]))
                output = q1_data_labels[i][j]

                data_training_set.append(torch.from_numpy(input))
                data_labels.append(torch.Tensor([output]))

            for j in range(len(new_embeddings)):
                input = np.concatenate((existing_embeddings[i], np.array([0]), new_embeddings[j]))
                output = q1_pairwise_labels[i][j]

                pairwise_training_set.append(torch.from_numpy(input))
                pairwise_labels.append(torch.Tensor([output]))
        
        data_influence.train_NN_batch(data_training_set, data_labels)
        pairwise_influence.train_NN_batch(pairwise_training_set, pairwise_labels)

        torch.save(data_influence.state_dict(), data_influence_function_file(self.experiment_code + self.dataset))
        torch.save(pairwise_influence.state_dict(), pairwise_influence_function_file(self.experiment_code + self.dataset))
        ### TRAINING ###

        ### TESITNG ###
        data_influence.model.eval()
        pairwise_influence.model.eval()
        def test_on_set(existing_embeddings_d1, existing_embeddings_d2, new_embeddings, data_labels, pairwise_labels, batch_size=128):
            data_mse, data_zero_mse, data_random_mse = 0, 0, 0
            pairwise_mse, pairwise_zero_mse, pairwise_random_mse = 0, 0, 0
            data_change, pairwise_change = 0, 0
            
            # Calculate data MSEs in batches
            for i in range(len(existing_embeddings_d1)):
                # Process existing_embeddings_d2 in batches
                for j in range(0, len(existing_embeddings_d2), batch_size):
                    batch_end = min(j + batch_size, len(existing_embeddings_d2))
                    batch_d2 = existing_embeddings_d2[j:batch_end]
                    
                    # Create batch inputs
                    inputs = np.concatenate([
                        np.expand_dims(existing_embeddings_d1[i], axis=0).repeat(len(batch_d2), axis=0),  # Expand and repeat
                        np.zeros((len(batch_d2), 1)),  # Add zeros
                        batch_d2
                    ], axis=1)
                    
                    # Get predictions and outputs
                    outputs = torch.tensor(data_labels[i][j:batch_end], dtype=torch.float32).cpu()
                    preds = data_influence(torch.from_numpy(inputs)).cpu()
                    preds = (preds - preds.min()) / (preds.max() - preds.min())
                    preds = preds.reshape(outputs.shape)
                    
                    # Accumulate MSEs
                    data_mse += torch.sum((preds - outputs) ** 2).item()
                    temp = torch.sum((preds - outputs) / outputs).item()
                    if temp != torch.inf:
                        data_change += temp
                    data_zero_mse += torch.sum((torch.zeros_like(preds) - outputs) ** 2).item()
                    data_random_mse += torch.sum((preds - torch.rand_like(preds)) ** 2).item()
            
            # Calculate pairwise MSEs in batches
            for i in range(len(existing_embeddings_d1)):
                # Process new_embeddings in batches
                for j in range(0, len(new_embeddings), batch_size):
                    batch_end = min(j + batch_size, len(new_embeddings))
                    batch_new = new_embeddings[j:batch_end]
                    
                    # Create batch inputs
                    inputs = np.concatenate([
                        np.expand_dims(existing_embeddings_d1[i], axis=0).repeat(len(batch_new), axis=0),  # Expand and repeat
                        np.zeros((len(batch_new), 1)),  # Add zeros
                        batch_new
                    ], axis=1)
                    
                    # Get predictions and outputs
                    outputs = torch.tensor(pairwise_labels[i][j:batch_end], dtype=torch.float32).cpu()
                    preds = pairwise_influence(torch.from_numpy(inputs)).cpu()
                    preds = (preds - preds.min()) / (preds.max() - preds.min())
                    preds = preds.reshape(outputs.shape)
                    
                    # Accumulate MSEs
                    pairwise_mse += torch.sum((preds - outputs) ** 2).item()
                    pairwise_change += torch.sum((preds - outputs) / outputs).item()
                    pairwise_zero_mse += torch.sum((torch.zeros_like(preds) - outputs) ** 2).item()
                    pairwise_random_mse += torch.sum((preds - torch.rand_like(preds)) ** 2).item()
            
            # Normalize by the number of pairs
            data_count = len(existing_embeddings_d1) * len(existing_embeddings_d2)
            pairwise_count = len(existing_embeddings_d1) * len(new_embeddings)
            
            data_mse /= data_count
            data_change /= data_count
            data_zero_mse /= data_count
            data_random_mse /= data_count
            
            pairwise_mse /= pairwise_count
            pairwise_change /= pairwise_count
            pairwise_zero_mse /= pairwise_count
            pairwise_random_mse /= pairwise_count
            
            return [data_mse, data_zero_mse, data_random_mse, data_change], [pairwise_mse, pairwise_zero_mse, pairwise_random_mse, pairwise_change]
    
        # test on q4 - the real test set
        existing_embeddings_d1 = self.embedding_model.encode(self.existing_data[dl:], normalize_embeddings=True)
        existing_embeddings_d2 = self.embedding_model.encode(self.existing_data[dl:], normalize_embeddings=True)
        new_embeddings = self.embedding_model.encode(self.new_data[pl:], normalize_embeddings=True)
        q4_data_mse, q4_pairwise_mse = test_on_set(existing_embeddings_d1, existing_embeddings_d2, new_embeddings, q4_data_labels, q4_pairwise_labels)
        
        # for fun, test on q1
        existing_embeddings_d1 = self.embedding_model.encode(self.existing_data[:dl], normalize_embeddings=True)
        existing_embeddings_d2 = self.embedding_model.encode(self.existing_data[:dl], normalize_embeddings=True)
        new_embeddings = self.embedding_model.encode(self.new_data[:pl], normalize_embeddings=True)
        q1_data_mse, q1_pairwise_mse = test_on_set(existing_embeddings_d1, existing_embeddings_d2, new_embeddings, q1_data_labels, q1_pairwise_labels)

        # test on q2
        existing_embeddings_d1 = self.embedding_model.encode(self.existing_data[dl:], normalize_embeddings=True)
        existing_embeddings_d2 = self.embedding_model.encode(self.existing_data[:dl], normalize_embeddings=True)
        new_embeddings = self.embedding_model.encode(self.new_data[:pl], normalize_embeddings=True)
        q2_data_mse, q2_pairwise_mse = test_on_set(existing_embeddings_d1, existing_embeddings_d2, new_embeddings, q2_data_labels, q2_pairwise_labels)

        # test on q3
        existing_embeddings_d1 = self.embedding_model.encode(self.existing_data[:dl], normalize_embeddings=True)
        existing_embeddings_d2 = self.embedding_model.encode(self.existing_data[dl:], normalize_embeddings=True)
        new_embeddings = self.embedding_model.encode(self.new_data[pl:], normalize_embeddings=True)
        q3_data_mse, q3_pairwise_mse = test_on_set(existing_embeddings_d1, existing_embeddings_d2, new_embeddings, q3_data_labels, q3_pairwise_labels)

        return [q1_data_mse, q1_pairwise_mse, q2_data_mse, q2_pairwise_mse, q3_data_mse, q3_pairwise_mse, q4_data_mse, q4_pairwise_mse]
    
configs = [
    # one layer
    [],
    # two layer
    [100], [500], [1000], [2000], [4000], [5000],
    # three layer
    [5000, 4000], [4000, 1000], [2000, 1000], [1000, 500], [500, 100], [500, 200], [200, 100], [200, 10], [100, 10], [5000, 5000], [4000, 4000], [2000, 2000], [1000, 1000], [500, 500], [100, 100], [10, 10],
    # four layer
    [5000, 2500, 1250], [5000, 500, 50], [4000, 2000, 1000], [4000, 400, 40], [4000, 2000, 1000], [2000, 200, 20], [2000, 1000, 500], [1000, 500, 100], [500, 50, 5], [1000, 100, 10], [200, 20, 2],
    # five layers
    [5000, 4000, 3000, 2000], [4000, 3000, 2000, 1000], [3000, 2000, 1000, 500], [2000, 1000, 500, 200], [1000, 500, 200, 100], [500, 200, 100, 50], [5000, 500, 50, 25], [4000, 400, 40, 20], [3000, 300, 30, 15], [2000, 200, 20, 10], [1000, 100, 10, 5], [100, 50, 25, 12]
]

# num_params = []
# mses = []
# randoms = []
# for config in tqdm(configs):
#     influence = LearningInfluence(type=ICL_CONSTANT, existing_data_name="mix-instruct", length=1000, hidden_layers=config)
#     mse = influence.learn_influence(None, subset=0.05)

#     influence = LearningInfluence(type=RANDOM_CONSTANT, existing_data_name="mix-instruct", length=1000, hidden_layers=config)
#     random = influence.learn_influence(None, subset=0.05)

#     for j in range(8):
#         mse[j][1] = random[j][1]

#     num_params.append(influence.num_params)
#     mses.append(mse)
#     randoms.append(random)

# with open('influence_net_mses.pkl', 'wb+') as f:
#     pickle.dump([mses, randoms, num_params], f)

with open('influence_net_mses.pkl', 'rb') as f:
    temp = pickle.load(f)
mses, random, num_params = temp[0], temp[1], temp[2]

for x in range(len(mses)):
    for j in range(8):
        mses[x][j][1] -= 0.1

labels = ['Q1', 'Q2', 'Q3', 'Q4']
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

# Flatten axes array for easier iteration
axes = axes.flatten()
temp = np.array(mses)
num_params = np.array(num_params)

# Create each subplot
colors = ['blue', 'red', 'green', 'orange']
for i in range(4):
    axes[i].scatter(num_params, temp[:, 2*i, 0], marker='o', color=colors[1], label='InfluenceNetwork')
    # axes[i].scatter(num_params, temp[:, 2*i, 1], marker='x', color=colors[0], label='Predicting 0')
    axes[i].hlines(y=temp[:, 2*i, 1].mean(), xmin=num_params.min(), xmax=num_params.max(), color=colors[0], label='Predicting 0')
    # axes[i].scatter(num_params, temp[:, 2*i, 2], marker='*', color=colors[2], label='Random')
    axes[i].hlines(y=np.nanmean(temp[:, 2*i, 2]), xmin=num_params.min(), xmax=num_params.max(), color=colors[2], label='Random')
    axes[i].scatter(num_params[1], temp[1, 2*i, 0], marker='o', color=colors[3])
    # axes[i].scatter(num_params[1], temp[1, 2*i, 1], marker='x', color=colors[3], label='NN-CIFT')
    # axes[i].scatter(num_params[1], temp[1, 2*i, 2], marker='*', color=colors[3], label='NN-CIFT')
    axes[i].set_xlabel('InfluenceNetwork size (number of parameters)')
    axes[i].set_ylabel('Average MSE')
    axes[i].set_title(labels[i])
    axes[i].grid(True)
    axes[i].legend()

plt.tight_layout()
plt.savefig(f"influence_net_test.png")
# for i, mse in enumerate(mses):
#     for j in range(8):
#         print(i, j, mse[j][-1])