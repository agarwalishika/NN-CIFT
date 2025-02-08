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
    def __init__(self, dim, hidden_size=100):
        super(InfluenceNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.float().to(device)
        return self.fc2(self.activate(self.fc1(x)))
    
    def train_NN_batch(self, X, Y, num_epochs=20, lr=0.0001, batch_size=64):
        self.train()
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
                y = torch.reshape(y, (1,-1))
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
    def __init__(self, type, existing_data_name="llm-blender/mix-instruct", new_data_name=None, length=1000000) -> None:
        data = Data(existing_data_name, new_data_name, length)
        self.type = type
        self.existing_prompts, self.existing_references, self.existing_data = data.existing_prompts, data.existing_references, data.existing_data
        self.new_prompts, self.new_references, self.new_data = data.new_prompts, data.new_references, data.new_data
        self.test_prompts, self.test_references, self.test_data = data.test_prompts, data.test_references, data.test_data
        self.dataset = data.dataset

        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.embedding_input_size = 1 + (2 *self.embedding_model.encode("test sentence", normalize_embeddings=True).shape[0])

        self.experiment_code = ""

    
    def learn_influence(self, util_file, subset=0.5):
        data_influence = InfluenceNetwork(self.embedding_input_size).to(device)
        pairwise_influence = InfluenceNetwork(self.embedding_input_size).to(device)

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
        def test_on_set(existing_embeddings_d1, existing_embeddings_d2, new_embeddings, data_labels, pairwise_labels, batch_size=128):
            data_mse, data_zero_mse, data_random_mse = 0, 0, 0
            pairwise_mse, pairwise_zero_mse, pairwise_random_mse = 0, 0, 0
            
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
                    data_mse += torch.sum((preds - outputs)).item()
                    data_zero_mse += torch.sum((torch.zeros_like(preds) - outputs)).item()
                    data_random_mse += torch.sum((preds - torch.rand_like(preds))).item()
            
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
                    pairwise_zero_mse += torch.sum((torch.zeros_like(preds) - outputs) ** 2).item()
                    pairwise_random_mse += torch.sum((preds - torch.rand_like(preds)) ** 2).item()
            
            # Normalize by the number of pairs
            data_count = len(existing_embeddings_d1) * len(existing_embeddings_d2)
            pairwise_count = len(existing_embeddings_d1) * len(new_embeddings)
            
            data_mse /= data_count
            data_zero_mse /= data_count
            data_random_mse /= data_count
            
            pairwise_mse /= pairwise_count
            pairwise_zero_mse /= pairwise_count
            pairwise_random_mse /= pairwise_count
            
            return [data_mse, data_zero_mse, data_random_mse], [pairwise_mse, pairwise_zero_mse, pairwise_random_mse]
    
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
    
influence = LearningInfluence(type=ICL_CONSTANT, existing_data_name="mix-instruct", length=1000)

subsets = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3] #, 0.4, 0.5, 0.6, 0.7, 0.8]
mses = []
for subset in tqdm(subsets):
    mses.append(influence.learn_influence(utility_file, subset=0.2))