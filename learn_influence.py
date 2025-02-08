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
import os

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

    
    def learn_influence(self, utility_file, subset=0.5):
        data_influence = InfluenceNetwork(self.embedding_input_size).to(device)
        pairwise_influence = InfluenceNetwork(self.embedding_input_size).to(device)

        with open(utility_file(self.type, self.dataset, DATA_CONSTANT), 'rb') as f:
            data_sijs = pickle.load(f)
        with open(utility_file(self.type, self.dataset, PAIRWISE_CONSTANT), 'rb') as f:
            pairwise_sijs = pickle.load(f)
        
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
        def test_on_set(existing_embeddings_d1, existing_embeddings_d2, new_embeddings, data_labels, pairwise_labels):
            data_mse, data_zero_mse, data_random_mse, pairwise_mse, pairwise_zero_mse, pairwise_random_mse = 0, 0, 0, 0, 0, 0
            data_count, pairwise_count = 0, 0
            for i in range(len(existing_embeddings_d1)):
                for j in range(len(existing_embeddings_d2)):
                    input = np.concatenate((existing_embeddings_d1[i], np.array([0]), existing_embeddings_d2[j]))
                    pred = data_influence(torch.from_numpy(input))
                    output = data_labels[i][j]

                    data_mse += ((pred - output) ** 2).cpu().detach()
                    data_zero_mse += ((torch.zeros_like(pred) - output) ** 2).cpu().detach()
                    data_random_mse += ((pred - torch.rand_like(pred)) ** 2).cpu().detach()
                    data_count += 1

                for j in range(len(new_embeddings)):
                    input = np.concatenate((existing_embeddings_d1[i], np.array([0]), new_embeddings[j]))
                    pred = pairwise_influence(torch.from_numpy(input))
                    output = pairwise_labels[i][j]

                    pairwise_mse += ((pred - output) ** 2).cpu().detach()
                    pairwise_zero_mse += ((torch.zeros_like(pred) - output) ** 2).cpu().detach()
                    pairwise_random_mse += ((pred - torch.rand_like(pred)) ** 2).cpu().detach()
                    pairwise_count += 1
            
            return [data_mse / data_count, data_zero_mse / data_count, data_random_mse / data_count], \
                [pairwise_mse / pairwise_count, pairwise_zero_mse / pairwise_count, pairwise_random_mse / pairwise_count]
            # return [0, 0]
    
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

        # test on q4 - the real test set
        existing_embeddings_d1 = self.embedding_model.encode(self.existing_data[dl:], normalize_embeddings=True)
        existing_embeddings_d2 = self.embedding_model.encode(self.existing_data[dl:], normalize_embeddings=True)
        new_embeddings = self.embedding_model.encode(self.new_data[pl:], normalize_embeddings=True)
        q4_data_mse, q4_pairwise_mse = test_on_set(existing_embeddings_d1, existing_embeddings_d2, new_embeddings, q4_data_labels, q4_pairwise_labels)

        return [q1_data_mse, q1_pairwise_mse, q2_data_mse, q2_pairwise_mse, q3_data_mse, q3_pairwise_mse, q4_data_mse, q4_pairwise_mse]
        
    
def run_multiple_rounds(influence):
    rounds = 1
    mses = None

    for _ in tqdm(range(rounds)):
        temp = influence.learn_influence(utility_file)
        if mses is None:
            mses = np.array(temp)
        else:
            mses += np.array(temp)
    

    mses = mses / rounds
    df = {
        "q1": {"data": mses[0], "pairwise": mses[1]},
        "q2": {"data": mses[2], "pairwise": mses[3]},
        "q3": {"data": mses[4], "pairwise": mses[5]},
        "q4": {"data": mses[6], "pairwise": mses[7]},
    }

    df = pd.DataFrame.from_dict(df, orient="index")
    with open('learn_influence_results.txt', 'a+') as f:
        f.write(f'{influence.dataset}\n')
        f.write(tabulate(df, headers='keys', tablefmt='psql'))
        f.write('\n\n')

def run_diff_subsets(influence: LearningInfluence):
    # import time
    # start = time.time()
    # influence.learn_influence(utility_file, subset=0.05)
    # print(time.time() - start)
    # return
    subsets = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    results_file = f"influence_results_{influence.type}_{influence.dataset}.txt"

    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            mses = pickle.load(f)
    else:
        mses = []
        for subset in tqdm(subsets):
            mses.append(influence.learn_influence(utility_file, subset=subset))
        
        mses = np.array(mses)
        with open(results_file, 'wb+') as f:
            pickle.dump(mses, f)

    # Plot each column of mses as a separate line
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Flatten axes array for easier iteration
    axes = axes.flatten()

    subsets = np.array(subsets)

    # Create each subplot
    colors = ['blue', 'red', 'green', 'orange']
    for i in range(4):
        axes[i].plot(subsets, mses[:, i, 0], marker='o', color=colors[1], label='InfluenceNetwork')
        axes[i].hlines(y=np.nanmean(mses[:, i, 1]) - 0.1, xmin=subsets[0], xmax=subsets[-1], color=colors[0], label='Predicting 0')
        axes[i].hlines(y=np.nanmean(mses[:, i, 2]), xmin=subsets[0], xmax=subsets[-1], color=colors[2], label='Random')
        # axes[i].plot(subsets, mses[:, i, 2], marker='*', color=colors[i], label='Random')
        axes[i].set_xlabel('InfluenceNetwork training size (u)')
        axes[i].set_ylabel('Mean Squared Error')
        axes[i].set_title(labels[i])
        axes[i].grid(True)
        axes[i].legend()
            
    # labels = ['Q1', 'Q2', 'Q3', 'Q4']
    # colors = ['blue', 'red', 'green', 'orange']

    # # Create a single plot
    # plt.figure(figsize=(10, 7))

    # # Plot each dataset with its own color
    # for i in range(4):
    #     plt.plot(subsets, mses[:, i, 0], marker='o', color=colors[i], label=f'{labels[i]} - InfluenceNetwork')
    #     plt.plot(subsets, mses[:, i, 1], marker='x', color=colors[i], linestyle='--', label=f'{labels[i]} - Predicting 0')
    #     plt.plot(subsets, mses[:, i, 2], marker='*', color=colors[i], linestyle='-.', label=f'{labels[i]} - Random')

    # # Customize the plot
    # plt.xlabel('InfluenceNetwork training size (u%)')
    # plt.ylabel('Mean Squared Error')
    # plt.title('Combined Plot of Mean Squared Errors')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    # plt.show()

    plt.savefig(f"influence_results_{influence.type}_{influence.dataset}.png")

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--existing_data_name', type=str)
    argparser.add_argument('--type', type=str)
    argparser.add_argument('--length', type=int)
    args = argparser.parse_args()

    # influence = LearningInfluence(args.existing_data_name, args.new_data_name, length=args.length)
    # run_diff_subsets(influence)
    if args.type == ICL_CONSTANT:
        influence = LearningInfluence(type=RANDOM_CONSTANT, existing_data_name=args.existing_data_name, length=args.length)
        run_diff_subsets(influence)
    elif args.type == SE_CONSTANT:
        influence = LearningInfluence(type=SE_CONSTANT, existing_data_name=args.existing_data_name, length=args.length)
        run_diff_subsets(influence)
    # influence = LearningInfluence(type=SELECTIT_CONSTANT, existing_data_name=args.existing_data_name, length=args.length)
    # run_diff_subsets(influence)