import os
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from argparse import ArgumentParser
from config import *
import submodlib as sb
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bge_evaluation import *
from fine_tune_model import FineTunedModel
import time

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

class LinearInfluenceNetwork(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(LinearInfluenceNetwork, self).__init__()
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

class LearningLinearInfluence:
    def __init__(self, type, existing_data_name="llm-blender/mix-instruct", new_data_name=None, length=1000000) -> None:
        data = Data(existing_data_name, new_data_name, length)
        self.type = type
        self.existing_prompts, self.existing_references, self.existing_data = data.existing_prompts, data.existing_references, data.existing_data
        self.new_prompts, self.new_references, self.new_data = data.new_prompts, data.new_references, data.new_data
        self.test_prompts, self.test_references, self.test_data = data.test_prompts, data.test_references, data.test_data
        self.dataset = data.dataset

        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.embedding_input_size = self.embedding_model.encode("test sentence", normalize_embeddings=True).shape[0]

        self.experiment_code = ""

    
    def learn_influence(self, utility_file, subset=0.5):
        data_influence = LinearInfluenceNetwork(self.embedding_input_size).to(device)

        with open(utility_file(self.type, self.dataset, DATA_CONSTANT), 'rb') as f:
            data_sijs = pickle.load(f)
        
        dl = int(data_sijs.shape[0] * subset)

        q1_data_labels = data_sijs[:dl] # training
        q2_data_labels = data_sijs[dl:] # validation part 1

        ### TRAINING ###
        data_training_set = []
        data_labels = []

        existing_embeddings = self.embedding_model.encode(self.existing_data[:dl], normalize_embeddings=True)

        for i in range(len(existing_embeddings)):
            data_training_set.append(torch.from_numpy(existing_embeddings[i]))
            data_labels.append(torch.Tensor([q1_data_labels[i]]))
        
        data_influence.train_NN_batch(data_training_set, data_labels)

        torch.save(data_influence.state_dict(), data_influence_function_file(self.experiment_code + self.dataset))
        ### TRAINING ###

        ### TESITNG ###
        def test_on_set(existing_embeddings_d1, data_labels):
            data_mse = 0
            data_count = 0
            for i in range(len(existing_embeddings_d1)):
                    pred = data_influence(torch.from_numpy(existing_embeddings_d1[i]))
                    output = data_labels[i]

                    data_mse += ((pred - output) ** 2).cpu().detach()
                    data_count += 1
            
            return data_mse / data_count
    
        # for fun, test on q1
        existing_embeddings_d1 = self.embedding_model.encode(self.existing_data[:dl], normalize_embeddings=True)
        q1_data_mse = test_on_set(existing_embeddings_d1, q1_data_labels)

        # test on q2
        existing_embeddings_d1 = self.embedding_model.encode(self.existing_data[dl:], normalize_embeddings=True)
        q2_data_mse = test_on_set(existing_embeddings_d1, q2_data_labels)

        return [q1_data_mse, q2_data_mse]
        

class UsingLinearInfluence:
    def __init__(self, influence: LearningLinearInfluence, influence_network_subset_perc, subset_selection_perc, use_selectit) -> None:
        self.influence = influence

        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.embedding_input_size = self.embedding_model.encode("test sentence", normalize_embeddings=True).shape[0]

        self.language_model_name = "distilbert/distilgpt2"
        self.init_models(self.language_model_name)

        self.influence_network_subset_perc = influence_network_subset_perc
        self.subset_selection_perc = subset_selection_perc
        self.use_selectit = use_selectit

        self.experiment_code = f'{self.influence.type}_{self.influence_network_subset_perc}_{self.subset_selection_perc}'
        self.influence.experiment_code = self.experiment_code
    
    def write_costs(self, description, time):
        return
        with open(f'cache/costs/{self.influence_network_subset_perc}_{self.subset_selection_perc}_{self.use_selectit}.txt', 'a+') as f:
            f.write(f'{description},{time}\n')
    
    def init_models(self, language_model_name):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            language_model_name, 
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        
        ## left padding for generation
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.model_max_length = 1024
        self.model.eval()

    def use_subset_icl(self, subset=None):
        start = time.time()
        if subset is not None:
            prompts, references = self.create_icl_inference_data(subset[0], self.influence.test_prompts, self.influence.test_references)
        else:
            prompts, references = self.influence.test_prompts, self.influence.test_references
        self.write_costs('icl_subset_augmentation', time.time() - start)
        return self.use_subset(prompts, references, is_peft=False)

    def use_subset_peft(self, subset=None):
        start = time.time()
        if subset is not None:
            del self.model, self.tokenizer
            torch.cuda.empty_cache()

            model_dir = f'ft_model_{len(subset[0])}/'
            ft = FineTunedModel(self.language_model_name)
            ft.fine_tune(inputs=subset[1], outputs=subset[2], model_dir=model_dir)
            self.init_models(model_dir)
        
        self.write_costs('peft_subset_learning', time.time() - start)
        return self.use_subset(self.influence.test_prompts, self.influence.test_references, is_peft=True)

        
    def use_subset(self, prompts, references, is_peft):
        start = time.time()
        max_length = int(self.tokenizer.model_max_length)
        batch_size = 4

        all_gen_texts = []

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_references = references[i:i+batch_size]

            # Modify prompts in place
            for j, prompt in enumerate(batch_prompts):
                batch_prompts[j] = prompt + "\n### Output:\n"

            tokenized_output = self.tokenizer(
                list(batch_prompts), 
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                gen_tokens = self.model.generate(
                    **tokenized_output,
                    max_new_tokens=150,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    min_p=0.1,
                    temperature=0.2,
                )

            # Extract only the new tokens
            new_tokens = gen_tokens[:, tokenized_output.input_ids.shape[1]:]
            
            # Decode only the new tokens
            batch_gen_texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            batch_gen_texts = [gen_text.replace("\n", " ") for gen_text in batch_gen_texts]
            all_gen_texts.extend(batch_gen_texts)
    
        if is_peft:
            self.write_costs('evaluating_subset_peft', time.time() - start)
        else:
            self.write_costs('evaluating_subset_icl', time.time() - start)
        return prompts, all_gen_texts


    def create_icl_inference_data(self, training_data, validation_prompts, validation_references, k=5):
        prompts = []
        references = []
        pool_embeddings = pool_embeddings = torch.stack([torch.Tensor(self.embedding_model.encode(sentences=p, normalize_embeddings=True)) for p in training_data])

        for j in range(len(validation_prompts)):
            prompt_j = validation_prompts[j]
            nearest_indices, pool_embeddings = self.find_nearest_neighbors(pool_embeddings=pool_embeddings, query=prompt_j, k=k)
            prompt = ""
            for i, n in enumerate(nearest_indices):
                prompt += f"Example {i+1}:\n{training_data[n]}\n----------\n"
            
            prompt += prompt_j
            reference = validation_references[j]

            prompts.append(prompt)
            references.append(reference)
        return prompts, references

    def find_nearest_neighbors(self, query, pool=None, pool_embeddings=None, return_sim=False, k=5):
        """
        Given a query (singular or list), return the nearest neighbors wrt cosine simliarity in the pool set.

        Args:
            query: singular prompt or list of prompts to search for
            pool: list of prompts to search from (or provide pool_embeddings)
            pool_embeddings: embeddings of the list of prompts to search from (or provide pool)
            return_sim: True if just the similarity values should be returned, False if the indices and pool_embeddings should be returned
            k: number of nearest neighbors
        Returns:
            Either the similarity values, or the indices/pool_embeddings (for future use)
        """
        assert not (pool == None and pool_embeddings == None)

        # Get embeddings for pool and query
        if pool_embeddings == None:
            pool_embeddings = torch.stack([torch.Tensor(self.embedding_model.encode(sentences=p, normalize_embeddings=True)) for p in pool])
        if type(query) is list:
            query_embedding = torch.stack([torch.Tensor(self.embedding_model.encode(sentences=q, normalize_embeddings=True)) for q in query])
        else:
            query_embedding = torch.stack([torch.Tensor(self.embedding_model.encode(sentences=query, normalize_embeddings=True))])
        similarities = cosine_similarity(query_embedding.cpu(), pool_embeddings.cpu())

        if return_sim:
            return similarities

        nearest_indices = similarities.argsort()[0][-k:][::-1]
        return nearest_indices, pool_embeddings

    def create_subset(self, subset_selec_perc, influence_network_perc, use_selectit=False):
        start = time.time()
        if subset_selec_perc == 0:
            self.write_costs(f'subset_selection_creation_{use_selectit}', time.time() - start)
            return None
        
        dl = int(len(self.influence.existing_data) * influence_network_perc)
        
        if use_selectit:
            with open(utility_file(self.influence.type, self.influence.dataset, DATA_CONSTANT), 'rb') as f:
                data_sijs = pickle.load(f)
            data_sijs = data_sijs[dl:]
        else:
            data_influence = LinearInfluenceNetwork(self.embedding_input_size).to(device)
            data_influence.load_state_dict(torch.load(data_influence_function_file(self.experiment_code + self.influence.dataset)))
            
            existing_embeddings_d1 = self.embedding_model.encode(self.influence.existing_data[dl:], normalize_embeddings=True)

            data_sijs = []
            for i in range(len(existing_embeddings_d1)):
                input = existing_embeddings_d1[i]
                data_sijs.append(data_influence(torch.from_numpy(input)).detach().cpu())
        
        data_sijs = np.argsort(np.concatenate(data_sijs))
        subset = data_sijs[:int(len(data_sijs) * subset_selec_perc)]
        subset = (np.array(self.influence.new_data[dl:])[subset], np.array(self.influence.new_prompts[dl:])[subset], np.array(self.influence.new_references[dl:])[subset])
        
        self.write_costs(f'subset_selection_creation_{use_selectit}', time.time() - start)
        return subset

peft_path_f = lambda type, ins_p, sss_p, use_selectit: f'experiments_{suffix}/{type}/peft/{ins_p}_{sss_p}_{use_selectit}.pkl'
icl_path_f = lambda type, ins_p, sss_p, use_selectit: f'experiments_{suffix}/{type}/icl/{ins_p}_{sss_p}_{use_selectit}.pkl'
def exists(type, ins_p, sss_p, use_selectit=[True, False]):
    peft_path_T = peft_path_f(type, ins_p, sss_p, True)
    peft_path_F = peft_path_f(type, ins_p, sss_p, False)
    icl_path_T = icl_path_f(ins_p, sss_p, True)
    icl_path_F = icl_path_f(ins_p, sss_p, False)

    return os.path.exists(peft_path_T) and os.path.exists(peft_path_F) and os.path.exists(icl_path_T) and os.path.exists(icl_path_F) 

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--existing_data_name', type=str)
    argparser.add_argument('--length', type=int)
    argparser.add_argument('--type', type=str, default=SELECTIT_CONSTANT)
    argparser.add_argument('--suffix', type=str, default="mixinstruct")
    args = argparser.parse_args()

    suffix = args.suffix

    # FOR HYPERPARAMETER STUDY:
    # influence_network_subset = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # subset_selection_subset = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # FOR PERFORMING EXPERIMENTS
    influence_network_subset = [0.05]
    subset_selection_subset = [0.1, 0.2, 0.3]
    influence = LearningLinearInfluence(args.type, args.existing_data_name, args.existing_data_name, length=args.length)

    for ins_p in tqdm(influence_network_subset, desc="influence network subset"):
        for sss_p in tqdm(subset_selection_subset, desc="subset_selection_subset"):
            # if exists(ins_p, sss_p):
            #     print('CONTINUING')
            #     continue
            if sss_p > 0:
                influence.experiment_code = f'{influence.type}_{ins_p}_{sss_p}'
                influence.learn_influence(utility_file, ins_p)

            for use_selectit in reversed([False]):
                using_influence = UsingLinearInfluence(influence, ins_p, sss_p, use_selectit)
                subset = using_influence.create_subset(subset_selec_perc=sss_p, influence_network_perc=ins_p)
                
                peft_path = peft_path_f(influence.type, ins_p, sss_p, use_selectit)
                if not os.path.exists(peft_path):
                    peft_prompts, peft_texts = using_influence.use_subset_peft(subset)
                    with open(peft_path, 'wb') as f:
                        pickle.dump([peft_prompts, peft_texts], f)

                icl_path = icl_path_f(influence.type, ins_p, sss_p, use_selectit)
                if not os.path.exists(icl_path):
                    icl_prompts, icl_texts = using_influence.use_subset_icl(subset)
                    with open(icl_path, 'wb') as f:
                        pickle.dump([icl_prompts, icl_texts], f)
                
                del using_influence