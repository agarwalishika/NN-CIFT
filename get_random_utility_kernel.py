from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from data_loader import Data
from config import *

class RandomUtility:
    def __init__(self) -> None:
        """
        Initialize the ModelDependentICLUtility class.

        Args:
            model: Pre-trained language model.
            tokenizer: Tokenizer corresponding to the pre-trained language model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        hi = 9
       
    def calculate_random_utility(self, 
                              train_prompts, 
                              train_responses, 
                              valid_prompts=None, 
                              valid_responses=None):
        """
        Calculate the in-context learning (ICL) utility for a given set of prompts and responses.

        Args:
            train_prompts (list): List of training prompts.
            train_responses (list): List of training responses.
            valid_prompts (list, optional): List of validation prompts. Defaults to None.
            valid_responses (list, optional): List of validation responses. Defaults to None.
            kernel_type (str): Type of kernel to use for calculating utility ('euclidean' or 'exponential').
            scaling (str): Method to scale the utility values ('min-max').

        Returns:
            utility_kernel (np.ndarray): Utility kernel matrix.
            u_{ij}  means how much the j-th example in the training set is useful for the i-th example in the validation set.
        """
        if valid_prompts is None or valid_responses is None:
            valid_prompts = train_prompts
            valid_responses = train_responses
        
        n_train = len(train_prompts)
        n_valid = len(valid_prompts)
        utility_kernel = np.random.rand(n_valid, n_train)
        
        return utility_kernel

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--existing_data_name', type=str)
    argparser.add_argument('--new_data_name', type=str)
    argparser.add_argument('--length', type=int)
    argparser.add_argument('--between', type=str)
    args = argparser.parse_args()

    data = Data(args.existing_data_name, args.new_data_name, args.length)

    mod_dep = RandomUtility()
    if "data" in args.between:
        sijs = mod_dep.calculate_random_utility(data.existing_prompts, data.existing_references)
        constant = DATA_CONSTANT
    elif "pairwise" in args.between:
        sijs = mod_dep.calculate_random_utility(data.existing_prompts, data.existing_references, data.new_prompts, data.new_references)
        constant = PAIRWISE_CONSTANT
    
    with open(utility_file(RANDOM_CONSTANT, data.dataset, constant), 'wb+') as f:
        pickle.dump(sijs, f)