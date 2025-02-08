import os
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from argparse import ArgumentParser
from config import *
from learn_influence import LearningInfluence, InfluenceNetwork
import submodlib as sb
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bge_evaluation import *
from fine_tune_model import FineTunedModel
import time

peft_path_f = lambda type, ins_p, sss_p, use_delift: f'experiments_{suffix}/{type}/peft/{ins_p}_{sss_p}_{use_delift}.pkl'
peft_code_f = lambda type, ins_p, sss_p, use_delift: f'{suffix},{type},peft,{ins_p},{sss_p},{use_delift}.pkl'
icl_path_f = lambda type, ins_p, sss_p, use_delift: f'experiments_{suffix}/{type}/icl/{ins_p}_{sss_p}_{use_delift}.pkl'
icl_code_f = lambda type, ins_p, sss_p, use_delift: f'{suffix},{type},icl,{ins_p},{sss_p},{use_delift}.pkl'
def exists(current_results, code):
    for line in current_results:
        if code in line:
            return True
    return False

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--existing_data_name', type=str)
    argparser.add_argument('--new_data_name', type=str)
    argparser.add_argument('--length', type=int)
    argparser.add_argument('--results_file', type=str)
    argparser.add_argument('--suffix', type=str)
    
    args = argparser.parse_args()

    suffix = args.suffix

    results_file = args.results_file
    with open(results_file, 'r+') as f:
        current_results = f.readlines()

    # influence_network_subset = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # subset_selection_subset = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    influence_network_subset = [0.05]
    subset_selection_subset = [0.3]
    types = [ICL_CONSTANT, SE_CONSTANT, SELECTIT_CONSTANT, LESS_CONSTANT]
    influence = LearningInfluence(args.existing_data_name, args.new_data_name, length=args.length)
    test_prompts = influence.test_prompts
    test_references = influence.test_references

    judge = load_prometheus()

    for type in types:
        for ins_p in tqdm(reversed(influence_network_subset), desc="influence network subset"):
            for sss_p in tqdm(reversed(subset_selection_subset), desc="subset_selection_subset"):
                for use_delift in reversed([False]):
                    peft_path = peft_path_f(type, ins_p, sss_p, use_delift)
                    peft_code = peft_code_f(type, ins_p, sss_p, use_delift)
                    # try:
                    if not exists(current_results, peft_code):
                        with open(peft_path, 'rb') as f:
                            peft = pickle.load(f)
                        peft_prompts, peft_texts = peft[0], peft[1]

                        bge_peft = evaluate_bge(peft_texts, test_references).mean()
                        rouge_peft = evaluate_rouge(peft_texts, test_references).mean()
                        prometheus_peft = evaluate_prometheus(judge, peft_prompts, peft_texts, test_references)
                        prometheus_peft = prometheus_peft[prometheus_peft != None].mean()
                        with open(results_file, 'a+') as f:
                            f.write(f'{peft_code},{bge_peft},{rouge_peft}, {prometheus_peft}\n')
                    # except:
                    #     with open(results_file, 'a+') as f:
                    #         f.write(f'{peft_code},ERROR\n')
                    
                    icl_path = icl_path_f(type, ins_p, sss_p, use_delift)
                    icl_code = icl_code_f(type, ins_p, sss_p, use_delift)
                    # try:
                    if not exists(current_results, icl_code):
                        with open(icl_path, 'rb') as f:
                            icl = pickle.load(f)
                        icl_prompts, icl_texts = icl[0], icl[1]

                        bge_icl = evaluate_bge(icl_texts, test_references).mean()
                        rouge_icl = evaluate_rouge(icl_texts, test_references).mean()
                        prometheus_icl = evaluate_prometheus(judge, icl_prompts, icl_texts, test_references).mean()
                        with open(results_file, 'a+') as f:
                            f.write(f'{icl_code},{bge_icl},{rouge_icl}, {prometheus_icl}\n')
                    # except:
                    #     with open(results_file, 'a+') as f:
                    #         f.write(f'{icl_code},ERROR\n')