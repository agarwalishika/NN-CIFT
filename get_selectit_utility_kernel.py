import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fine_tune_model import FineTunedModel
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from data_loader import Data
from config import *
import numpy as np
import random
import torch
from use_influence import UsingInfluence
from learn_influence import LearningInfluence

# code for baseline SelectIT
# github: https://github.com/Blue-Raincoat/SelectIT/blob/main/
# paper: https://arxiv.org/pdf/2402.16705v1

class SelectIT:
    def __init__(self, model_name="distilbert/distilgpt2", device='cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """
        Initialize the SelectIT class.

        Args:
            model: Pre-trained language model.
            tokenizer: Tokenizer corresponding to the pre-trained language model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name    
        self.RATING_PROMPTS = ["Assign a score from 1 to 5 to each assistant based on how accurately they follow the instructions and response provided, ensuring the score is represented clearly on its own.",
                        "Score each assistant on a scale from one to five, reflecting the accuracy of their adherence to the instructions and input, and present this score plainly without the need for extra details.",
                        "Rate each assistant's response accuracy to the given task and input on a scale of 1 to 5, with 5 being the most precise; the score should be self-explanatory and presented as a single line.",
                        "Evaluate the assistants by the correctness of their response to the provided directions and response, giving them a rating between 1 and 5—where a 5 indicates top accuracy—and output the score clearly as a standalone line.", 
                        "Rate each assistant on a scale of 1 to 5 based on their adherence to the instructions and the accuracy of their responses, with the score clearly displayed.", 
                        "For each assistant, allocate a score between 1 and 5 reflecting how precisely they follow instructions and their response accuracy, making sure the score is distinctly visible.", 
                        "Score each assistant from one to five, judging by how well they follow the given instructions and the correctness of their replies, and ensure that the score is clearly marked.", 
                        "Assign to every assistant a score ranging from 1 to 5, evaluating their compliance with instructions and the precision of their feedback, with the score being conspicuously presented.", 
                        "Provide a score of 1 to 5 for each assistant, considering their fidelity to instructions and the accuracy of their answers, and ensure the score is clearly indicated."
                    ]
        
    def init_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left', trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        self.device = device

    def construction_rps(self, prompts, references):
        promote = random.choice(self.RATING_PROMPTS)
        instruction = "Instruction:"
        response = "Response:"
        ress = '\nThe answer is: \n'
        rating_prompt_list = []
        for ins, res in zip(prompts, references):
            rating_prompt = promote + '\n' + instruction + ins + '\n' + response + res + ress
            rating_prompt_list.append(rating_prompt)
        return rating_prompt_list

    def sentence_level_self_reflection(self, prompts, references, alpha=0.2, k=5):
        # self.model.to('cuda')
        rps = self.construction_rps(prompts, references)
        pro = []
        for idx, p in enumerate(rps):
            tokenized = self.tokenizer(p, padding=True, return_tensors="pt").to(self.device)
            tokenized.input_ids = tokenized.input_ids.cuda()
            tokenized.attention_mask = tokenized.attention_mask.cuda()
            with torch.no_grad():
                try:
                    outputs = self.model(**tokenized)
                    predictions = outputs[0]
                    logits = predictions[:, -1, :]
                    softmax_logits = torch.softmax(logits.float(), dim=-1)
                    if "" in self.model_name:
                        for index in range(1):
                            tmp_res = [float(softmax_logits[index][29896]), float(softmax_logits[index][29906]),
                                    float(softmax_logits[index][29941]), float(softmax_logits[index][29946]),
                                    float(softmax_logits[index][29945])]
                            pro.append(tmp_res)
                    elif "Qwen" in self.model_name:
                        for index in range(1):
                            tmp_res = [float(softmax_logits[index][16]), float(softmax_logits[index][17]),
                                    float(softmax_logits[index][18]), float(softmax_logits[index][19]),
                                    float(softmax_logits[index][20])]
                            pro.append(tmp_res)
                except Exception as ex:
                    print(ex)
        # self.model.to('cpu')
        pro_softmax = []
        for item in pro:
            tmp_pro_softmax = item
            tmp0_pro_softmax = []
            tmp1_pro_softmax = []
            for idx, item in enumerate(tmp_pro_softmax):
                tmp0_pro_softmax.append(np.exp(tmp_pro_softmax[idx] / sum(tmp_pro_softmax)))
            for jdx, item in enumerate(tmp0_pro_softmax):
                tmp1_pro_softmax.append(tmp0_pro_softmax[jdx] / sum(tmp0_pro_softmax))
            pro_softmax.append(tmp1_pro_softmax)

        data_num = int(len(pro_softmax) / k)
        sentence_level_score = []
        for idx in range(data_num):
            token_level_score = []
            for id in range(idx * k, (idx + 1) * k):
                score_base = np.argmax(pro_softmax[id])
                tmp_sum = 0
                for tmp_idx in range(k):
                    tmp_sum += pro_softmax[id][score_base] - pro_softmax[id][tmp_idx]
                tmp_sum = tmp_sum / (k - 1 + 1e-10)
                token_score = (score_base + 1) * tmp_sum
                token_level_score.append(token_score)
            avg = np.average(token_level_score)
            std = np.std(token_level_score)
            sentence_level_score.append(avg / (1 + alpha * std))

        return sentence_level_score

    def model_level_self_reflection(self, prompts, references, alpha=0.2, k=5):
        self.init_models()
        model_level_score = []
        for _ in range(3):
            model_level_score.append(self.sentence_level_self_reflection(prompts, references, alpha=alpha, k=k))
        selectit_score = []
        data_num = int(len(model_level_score[0]))
        for idx in range(data_num):
            selectit_score.append(
                (model_level_score[0][idx] + model_level_score[1][idx] + model_level_score[2][idx]) / 3
                )
        
        del self.tokenizer, self.model
        return np.array(selectit_score)

    def get_subset(self, prompts, references, alpha=0.2, k=5, proportion=0.3):
        select_it_scores = self.model_level_self_reflection(prompts, references, alpha, k)
        inds = np.argsort(select_it_scores)
        inds = inds[:int(len(inds) * proportion)]

        # add utility values to subset
        sub_prompts, sub_references = [], []
        for i in inds:
            sub_prompts.append(prompts[i])
            sub_references.append(references[i])
        
        return sub_prompts, sub_references

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--existing_data_name', type=str)
    argparser.add_argument('--length', type=int)
    args = argparser.parse_args()

    data = Data(args.existing_data_name, args.existing_data_name, args.length)

    selectit = SelectIT()
    sijs = selectit.model_level_self_reflection(data.existing_prompts, data.existing_references, k=1)
    constant = DATA_CONSTANT
    
    with open(utility_file(SELECTIT_CONSTANT, data.dataset, constant), 'wb+') as f:
        pickle.dump(sijs, f)