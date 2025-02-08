from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from data_loader import Data
from config import *
import scipy.stats

def calculate_confidence(model, tokenizer, prompts, batch_size=4, save_path=None):
    def encode_data_point(prompt):
        encoded = tokenizer(prompt, return_tensors='pt', padding=True).to(model.device)

        inputs = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

        return inputs

    decoded_outputs, confidences = [], []
    prompts = list(prompts)

    for_range = tqdm(range(0, len(prompts), batch_size), desc="batch inference") if len(prompts) > 1 else range(0, len(prompts), batch_size)

    for i in for_range:
        batch_prompts = prompts[i:i+batch_size]
        encoded_input = encode_data_point(batch_prompts)
        with torch.no_grad():
            outputs = model.generate(
                **encoded_input,
                temperature=0.1,
                do_sample=True,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_scores=True
            )
        answer_start = len(encoded_input['input_ids'][0])
        decoded_output = [
            tokenizer.decode(sequence[answer_start:], skip_special_tokens=True)
            for sequence in outputs.sequences
        ]

        if len(batch_prompts) > 1:
            get_scores = lambda i: torch.stack([outputs.scores[j][i] for j in range(len(outputs.scores))]) 
            scores = [get_scores(i) for i in range(len(batch_prompts))]
        else:
            scores = [torch.stack(outputs.scores).squeeze()]

        batch_confs = []
        for score in scores:
            probs = F.softmax(score, dim=-1)
            top_probs, _ = torch.topk(probs, k=20, dim=-1)
            batch_confs.append(top_probs)
            # batch_confs.append((top_probs[:, 0] - top_probs[:, 1]).mean())

        decoded_outputs.extend(decoded_output)
        confidences.extend(batch_confs)

        if save_path:
            with open(save_path, "wb+") as f:
                    pickle.dump([decoded_outputs, confidences], f)
        
    return decoded_output, confidences

class ModelDependentConfidenceUtility:
    def __init__(self, model_name="distilbert/distilgpt2", device='cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """
        Initialize the ModelDependentConfidenceUtility class.

        Args:
            model: Pre-trained language model.
            tokenizer: Tokenizer corresponding to the pre-trained language model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        self.model_name = model_name
        self.device = device
        self.instruction_with_icl = lambda ex_in, ex_out, prompt: f"Use the following example as a guide to answer the query in the same format and style, providing a clear and concise response.\nExample:\n{ex_in}\n{ex_out}\nQuery:\n{prompt}\n"

    def get_raw_confidences(self, path, train_prompts, train_references, valid_prompts=None):
        if valid_prompts is None:
            valid_prompts = train_prompts

        _, raw_confidence_without = calculate_confidence(self.model, self.tokenizer, valid_prompts)
        raw_confidence_without = raw_confidence_without * len(raw_confidence_without)
    
        raw_confidence_with = []
        for i in tqdm(range(len(train_prompts)), desc="active icl utility"):
            icl_prompts = [self.instruction_with_icl(train_prompts[i], train_references[i], prompt) for prompt in valid_prompts]
            _, batch_raw_confidence_with = calculate_confidence(self.model, self.tokenizer, icl_prompts)
            raw_confidence_with.append(batch_raw_confidence_with)
        
        with open(path, 'wb+') as f:
            pickle.dump([raw_confidence_without, raw_confidence_with], f)
        
        return path

    def calculate_active_utility(self, confidence_path, shorthand):
        with open(confidence_path, 'rb') as f:
            scores_without, scores_with = pickle.load(f)

        confidence_with_icl = []
        for score in scores_with:
            confidence_with_icl.append(self.get_scores(shorthand, score))
        confidence_with_icl = torch.Tensor(confidence_with_icl)

        confidence_without_icl = torch.Tensor(self.get_scores(shorthand, scores_without))
        confidence_without_icl = confidence_without_icl.reshape(confidence_with_icl.shape)

        return (confidence_without_icl - confidence_with_icl).detach().numpy()
    
    def get_scores(self, utility_criteria, confidence):
        if "min" in utility_criteria:
            scores = self.get_min_confidence(confidence)
        elif "max" in utility_criteria:
            scores = self.get_max_confidence(confidence)
        elif "mean" in utility_criteria:
            scores = self.get_mean_confidence(confidence)
        elif "entropy" in utility_criteria:
            scores = self.get_entropy_confidence(confidence)
        return scores

    
    def get_min_confidence(self, confidence):
        scores = []
        for i in range(len(confidence)):
            top_probs, _ = torch.topk(confidence[i], k=2, dim=-1)
            top_two_diff = top_probs[:, 0] - top_probs[:, 1]
            scores.append(top_two_diff.min())
        return scores
    
    def get_max_confidence(self, confidence):
        scores = []
        for i in range(len(confidence)):
            top_probs, _ = torch.topk(confidence[i], k=2, dim=-1)
            top_two_diff = top_probs[:, 0] - top_probs[:, 1]
            scores.append(top_two_diff.max())
        return scores
    
    def get_mean_confidence(self, confidence):
        scores = []
        for i in range(len(confidence)):
            top_probs, _ = torch.topk(confidence[i], k=2, dim=-1)
            top_two_diff = top_probs[:, 0] - top_probs[:, 1]
            scores.append(top_two_diff.mean())
        return scores
    
    def get_entropy_confidence(self, confidence):
        scores = []
        for i in range(len(confidence)):
            scores.append(scipy.stats.entropy(confidence[i].cpu()).mean())
        return scores

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--existing_data_name', type=str)
    argparser.add_argument('--new_data_name', type=str)
    argparser.add_argument('--length', type=int)
    argparser.add_argument('--shorthand', type=str, default="mean")
    args = argparser.parse_args()

    if args.shorthand not in CONFIDENCE_CALCULATION_SHORTHANDS:
        0/0

    data = Data(args.existing_data_name, args.new_data_name, args.length)

    conf_dep = ModelDependentConfidenceUtility()
    path = raw_confidence_file(data.dataset)
    conf_dep.get_raw_confidences(path, data.existing_prompts, data.existing_references, data.new_prompts)
    sijs = conf_dep.calculate_active_utility(path, args.shorthand)
    
    with open(confidence_utility_file(data.dataset, args.shorthand), 'wb+') as f:
        pickle.dump(sijs, f)