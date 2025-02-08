from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import evaluate
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from tqdm import tqdm

embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to('cpu')
embedding_model.eval()

if embedding_tokenizer.pad_token is None:
    embedding_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    embedding_model.resize_token_embeddings(len(embedding_tokenizer))
    
embedding_model = embedding_model.to('cuda')

def find_embedding(prompt):
    encoded_input = embedding_tokenizer(prompt, padding=True, truncation=True, return_tensors='pt').to(embedding_model.device)
    with torch.no_grad():
        model_output = embedding_model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings).squeeze()
    return sentence_embeddings

def evaluate_bge(predictions, references):
    metrics = []

    for pred, ref in zip(predictions, references):
        pred_emb = find_embedding(pred)
        ref_emb = find_embedding(ref)
        metrics.append(ref_emb.dot(pred_emb).item())    
    return np.array(metrics)

def evaluate_rouge(predictions, references):
    metrics = []
    rouge_metric = evaluate.load('rouge')

    return np.array([rouge_metric.compute(predictions=predictions, references=references)['rouge1']])

def process(output):
    """
    Strips and returns only the first line (everything before the first \n).
    """

    output = output.strip()
    if '\n' in output:
        output = output[:output.find('\n')]
    if '</s>' in output:
        output = output[:output.find('</s>')]
    return output

def load_prometheus():
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
    return judge

def evaluate_prometheus(judge: PrometheusEval, instructions, predictions, references, return_individual=False):
    """
    Calculates the similarity between predictions and references using Prometheus (LLM-as-a-Judge).

    Args:
        predictions: list of strings for the hypothesis
        references: list of strings for the reference
        return_invidiual: if True, it will return the individual scores for corresponding prediction-reference pairs
    Returns:
        np array of metrics of size 1x1 if return_individual is True, else 1x|predictions|
    """

    rubric_data = {
    "criteria": """Evaluate the model's ability to follow instructions and deliver a high-quality response across the following dimensions:
1. **Instruction Following**: How accurately and fully does the model adhere to the given instruction?
2. **Accuracy**: Is the information correct, reliable, and factually sound?
3. **Relevance**: Does the response directly address the question or task without unnecessary information?
4. **Completeness**: Does the response cover all essential aspects of the instruction or question?
5. **Depth**: How thoroughly does the response explore the topic? Does it demonstrate insightful analysis where appropriate?
6. **Clarity**: Is the response well-organized, easy to follow, and free from ambiguity or confusion?
7. **Creativity**: Does the response offer original or innovative approaches where applicable?
8. **Helpfulness**: Does the response effectively meet the user's needs and provide value in solving the problem or addressing the query?""",
    
    "score1_description": "The response fails to meet expectations across most or all criteria. It does not follow the instruction, contains significant errors or misinformation, lacks relevance, is incomplete or shallow, unclear, unoriginal, and unhelpful.",
    
    "score2_description": "The response shows major deficiencies across several criteria. It partially follows the instruction but includes significant inaccuracies, is often irrelevant, incomplete, or lacks depth, clarity, creativity, and helpfulness.",
    
    "score3_description": "The response is average, meeting some but not all criteria. It follows the instruction but may fall short in terms of accuracy, depth, relevance, or helpfulness. Improvements in clarity and insightfulness may be needed.",
    
    "score4_description": "The response is strong, performing well across most criteria. It follows the instruction closely, is mostly accurate and relevant, provides good depth, and is well-structured. Minor improvements could enhance clarity, creativity, or helpfulness.",
    
    "score5_description": "The response excels in all or nearly all criteria. It fully follows the instruction, is highly accurate, directly relevant, complete, and demonstrates depth and insight. The response is well-organized, creative where appropriate, and very helpful in addressing the user's needs.",
    }

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    # feedbacks, metrics = judge.absolute_grade(
    #     instructions=instructions,
    #     responses=predictions,
    #     rubric=score_rubric,
    #     reference_answers=references
    # )

    bs = 15

    metrics = []
    for i in range(0, len(predictions), bs):
        try:
            _, scores = judge.absolute_grade(
                instructions=instructions[i:i+bs],
                responses=predictions[i:i+bs],
                rubric=score_rubric,
                reference_answers=references[i:i+bs]
            )

            metrics.extend(scores)
        except:
            continue
        
    # # clean up memory
    # del model
    # del judge
    # torch.cuda.empty_cache()
    
    return np.array(metrics)