from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from data_loader import Data
from config import *
from sympy import use
import torch
import faiss
import numpy as np
from torch.nn import functional as F

__DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def convert_distances_to_similarities(distances, metric, kw):
    """
    Convert distances to similarities based on the specified metric.

    Parameters:
    distances (torch.Tensor): Tensor of distances.
    metric (str): The metric to use ('cosine', 'rbf').
    kw (float): Kernel width for rbf metric.
    """

    assert metric in ['cosine', 'dot', 'rbf'], "Metric must be 'cosine', 'dot' or 'rbf'."
    if metric in ['cosine', 'dot']:
        similarities = 1 - ( (distances ** 2)/2)
    elif metric == 'rbf':
        # For rbf, an example conversion could be using the exponential decay
        # Get squared distance matrix
        similarities = distances ** 2
        avg_dist = torch.mean(similarities)
        torch.div(similarities, kw*avg_dist, out=similarities)
        torch.exp(-similarities, out=similarities)
    else:
        raise ValueError(f"Unknown metric for sparse similarity: {metric}")
    return similarities

def convert_compatible_faiss_tensor(tensor):
    """
    Ensure the tensor is compatible with FAISS, which requires 32-bit floating-point tensors.
    
    Args:
        tensor (torch.Tensor): The tensor to check and convert if necessary.

    Returns:
        torch.Tensor: A tensor compatible with FAISS.
    """
    if tensor.dtype != torch.float32:
        return tensor.type(torch.float32)
    return tensor
    
    
def compute_similarity_chunk(chunk, knn, num_neighbors, metric, kw, scaling):
    """
    Compute similarities for a chunk of data.

    Parameters:
    chunk (torch.tensor): A chunk of the array1.
    knn (faiss.IndexFlatL2): KNN Index object used for search
    num_neighbors (int): Number of neighbors to find for each row in chunk.
    metric (str): The metric to use.
    kw (float): Kernel width for rbf metric.
    scaling (str, optional): Type of scaling to apply.

    Returns:
    tuple: Indices of neighbors, calculated similarities.
    """
    # Assume faiss.Index is built outside and passed as a parameter 	

    distances, indices = knn.search(chunk, num_neighbors)  
    distances = torch.tensor(distances, dtype=torch.float32)
    similarities = convert_distances_to_similarities(distances, metric, kw)

    if scaling == 'min-max':
        min_val, max_val = similarities.min(), similarities.max()
        if max_val != min_val:
            similarities = (similarities - min_val) / (max_val - min_val)
    elif scaling == 'additive':
        similarities = (similarities + 1) / 2

    return indices, similarities

def compute_pairwise_sparse(tensor1, tensor2, num_neighbors, batch_size, metric='cosine', scaling=None, 
                            kw=0.1, n_list=100, use_inverse_index=False, device=__DEVICE):
    """
    Compute pairwise similarities between rows of two tensors using sparse representation.

    Parameters:
    tensor1 (torch.Tensor): First Tensor.
    tensor2 (torch.Tensor): Second Tensor.
    num_neighbors (int): Number of neighbors to find for each row in array1.
    batch_size (int): Size of each batch for computation.
    metric (str): The metric to use ('cosine', 'dot', 'euclidean').
    scaling (str, optional): Type of scaling to apply.
    kw (float, optional): Kernel width for rbf metric.
    n_trees (int, optional): Number of trees for Annoy index.
    """

    if tensor2 is None:
        tensor2 = tensor1

    tensor1, tensor2 = convert_compatible_faiss_tensor(tensor1), convert_compatible_faiss_tensor(tensor2)
    if metric == 'cosine':
        tensor1, tensor2 = F.normalize(tensor1, p=2, dim=1), F.normalize(tensor2, p=2, dim=1)

    # FAISS Index Creation
    def create_faiss_index(tensor, device, use_inverse_index, n_list):
        index_type = faiss.IndexFlatL2 if not use_inverse_index else faiss.IndexIVFFlat
        res = None

        # Initialize GPU resources if not using CPU
        if device != 'cpu':
            res = faiss.StandardGpuResources()
            res.noTempMemory()

        # Handle the creation of a flat or IVF index
        if not use_inverse_index:
            knn_index = index_type(tensor.shape[1])
            if device != 'cpu':
                # Correctly transfer the created flat index to GPU
                knn_index = faiss.index_cpu_to_gpu(res, 0, knn_index)
        else:
            quantizer = faiss.IndexFlatL2(tensor.shape[1])
            knn_index = index_type(quantizer, tensor.shape[1], n_list, faiss.METRIC_L2)
            if device != 'cpu':
                # Transfer the trained IVF index to GPU
                knn_index = faiss.index_cpu_to_gpu(res, 0, knn_index)
            knn_index.train(tensor.cpu().numpy())
            knn_index.nprobe = 10  # Set nprobe for IVF index

        knn_index.add(tensor.cpu().numpy())  # Add the tensor to the index
        return knn_index


    knn_index = create_faiss_index(tensor2, device, use_inverse_index, n_list)


    # Initialize the indices and values for the sparse tensor
    indices = torch.empty((2, 0), dtype=torch.long)
    values = torch.empty((0,), dtype=torch.float)

    for i in range(0, tensor1.shape[0], batch_size):
        chunk = tensor1[i:i + batch_size]
        idx, sim = compute_similarity_chunk(chunk, knn_index, num_neighbors, 
                                         metric, kw, scaling)
        
        # Compute row indices for the current chunk
        row_indices = torch.arange(i, i + chunk.size(0)).unsqueeze(0).repeat(num_neighbors, 1).transpose(0, 1).reshape(-1)
        col_indices = torch.tensor(idx).reshape(-1)  # Flatten the indices
        
        # Stack row and column indices to match the required shape for sparse tensors
        chunk_indices = torch.vstack((row_indices, col_indices))
        
        # Concatenate the new indices and values
        indices = torch.cat((indices, chunk_indices), dim=1)
        values = torch.cat((values, sim.flatten()))

    # Create the sparse tensor
    size = (tensor1.shape[0], tensor2.shape[0])
    similarity = torch.sparse_coo_tensor(indices, values, size)
    return similarity

def encode(prompts, references, embedding_model_name='BAAI/bge-large-en-v1.5'):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    tokenizer.max_subtokens_sequence_length = 512
    tokenizer.model_max_length = 512
    model = AutoModel.from_pretrained(embedding_model_name).to('cuda')
    model.eval()

    data_to_encode = [p + " " + r for p, r in zip(prompts, references)]

    encoded_input = tokenizer(data_to_encode, padding=True, truncation=True, return_tensors='pt').to(model.device)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
            
    bs = 16
    output = []
    for i in range(0, len(encoded_input['input_ids']), bs):
        output.extend(model(input_ids=input_ids[i:i+bs], attention_mask=attention_mask[i:i+bs]).pooler_output.detach())

    output = torch.stack(output)
    del tokenizer, model
    return output

class ModelIndependentICLUtility:
    def __init__(self):
        pass

    def calculate_se_utility(self, data1_prompts, data1_references, data2_prompts=None, data2_references=None, metric='cosine', 
                        batch_size=10000, scaling=None, kw=0.1, device='cuda'):
        
        tensor1 = encode(data1_prompts, data1_references)
        tensor2 = None if data2_prompts is None else encode(data2_prompts, data2_references)
        return self.compute_pairwise_similarities(tensor1, tensor2, metric, batch_size, scaling, kw, device)
    
    def compute_pairwise_similarities(self, tensor1, tensor2=None, metric='cosine', 
                        batch_size=10000, scaling=None, kw=0.1, device='cuda'):
        """
        Compute pairwise similarities between rows of two arrays, either using dense or sparse representation.

        Parameters:
        tensor1 (torch.Tensor): First matrix.
        tensor2 (torch.Tensor, optional): Second matrix. If None, uses array1.
        sparse (bool): If True, use sparse computation. Otherwise, use dense computation.
        m_neighbors (int): Number of neighbors (used in sparse computation).
        metric (str): The metric to use ('cosine', 'dot', 'euclidean').
        batch_size (int): Size of each batch for dense computation.
        scaling (str, optional): Type of scaling to apply in dense computation.
        kw (float, optional): Kernel width for rbf metric.
        n_list (int, optional): Number of list for Faiss Inverse Index Building.
        device (str): Device to perform computation ('cuda' or 'cpu').

        Returns:
        np.ndarray or csr_matrix: Matrix representing pairwise similarities.
        """
    
        assert batch_size > 0, "Batch size must be positive."

        if tensor2 is None:
            tensor2 = tensor1

        tensor1, tensor2 = tensor1.to(device), tensor2.to(device)

        n_samples1, n_samples2 = tensor1.size(0), tensor2.size(0)

        # Initialize a results matrix in the CPU memory to save GPU memory
        results = torch.zeros(n_samples1, n_samples2, device='cpu')

        # Normalizing tensors if metric is cosine for cosine similarity computation
        if metric == 'cosine':
            tensor1, tensor2 = F.normalize(tensor1, p=2, dim=1), F.normalize(tensor2, p=2, dim=1)

        # Function to calculate the metric
        def calculate_metric(a, b, metric):
            if metric in ['cosine', 'dot']:
                return torch.mm(a, b.T)
            elif metric == 'euclidean':
                return torch.cdist(a, b, p=2)
            elif metric == 'rbf':
                distance = torch.cdist(a, b)
                squared_distance = distance ** 2
                avg_dist = torch.mean(squared_distance)
                torch.div(squared_distance, kw*avg_dist, out=squared_distance)
                torch.exp(-squared_distance, out=squared_distance)
                return squared_distance
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Process in batches to manage memory usage
        for i in range(0, n_samples1, batch_size):
            end_i = min(i + batch_size, n_samples1)
            rows = tensor1[i:end_i]

            for j in range(0, n_samples2, batch_size):
                end_j = min(j + batch_size, n_samples2)
                cols = tensor2[j:end_j]

                # Compute metric for the current batch and store results on CPU
                batch_results = calculate_metric(rows, cols, metric).to('cpu')
                results[i:end_i, j:end_j] = batch_results

        # Apply scaling if specified
        if scaling == 'min-max':
            min_val, max_val = results.min(), results.max()
            if max_val != min_val:
                results = (results - min_val) / (max_val - min_val)
        elif scaling == 'additive':
            results = (results + 1) / 2

        return results


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--existing_data_name', type=str)
    argparser.add_argument('--new_data_name', type=str)
    argparser.add_argument('--length', type=int)
    argparser.add_argument('--between', type=str)
    args = argparser.parse_args()

    data = Data(args.existing_data_name, args.new_data_name, args.length)

    mod_dep = ModelIndependentICLUtility()
    if "data" in args.between:
        sijs = mod_dep.calculate_se_utility(data.existing_prompts, data.existing_references)
        constant = DATA_CONSTANT
    elif "pairwise" in args.between:
        sijs = mod_dep.calculate_se_utility(data.existing_prompts, data.existing_references, data.new_prompts, data.new_references)
        constant = PAIRWISE_CONSTANT
    
    with open(utility_file(SE_CONSTANT, data.dataset, constant), 'wb+') as f:
        pickle.dump(sijs, f)