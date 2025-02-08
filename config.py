# utility_file = lambda dataset, between: f"utility/{dataset}_{between}.pkl"

DATA_CONSTANT = 'data'
PAIRWISE_CONSTANT = 'pairwise'
ICL_CONSTANT = "icl"
SE_CONSTANT = "se"
SELECTIT_CONSTANT = "selectit"
RANDOM_CONSTANT = "random"
LESS_CONSTANT  = "less"
def utility_file(type, dataset, between):
    if between == DATA_CONSTANT:
        return f"utility/{type}_{dataset[:dataset.find('|')]}_{between}.pkl"
    return f"utility/{type}_{dataset}_{between}.pkl"

device='cuda'
data_influence_function_file = lambda dataset: f"influence_functions/{dataset}_data.pt"
pairwise_influence_function_file = lambda dataset: f"influence_functions/{dataset}_private.pt"

CONFIDENCE_CALCULATION_SHORTHANDS = ['min', 'max', 'mean', 'entropy']
raw_confidence_file = lambda dataset: f"utility/raw_confidence_{dataset}.pkl"
confidence_utility_file = lambda dataset, shorthand: f"utility/confidence_utility_{dataset}_{shorthand}.pkl"