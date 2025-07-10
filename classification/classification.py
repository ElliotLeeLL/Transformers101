import torch
from datasets import load_dataset
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

from utils.custom_utils import evaluate_performance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset into local variable
data = load_dataset('rotten_tomatoes')

# Build a pipeline
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

pipeline = pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True,
    device=device
)

# Get results from the model
y_pred = []
for output in tqdm(pipeline(KeyDataset(data['test'], "text")), total=len(data['test'])):
    negative_score = output[0]['score']
    positive_score = output[2]['score']
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)

# Evaluate model performance
evaluate_performance(
    data['test']['label'], y_pred
)




