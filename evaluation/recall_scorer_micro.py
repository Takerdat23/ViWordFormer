import torch
from sklearn.metrics import recall_score

class Recall_micro:
    def __str__(self) -> str:
        return "recall"
    
    def compute(self, inputs: torch.Tensor, labels: torch.Tensor):
        score = recall_score(labels, inputs, average="micro").item()

        return score