import torch
from sklearn.metrics import precision_score

class Precision:
    def __str__(self) -> str:
        return "precision"
    
    def compute(self, inputs: torch.Tensor, labels: torch.Tensor):
        score = precision_score(labels, inputs, average="macro").item()

        return score