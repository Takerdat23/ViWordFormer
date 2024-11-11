import torch
from sklearn.metrics import f1_score

class F1_micro:
    def __str__(self) -> str:
        return "f1 micro"
    
    def compute(self, inputs: torch.Tensor, labels: torch.Tensor):
        score = f1_score(labels, inputs, average="micro")

        return score
