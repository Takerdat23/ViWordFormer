import torch
import json
from builders.vocab_builder import META_VOCAB
from typing import List
from vocabs.phobert_vocab import PhoBERT_Vocab

@META_VOCAB.register()
class PhoBERT_ViCTSD_Vocab_Toxic(PhoBERT_Vocab):
    def __init__(self, config):
        super().__init__(config)

        self.make_vocab(config)
    
    
    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        labels = set()
 
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for key in data:
                labels.add(data[key]["toxicity"])
    

        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

    
    

    @property
    def total_labels(self) -> int:
        return len(self.i2l)
    
    def encode_label(self, label: str) -> torch.Tensor:
        return torch.Tensor([self.l2i[label]]).long()
    
    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        """
        label_vecs: (bs)
        """
        labels = []
        for vec in label_vecs:
            label_id = vec.item()
            labels.append(self.i2l[label_id])

        return labels