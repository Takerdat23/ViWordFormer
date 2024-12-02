import torch
import json
from builders.vocab_builder import META_VOCAB

from vocabs.phobert_vocab import PhoBERT_Vocab

@META_VOCAB.register()
class PhoBERT_PhoNER_Vocab(PhoBERT_Vocab):
    def __init__(self, config):
        super().__init__(config)

        self.make_vocab(config)

    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        labels = set()
        for json_dir in json_dirs:
            data = json.load(open(json_dir, encoding='utf-8'))
            for key in data:
                for label in data[key]["tags"]:
                    label = label.split("-")[-1]
                    labels.add(label)

        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

    @property
    def total_labels(self) -> int:
        return len(self.i2l)
    
    def encode_label(self, labels: list) -> torch.Tensor:
        
        labels = [self.l2i[label] for label in labels]
 
        return torch.Tensor(labels).long()
    
    def decode_label(self, label_vecs: torch.Tensor) -> list[str]:
        """
        label_vecs: (bs)
        """
        results = []
        batch_labels = label_vecs.tolist()
        for labels in batch_labels:
            result = []
            for label in labels:
                result.append(self.i2l[label])
            results.append(result)
        
        return results