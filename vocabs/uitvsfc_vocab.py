import torch

from collections import Counter
import json
from typing import List

from vocabs.vocab import Vocab
from vocabs.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB

@META_VOCAB.register()
class UIT_VSFC_Vocab(Vocab):
    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter = Counter()
        labels = set()
        for json_dir in json_dirs:
            data = json.load(open(json_dir))
            for item in data:
                tokens = preprocess_sentence(item["sentence"])
                counter.update(tokens)
                labels.add(item["topic"])
    
        min_freq = max(config.min_freq, 1)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        itos = []
        for word, freq in words_and_frequencies:
            if freq < min_freq:
                break
            itos.append(word)
        itos = self.specials + itos

        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        
        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}
    
    @property
    def total_tokens(self) -> int:
        return len(self.itos)

    def encode_label(self, label: str) -> torch.Tensor:
        return torch.Tensor([self.l2i[label]])
    
    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        '''
            label_vecs: (bs, 1)
        '''
        labels = []
        for vec in label_vecs:
            label_id = vec.item()
            labels.append(self.i2l[label_id])

        return labels
