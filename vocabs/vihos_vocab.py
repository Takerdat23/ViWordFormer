import torch
import json
from collections import Counter
from typing import List
import torch
from vocabs.vocab import Vocab
from vocabs.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB

@META_VOCAB.register()
class ViHOS(Vocab):
    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.unk_token = config.unk_token

        self.specials = [self.pad_token, self.unk_token]

        self.pad_idx = 0
        self.unk_idx = 1

    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter = Counter()
        labels = set()
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for key in data:
                tokens = preprocess_sentence(data[key]["review"])
                counter.update(tokens)
                labels.add(data[key]["label"])
    
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
        return len(self.itos_rhyme)
    
    @property
    def total_labels(self) -> int:
        return 2

    def encode_label(self, text: str, indices: list) -> torch.Tensor:
        
        toxic_indices = set(index for span in indices for index in span)
        label = [1 if i in toxic_indices else 0 for i in range(len(text.split()))]
 
        return torch.Tensor([label]).long()
    
    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        """
        label_vecs: (bs)
        """
        
        toxic_indices = []
        current_span = []

        for i, label in enumerate(label_vecs):
            if label == 1:
                # Start or continue a toxic span
                current_span.append(i)
            else:
                # If we hit a non-toxic label, save the current span if it exists
                if current_span:
                    toxic_indices.append(current_span)
                    current_span = []

        # Append any remaining toxic span
        if current_span:
            toxic_indices.append(current_span)
        
        return toxic_indices
