import unicodedata
import torch
import json
from collections import Counter
from typing import List
import torch
import pandas as pd 

from vocabs.base_newVocab import NewVocab
from vocabs.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB


@META_VOCAB.register()
class UIT_VSFC_newVocab(NewVocab):
    
    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.cls_token = config.cls_token
        self.unk_token = config.unk_token

        self.specials = [self.pad_token, self.cls_token, self.unk_token]

        self.pad_idx = (0, 0, 0)
        self.cls_idx = (1, 1, 1)
        self.unk_idx = (2, 2, 2)

    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter_am_dau = Counter()
        counter_van = Counter()
        counter_tone = Counter()
        labels = set()

        for json_dir in json_dirs:
            data = pd.read_csv(json_dir)
            for _, item in data.iterrows():
                tokens = preprocess_sentence(item["sentence"])
                for token in tokens:
                    am_dau, tone, van = self.split_vietnamese_word(token)
                    counter_am_dau.update([am_dau])
                    counter_tone.update([tone])
                    counter_van.update([van])
                labels.add(item["topic"])
    
        min_freq = max(config.min_freq, 1)

        # Sort by frequency and alphabetically for âm đầu, tone, and vần
        sorted_am_dau = sorted([item for item in counter_am_dau if counter_am_dau[item] >= min_freq])
        sorted_van = sorted([item for item in counter_van if counter_van[item] >= min_freq])
        sorted_tone = sorted([item for item in counter_tone if counter_tone[item] >= min_freq])

        self.itos_am_dau = {i: tok for i, tok in enumerate(self.specials + sorted_am_dau)}
        self.stoi_am_dau = {tok: i for i, tok in enumerate(self.specials + sorted_am_dau)}

        self.itos_van = {i: tok for i, tok in enumerate(self.specials + sorted_van)}
        self.stoi_van = {tok: i for i, tok in enumerate(self.specials + sorted_van)}

        self.itos_tone = {i: tok for i, tok in enumerate(self.specials + sorted_tone)}
        self.stoi_tone = {tok: i for i, tok in enumerate(self.specials + sorted_tone)}

        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}
    

    @property
    def total_tokens(self) -> int:
        return len(self.itos_van)
    
    @property
    def total_labels(self) -> int:
        return len(self.l2i)


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
    
    def Printing_test(self): 
        print("self.itos_am_dau", self.itos_am_dau)
        print("self.itos_van", self.itos_van)
        print("self.itos_tone", self.itos_tone)