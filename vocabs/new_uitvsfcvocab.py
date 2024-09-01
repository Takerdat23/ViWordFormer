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
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token

        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_idx = (0, 0, 0, 0, 0)
        self.cls_idx = (1, 1, 1, 1, 1)
        self.eos_idx = (2, 2, 2, 2, 2)
        self.unk_idx = (3, 3, 3, 3, 3)


    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter_am_dau = Counter()
        counter_am_dem = Counter()
        counter_tone = Counter()
        counter_am_chinh = Counter()
        counter_am_cuoi = Counter()
        labels = set()

        # Set of special tokens for easy checking
        special_tokens_set = set(self.specials)

        for json_dir in json_dirs:
            data = pd.read_csv(json_dir)
            for _, item in data.iterrows():
                tokens = preprocess_sentence(item["sentence"])
                for token in tokens:
                    am_dau, tone, am_dem, am_chinh, am_cuoi = self.split_vietnamese_word(token)
                    
                    # Ensure the token is not a special token
                    if am_dau not in self.specials:
                        counter_am_dau.update([am_dau])
                    if tone not in self.specials:
                        counter_tone.update([tone])
                    if am_dem not in self.specials:
                        counter_am_dem.update([am_dem])
                    if am_chinh not in self.specials:
                        counter_am_chinh.update([am_chinh])
                    if am_cuoi not in self.specials:
                        counter_am_cuoi.update([am_cuoi])
                
                labels.add(item["topic"])

        min_freq = max(config.min_freq, 1)
        
        # Sort by frequency and alphabetically, and filter by min frequency
        sorted_am_dau = sorted([item for item in counter_am_dau if counter_am_dau[item] >= min_freq])
        sorted_tone = sorted([item for item in counter_tone if counter_tone[item] >= min_freq])
        sorted_am_dem = sorted([item for item in counter_am_dem if counter_am_dem[item] >= min_freq])
        sorted_am_chinh = sorted([item for item in counter_am_chinh if counter_am_chinh[item] >= min_freq])
        sorted_am_cuoi = sorted([item for item in counter_am_cuoi if counter_am_cuoi[item] >= min_freq])

        # Add special tokens only once at the start of each vocabulary list
        self.itos_am_dau = {i: tok for i, tok in enumerate(self.specials + sorted_am_dau)}
        self.stoi_am_dau = {tok: i for i, tok in enumerate(self.specials + sorted_am_dau)}

        self.itos_am_dem = {i: tok for i, tok in enumerate(self.specials + sorted_am_dem)}
        self.stoi_am_dem = {tok: i for i, tok in enumerate(self.specials + sorted_am_dem)}

        self.itos_am_chinh = {i: tok for i, tok in enumerate(self.specials + sorted_am_chinh)}
        self.stoi_am_chinh = {tok: i for i, tok in enumerate(self.specials + sorted_am_chinh)}

        self.itos_am_cuoi = {i: tok for i, tok in enumerate(self.specials + sorted_am_cuoi)}
        self.stoi_am_cuoi = {tok: i for i, tok in enumerate(self.specials + sorted_am_cuoi)}

        self.itos_tone = {i: tok for i, tok in enumerate(self.specials + sorted_tone)}
        self.stoi_tone = {tok: i for i, tok in enumerate(self.specials + sorted_tone)}

        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

    @property
    def total_tokens(self) -> int:
        return len(self.itos_am_chinh)
    
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
    # Open the file in write mode, creating it if it doesn't exist
        with open("vocab_info.txt", "w", encoding="utf-8") as file:
            # Write Âm đầu details
            file.write("Âm đầu\n")
            file.write(f"self.itos_am_dau: {self.itos_am_dau}\n")
            file.write(f"length: {len(self.itos_am_dau)}\n\n")
            
            # Write Âm chính details
            file.write("Âm chính\n")
            file.write(f"self.itos_am_chinh: {self.itos_am_chinh}\n")
            file.write(f"length: {len(self.itos_am_chinh)}\n\n")
            
            # Write Âm đệm details
            file.write("Âm đệm\n")
            file.write(f"self.itos_am_dem: {self.itos_am_dem}\n")
            file.write(f"length: {len(self.itos_am_dem)}\n\n")
            
            # Write Âm cuối details
            file.write("Âm cuối\n")
            file.write(f"self.itos_am_cuoi: {self.itos_am_cuoi}\n")
            file.write(f"length: {len(self.itos_am_cuoi)}\n\n")
            
            # Write Thanh điệu details
            file.write("Thanh điệu\n")
            file.write(f"self.itos_tone: {self.itos_tone}\n")
            file.write(f"length: {len(self.itos_tone)}\n\n")
        
        print("Vocabulary details have been written to vocab_info.txt")
