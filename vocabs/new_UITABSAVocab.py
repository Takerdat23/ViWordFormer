import unicodedata
import torch
import json
from collections import Counter
from typing import List
import torch
import pandas as pd 
from tqdm import tqdm
from vocabs.viphervocab import ViPherVocab
from vocabs.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB
from .word_decomposation import split_word

@META_VOCAB.register()
class UIT_ABSA_newVocab(ViPherVocab):
    
    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
        self.space_token = config.space_token

        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token, self.space_token]
        
        self.pad_idx = (0, 0, 0)
        self.bos_idx = (1, 1, 1)
        self.eos_idx = (2, 2, 2)
        self.unk_idx = (3, 3, 3)
        self.space_idx = (4, 4 ,4)

        
        self.vietnamese = []
        self.nonvietnamese = []
    


    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter_onset = Counter()
        counter_tone = Counter()
        counter_rhyme = Counter()
    
        labels = set()

        for json_dir in json_dirs:
            data = pd.read_csv(json_dir)
            for _, item in tqdm(data.iterrows()):
                tokens = preprocess_sentence(item["review"])
                for token in tokens:
                    word_dict = split_word(token)
                    if word_dict['is_vietnamese']:
                        if token not in self.vietnamese:
                            self.vietnamese.append(token)
                            
                        onset = word_dict['onset']
                        tone = word_dict['tone']
                        rhyme = ''.join([word_dict['medial'], word_dict['nucleus'], word_dict['coda']])
                    
                    else:
                        # Handle non-Vietnamese words by splitting into characters
                        if token not in self.nonvietnamese:
                            self.nonvietnamese.append(token)
                        
                        for char in token:
                            onset, tone, rhyme = self.split_non_vietnamese_word(char)
                            # Ensure the token is not a special token
                            if onset not in self.specials:
                                counter_onset.update([onset])
                            if tone not in self.specials:
                                counter_tone.update([tone])
                            if rhyme not in self.specials:
                                counter_rhyme.update([rhyme])
                        continue  # Skip the rest of the loop for non-Vietnamese words
                        
                    # Process Vietnamese words
                    if onset not in self.specials:
                        counter_onset.update([onset])
                    if tone not in self.specials:
                        counter_tone.update([tone])
                    if rhyme not in self.specials:
                        counter_rhyme.update([rhyme])
                
                labels.add(item["label"])

        min_freq = max(config.min_freq, 1)
        
        # Sort by frequency and alphabetically, and filter by min frequency
        sorted_onset = sorted([item for item in counter_onset if counter_onset[item] >= min_freq])
        sorted_tone = sorted([item for item in counter_tone if counter_tone[item] >= min_freq])
        sorted_rhyme = sorted([item for item in counter_rhyme if counter_rhyme[item] >= min_freq])

        # Add special tokens only once at the start of each vocabulary list
        self.itos_onset = {i: tok for i, tok in enumerate(self.specials + sorted_onset)}
        self.stoi_onset = {tok: i for i, tok in enumerate(self.specials + sorted_onset)}

        self.itos_rhyme = {i: tok for i, tok in enumerate(self.specials + sorted_rhyme)}
        self.stoi_rhyme = {tok: i for i, tok in enumerate(self.specials + sorted_rhyme)}

        self.itos_tone = {i: tok for i, tok in enumerate(self.specials + sorted_tone)}
        self.stoi_tone = {tok: i for i, tok in enumerate(self.specials + sorted_tone)}

        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}
        

    @property
    def total_tokens(self) -> int:
        return len(self.itos_rhyme)
    
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
            file.write("Vocab\n")
            file.write(f"self.itos_onset: {self.itos_onset}\n")
            file.write(f"length: {len(self.itos_onset)}\n\n")
            file.write(f"self.itos_rhyme: {self.itos_rhyme}\n")
            file.write(f"length: {len(self.itos_rhyme)}\n\n")
            file.write(f"self.itos_tone: {self.itos_tone}\n")
            file.write(f"length: {len(self.itos_tone)}\n\n")
            file.write(f"self.vietnamese: {self.vietnamese}\n")
            file.write(f"length: {len(self.vietnamese)}\n\n")
            file.write(f"self.nonvietnamese: {self.nonvietnamese}\n")
            file.write(f"length: {len(self.nonvietnamese)}\n\n")
            
            
           
        print("Vocabulary details have been written to vocab_info.txt")
