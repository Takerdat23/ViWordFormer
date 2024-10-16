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
from .word_decomposation import is_Vietnamese, split_non_vietnamese_word



@META_VOCAB.register()
class UIT_ViSFD_newVocab_ABSA(ViPherVocab):
    
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
    
        aspects = set()
        sentiments = set()

        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for key in data:
              
                tokens = preprocess_sentence(data[key]["comment"])
                for token in tokens:
                    isVietnamese, wordsplit = is_Vietnamese(token)
                    if isVietnamese:
                        if token not in self.vietnamese:
                            self.vietnamese.append(token)
                            
                        onset, medial, nucleus, coda, tone = wordsplit
                        
                        if onset is None:
                            onset ='' 
                        if medial is None:
                            medial ='' 
                        if nucleus is None:
                            nucleus ='' 
                        if coda is None:
                           coda ='' 
                        if tone is None:
                            tone ='' 
                   
                        rhyme = ''.join([part for part in [medial, nucleus, coda] if part is not None])
                   

               
             
                    
                    else:
                        # Handle non-Vietnamese words by splitting into characters
                        if token not in self.nonvietnamese:
                            self.nonvietnamese.append(token)
                            
                        for char in token:
                            onset, tone, rhyme = split_non_vietnamese_word(char)
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
                
                
                for label in data[key]["label"]: 
                    aspects.add(label['aspect'])
                    sentiments.add(label['sentiment'])
                 

        min_freq = max(config.min_freq, 1)
        
        # Sort by frequency and alphabetically, and filter by min frequency
        sorted_onset = sorted(counter_onset)
        sorted_tone = sorted(counter_tone)
        sorted_rhyme = sorted(counter_rhyme)

        # Add special tokens only once at the start of each vocabulary list
        self.itos_onset = {i: tok for i, tok in enumerate(self.specials + sorted_onset)}
        self.stoi_onset = {tok: i for i, tok in enumerate(self.specials + sorted_onset)}

        self.itos_rhyme = {i: tok for i, tok in enumerate(self.specials + sorted_rhyme)}
        self.stoi_rhyme = {tok: i for i, tok in enumerate(self.specials + sorted_rhyme)}

        self.itos_tone = {i: tok for i, tok in enumerate(self.specials + sorted_tone)}
        self.stoi_tone = {tok: i for i, tok in enumerate(self.specials + sorted_tone)}

        aspects = list(aspects)
        sentiments = list(sentiments)
        self.i2a = {i: label for i, label in enumerate(aspects)}
        self.a2i = {label: i for i, label in enumerate(aspects)}
        self.i2s = {i: label for i, label in enumerate(sentiments, 1)}
        self.s2i = {label: i for i, label in enumerate(sentiments, 1)}
        

    @property
    def total_tokens(self) -> int:
        return len(self.itos_rhyme)
    
    @property
    def total_labels(self) -> int:
        return {
                "aspects" : len(self.i2a), 
                "sentiment": len(self.i2s)
               }


    def encode_label(self, labels: list) -> torch.Tensor:
        label_vector = torch.zeros(self.total_labels.aspects)
        for label in labels: 
            aspect = label['aspect']
            sentiment = label['sentiment']
            if sentiment is not None: 
                label_vector[aspect] = self.s2i[sentiment]  
        
        return torch.Tensor(label_vector).long()
           
    
    def decode_label(self, label_vecs: torch.Tensor) -> List[List[str]]:
        """
        Decodes the sentiment label vectors for a batch of instances.
        
        Args:
            label_vecs (torch.Tensor): Tensor of shape (bs, num_aspects) containing sentiment labels.
        
        Returns:
            List[List[str]]: A list of decoded labels (aspect -> sentiment) for each instance in the batch.
        """
        batch_decoded_labels = []
        
        # Iterate over each label vector in the batch
        for vec in label_vecs:
            instance_labels = []
            
            # Iterate over each aspect's sentiment value in the label vector
            for label_id in vec:
                label_id = label_id.item()  # Get the integer value of the label
                if label_id == 0 : 
                    continue
                decoded_label = self.i2s.get(label_id)  
                instance_labels.append(decoded_label)
            
            batch_decoded_labels.append(instance_labels)
        
        return batch_decoded_labels
    
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
            
            file.write(f"labels: {self.i2l}\n")
            file.write(f"length: {len(self.i2l)}\n\n")
            
            
           
        print("Vocabulary details have been written to vocab_info.txt")
