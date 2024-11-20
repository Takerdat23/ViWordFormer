import unicodedata
import torch
import json
from collections import Counter
from typing import List
import torch
import pandas as pd 
from vocabs.viphervocab import ViPherVocab
from vocabs.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB

from .word_decomposation import is_Vietnamese, split_non_vietnamese_word


@META_VOCAB.register()
class ViNLI_newVocab(ViPherVocab):
    
    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
        self.space_token = config.space_token
        self.seperate_token = config.sep_token

        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token, self.space_token, self.seperate_token]
        
        self.pad_idx = (0, 0, 0)
        self.bos_idx = (1, 1, 1)
        self.eos_idx = (2, 2, 2)
        self.unk_idx = (3, 3, 3)
        self.space_idx = (4, 4 ,4)
        self.sep_idx = (5, 5 ,5)
        
        self.nonvietnamese = []
        self.vietnamese = []
    


    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter_onset = Counter()
        counter_tone = Counter()
        counter_rhyme = Counter()
    
        labels = set()

        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for item in data:
                sentence1 =  data[item]["sentence_1"]
                sentence2 =  data[item]["sentence_2"]
                sentence = sentence1 + self.seperate_token + sentence2
                tokens = preprocess_sentence(sentence)
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
                
                labels.add(data[item]["label"])

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
    
    
    def encode_sentence(self, sentence1: str, sentence2: str) -> torch.Tensor:
        """Turn a sentence into a vector of triplets (onset, tone, rhyme)."""
        
        sentence =  " ".join([sentence1  , "<sep>" , sentence2])
        
        
        tokens = preprocess_sentence(sentence)
        

        vec = [self.bos_idx]

        for token in tokens:
            isVietnamese, wordsplit = is_Vietnamese(token)
            
            if isVietnamese:
              
                onset, medial, nucleus, coda, tone = wordsplit
                rhyme = ''.join(parts for parts in [medial, nucleus, coda] if parts is not None)
            else:
                vec.append(
                    (self.space_idx[0], self.space_idx[1], self.space_idx[2]))
                for char in token:
                    onset, tone, rhyme = split_non_vietnamese_word(char)

                    onset_idx = self.stoi_onset.get(onset, self.unk_idx[0])
                    tone_idx = self.stoi_tone.get(tone, self.unk_idx[1])
                    rhyme_idx = self.stoi_rhyme.get(rhyme, self.unk_idx[2])

                    vec.append((onset_idx, tone_idx, rhyme_idx))
                vec.append(
                    (self.space_idx[0], self.space_idx[1], self.space_idx[2]))
                continue

            onset_idx = self.stoi_onset.get(onset, self.unk_idx[0])
            tone_idx = self.stoi_tone.get(tone, self.unk_idx[1])
            rhyme_idx = self.stoi_rhyme.get(rhyme, self.unk_idx[2])

            vec.append((onset_idx, tone_idx, rhyme_idx))  # Append the triplet

        vec = vec + [self.eos_idx]
        return torch.Tensor(vec).long()
    def decode_sentence(self, encoded_sentence: torch.Tensor) -> str:
        """
        Decode a vector of triplets back into the original sentence.

        :param encoded_sentence: Tensor of shape (n_words, 3), where each triplet represents (onse index, tone index, rhyme index)
        :return: Decoded sentence as a string
        """
        
        tone_map =  {
            '\u0300': '<`>',
            '\u0301': '</>',
            '\u0303': '<~>',
            '\u0309': '<?>',
            '\u0323': '<.>',
        }
        words = []
        current_word = []
        i = 0  # Main loop index
      
        encoded_sentence = encoded_sentence[0]
        
        while i < len(encoded_sentence):

            onset_idx, tone_idx, rhyme_idx = encoded_sentence[i].tolist()

            # Decode the indices back to the actual components (onset, tone, rhyme)
            onset = self.itos_onset.get(onset_idx, "")
            tone = self.itos_tone.get(tone_idx, "")
            rhyme = self.itos_rhyme.get(rhyme_idx, "")
            
          

            tone_char = [k for k, v in tone_map.items() if v == tone]
            tone_char = tone_char[0] if tone_char else ""
          

            if onset == self.eos_token and tone == self.eos_token and rhyme == self.eos_token:

                words.append(self.eos_token)

            elif onset == self.bos_token and tone == self.bos_token and rhyme == self.bos_token:

                words.append(self.bos_token)
            
            elif onset == self.seperate_token and tone == self.seperate_token and rhyme == self.seperate_token:
            
                words.append(self.seperate_token)
           
            

            elif onset == self.space_token and tone == self.space_token and rhyme == self.space_token:
                # Handle non vietnamese words

                current_word = []
                i += 1

                while i < len(encoded_sentence):

                    onset_idx, tone_idx, rhyme_idx = encoded_sentence[i].tolist(
                    )

                    # Decode each character
                    onset = self.itos_onset.get(onset_idx, "")
                    tone = self.itos_tone.get(tone_idx, "")
                    rhyme = self.itos_rhyme.get(rhyme_idx, "")

                    # Stop at the next space token, indicating end of non-Vietnamese word
                    if onset == self.space_token and tone == self.space_token and rhyme == self.space_token:
                        i += 1
                        break

                    if onset != "":
                        current_word.append(onset)
                    else:
                        current_word.append(rhyme)

                    i += 1

                if current_word:

                    words.append("".join(current_word))
              
                current_word = []
                continue

            else:
                
                # Handle Vietnamese words
                word = self.recompose_word(onset, tone_char, rhyme)
                words.append(word)

            i += 1

        if current_word:
            words.append("".join(current_word))

        return ' '.join(words)

    
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
