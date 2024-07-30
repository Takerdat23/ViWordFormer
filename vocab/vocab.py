import torch

from collections import Counter
import json
from typing import List

class Vocab(object):
    """
        A base Vocab class that is used to create a vocabulary for one-sentence only as input datasets 
    """
    def __init__(self, config):
        self.padding_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token

        self.make_vocab([
            config.json_path.train,
            config.json_path.dev,
            config.json_path.test
        ])
        counter = self.freqs.copy()
    
        min_freq = max(config.MIN_FREQ, 1)

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        itos = specials
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq:
                break
            itos.append(word)

        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}

        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        self.padding_idx = self.stoi[self.padding_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]

    def make_vocab(self, json_dirs):
        pass

    def encode(self, sentence: List[str]) -> torch.Tensor:
        """ Turn a question into a vector of indices and a question length """
        vec = torch.ones(self.max_sentence_length).long() * self.padding_idx
        for i, token in enumerate([self.bos_token] + sentence + [self.eos_token]):
            vec[i] = self.stoi[token] if token in self.stoi else self.unk_idx
        return vec

    def decode(self, sentence_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            question_vecs: (bs, max_length)
        '''
        sentences = []
        for vec in sentence_vecs:
            question = " ".join([self.itos[idx] for idx in vec.tolist() if self.itos[idx] not in self.specials])
            if join_words:
                sentences.append(question)
            else:
                sentences.append(question.strip().split())

        return sentences

    def __eq__(self, other: "Vocab"):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos.values()) if sort else v.itos.values()
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
