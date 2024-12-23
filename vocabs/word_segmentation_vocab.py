import torch
import json
from collections import Counter
from typing import List
import torch
from vocabs.vocab import Vocab
from vocabs.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB

@META_VOCAB.register()
class WordSegmetationVocab(Vocab):
    
    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.cls_token = config.cls_token
        self.unk_token = config.unk_token

        self.specials = [self.pad_token, self.cls_token, self.unk_token]

        self.pad_idx = -100
        self.cls_idx = 1
        self.unk_idx = 2

    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter = Counter()
        labels = set()
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for key in data:
                tokens = data[key]["text"].split()
                counter.update(tokens)
                labels.update(data[key]["label"])
    
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

        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.stoi[self.pad_token] = -100
        self.itos = {i: tok for tok, i in self.stoi.items()}
        
        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}


    @property
    def total_tokens(self) -> int:
        return len(self.itos)
    
    @property
    def total_labels(self) -> int:
        return len(self.l2i)
    
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length """
        sentence = sentence.split()
        vec = [self.stoi[token] if token in self.stoi else self.unk_idx for token in sentence]
        vec = torch.Tensor(vec).long()

        return vec

    def encode_label(self, labels: list) -> torch.Tensor:
        
        labels = [self.l2i[label] for label in labels]
 
        return torch.Tensor(labels).long()
    
    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
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

