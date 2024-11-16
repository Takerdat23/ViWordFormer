import torch
import json
from collections import Counter
from typing import List
import torch
from vocabs.vocab import Vocab
from vocabs.utils import preprocess_sentence
from builders.vocab_builder import META_VOCAB

@META_VOCAB.register()
class UIT_ABSA_Vocab(Vocab):
    
    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.cls_token = config.cls_token
        self.unk_token = config.unk_token

        self.specials = [self.pad_token, self.cls_token, self.unk_token]

        self.pad_idx = 0
        self.cls_idx = 1
        self.unk_idx = 2

    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter = Counter()
        aspects = set()
        sentiments = set()
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for key in data:
                item = data[key]
                tokens = preprocess_sentence(item["sentence"])
                counter.update(tokens)
                for label in item["label"]: 
                    category = label["category"]
                    aspect = label["aspect"]
                    sentiment = label["sentiment"]
                    aspect = f"{category}#{aspect}"
                    
                    aspects.add(aspect)
                    sentiments.add(sentiment)

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

        aspects = list(aspects)
        self.i2a = {i: label for i, label in enumerate(aspects)}
        self.a2i = {label: i for i, label in enumerate(aspects)}
        
        sentiments = list(sentiments)
        self.i2s = {i: label for i, label in enumerate(sentiments)}
        self.s2i = {label: i for i, label in enumerate(sentiments)}

    @property
    def total_tokens(self) -> int:
        return len(self.itos)
    
  
    def total_labels(self) -> dict:
        return {
                "aspects" : len(self.i2a), 
                "sentiment": len(self.i2s)
               }
    

    def get_aspects_label(self) -> list:
       
        return list(self.a2i.keys() )
    
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length """
        sentence = preprocess_sentence(sentence)
        vec = [self.cls_idx] + [self.stoi[token] if token in self.stoi else self.unk_idx for token in sentence]
        vec = torch.Tensor(vec).long()

        return vec


    def encode_label(self, labels: list) -> torch.Tensor:
        label_vector = torch.zeros(self.total_labels()["aspects"])
        for label in labels: 
            category = label["category"]
            aspect = label["aspect"]
            sentiment = label["sentiment"]
            aspect = f"{category}#{aspect}"
            label_vector[self.a2i[aspect]] = self.s2i[sentiment]
        
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
            for i , label_id in enumerate(vec):
                label_id = label_id.item()  # Get the integer value of the label
                if label_id == 0: 
                    continue
                aspect = self.i2a.get(i)
                

                sentiment = self.i2s.get(label_id)  
                decoded_label = {"aspect": aspect, "sentiment": sentiment}
                instance_labels.append(decoded_label)
            
            batch_decoded_labels.append(instance_labels)
        
        return batch_decoded_labels
    