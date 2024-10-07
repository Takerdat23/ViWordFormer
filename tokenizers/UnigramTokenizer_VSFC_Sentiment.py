import sentencepiece as spm
import os
import torch
import json
from collections import Counter, defaultdict
from builders.vocab_builder import META_VOCAB
from typing import List
from vocabs.utils import preprocess_sentence



@META_VOCAB.register()
class UnigramTokenizer_VSFC_Sentiment(object):
    def __init__(self, config):
        
        self.model_prefix = config.model_prefix
        self.model_type = config.model_type
        self.corpus = []
        self.sp = None
    
    
     
    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        labels = set()
        words_counter = Counter()
        
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for item in data:
                words_split = preprocess_sentence(item["sentence"])
          
                words_counter.update(words_split)
                tokens = item["sentence"]
                
                self.corpus.append(tokens)
                labels.add(item["sentiment"])
        self.vocab_size =len(list(words_counter.keys()))
        self.train()
        
        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

    
    def train(self):
        """
        Args:
            input_file (str): path to .json file containing the data
        """
        # Check if model already exists
        
        text_data = '\n'.join(self.corpus)
            
        # Write text data to a temporary file
        temp_file = f"{self.model_prefix}_temp.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(text_data)
            
        spm.SentencePieceTrainer.train(
                f'--input={temp_file} --model_prefix={self.model_prefix} --vocab_size={self.vocab_size} --model_type={self.model_type}')
            
        # Remove the temporary file
        os.remove(temp_file)

    def encode_sentence(self, text, max_len=None, pad_token_id=0):
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")
        
        # Encode the text into token IDs
        input_ids = self.sp.encode_as_ids(text)
        
        if max_len is not None:
            if len(input_ids) > max_len:
                # Truncate if too long
                input_ids = input_ids[:max_len]
            else:
                # Pad if too short
                input_ids.extend([pad_token_id] * (max_len - len(input_ids)))
        
        return torch.tensor(input_ids)

    def decode_sentence(self, input_ids):
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")
        
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        
        return self.sp.decode_ids(input_ids)
    
    @property
    def total_labels(self) -> int:
        return len(self.l2i)
    
    @property
    def total_tokens(self) -> int:
        return len(self.token_to_id)
    
    
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

