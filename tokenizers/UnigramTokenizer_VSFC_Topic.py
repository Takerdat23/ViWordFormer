import sentencepiece as spm
import os
import torch
import json
from collections import Counter, defaultdict
from builders.vocab_builder import META_VOCAB
from typing import List
from vocabs.utils import preprocess_sentence
from utils.logging_utils import setup_logger



@META_VOCAB.register()
class UnigramTokenizer_VSFC_Topic(object):
    def __init__(self, config):
        self.logger = setup_logger()
        
        self.model_prefix = config.model_prefix
        self.model_type = config.model_type
        self.corpus = []
        self.sp = None
        self.load_model()
        self.make_vocab(config)
        
    
    
     
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
                labels.add(item["topic"])
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
        text_data = '\n'.join(self.corpus)  
        temp_file = f"{self.model_prefix}_temp.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(text_data)   
        spm.SentencePieceTrainer.train(
                f'--input={temp_file} --model_prefix={self.model_prefix} --vocab_size={self.vocab_size} --model_type={self.model_type}')
        self.load_model()
        os.remove(temp_file)
    
    def load_model(self):
        """Load the trained SentencePiece model."""
        model_file = f"{self.model_prefix}.model"
        if os.path.exists(model_file):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_file)
            self.logger.info(f"Model {model_file} loaded successfully.")
        else:
            self.logger.info(f"Model {model_file} not found. Train the model first.")
   

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
        return self.vocab_size
    
    
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

