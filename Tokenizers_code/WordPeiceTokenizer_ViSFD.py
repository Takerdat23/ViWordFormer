import os
import torch
import json
from collections import Counter
from typing import List
from tokenizers import Tokenizer, models, trainers, processors
from tokenizers.pre_tokenizers import Whitespace
from builders.vocab_builder import META_VOCAB
from vocabs.utils import preprocess_sentence


@META_VOCAB.register()
class WordPieceTokenizer_ViSFD(object):
    def __init__(self, config):
        self.model_prefix = config.model_prefix
        self.tokenizer = None
        self.corpus = []
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token

        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        self.make_vocab(config)

    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        aspects = set()
        sentiments = set()
        words_counter = Counter()        
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for item in data:
              
                words_split = preprocess_sentence(item["comment"])
             
                words_counter.update(words_split)
                tokens = item["comment"]
                
                self.corpus.append(tokens)
                
                
                for label in item["label"]: 
                    aspects.add(label['aspect'])
                    sentiments.add(label['sentiment'])
        self.vocab_size =len(list(words_counter.keys()))
        self.train()

        vocab_file = f"{self.model_prefix}-vocab.json"
        
        self.vocab = list(json.load(open(vocab_file, "r")).keys())

        aspects = list(aspects)
        self.i2a = {i: label for i, label in enumerate(aspects)}
        self.a2i = {label: i for i, label in enumerate(aspects)}
        
        sentiments = list(sentiments)
        self.i2s = {i: label for i, label in enumerate(sentiments, 1)}
        self.i2s[0] = None
        self.s2i = {label: i for i, label in enumerate(sentiments, 1)}
        self.s2i[None] = 0

    def train(self):
        model_file = f"{self.model_prefix}-tokenizer.json"
        if not os.path.exists(model_file):
            # Define the WordPiece tokenizer
            tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))
            tokenizer.pre_tokenizer = Whitespace()

            trainer = trainers.WordPieceTrainer(
                vocab_size=self.vocab_size,
                special_tokens=self.specials,
            )

            # Train the tokenizer
            tokenizer.train_from_iterator(self.corpus, trainer)
            tokenizer.save(model_file)

            # Save vocab file for lookup
            vocab_file = f"{self.model_prefix}-vocab.json"
            with open(vocab_file, "w", encoding="utf-8") as f:
                json.dump(tokenizer.get_vocab(), f, indent=4)

            self.tokenizer = tokenizer
            print(f"Tokenizer model saved to {model_file}")
        else:
            print(f"Tokenizer model already exists at {model_file}")
            self.load_model()

    def load_model(self):
        model_file = f"{self.model_prefix}-tokenizer.json"
        if os.path.exists(model_file):
            self.tokenizer = Tokenizer.from_file(model_file)
            print(f"Model {model_file} loaded successfully.")
        else:
            print(f"Model {model_file} not found. Train the model first.")

    def encode_sentence(self, text, max_len=None, pad_token_id=0):
        if not self.tokenizer:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        encoding = self.tokenizer.encode(text)
        input_ids = encoding.ids

        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids.extend([pad_token_id] * (max_len - len(input_ids)))

        return torch.tensor(input_ids)

    def decode_sentence(self, input_ids):
        if not self.tokenizer:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()


        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def tokenize(self, text, max_len=None, pad_token_id=0):
        if not self.tokenizer:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        encoding = self.tokenizer.encode(text)
        tokens = encoding.tokens
        input_ids = encoding.ids

        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids.extend([pad_token_id] * (max_len - len(input_ids)))

        return {"tokens": tokens, "input_ids": input_ids}
    @property
    def total_tokens(self) -> int:
        return self.vocab_size

    @property
    def total_labels(self) -> dict:
        return {
                "aspects" : len(self.i2a), 
                "sentiment": len(self.i2s)
        }
    

    def get_aspects_label(self) -> list:
       
        return list(self.a2i.keys() )
    
    
    def encode_label(self, labels: list) -> torch.Tensor:
        label_vector = torch.zeros(self.total_labels["aspects"])
        for label in labels: 
            aspect = label['aspect']
            sentiment = label['sentiment']
            # active the OTHERS case
            if aspect == "OTHERS":
                sentiment = "Positive"
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
    def Printing_test(self):
        with open("vocab_info.txt", "w", encoding="utf-8") as file:
            file.write(f"Vocab size: {len(self.vocab)}\n\n")
            file.write(f"Vocab: {list(self.vocab)}\n\n")
            file.write(f"Labels: {len(self.l2i)}\n\n")
            file.write(f"Labels: {self.l2i}\n\n")

        print("Vocabulary details have been written to vocab_info.txt")
