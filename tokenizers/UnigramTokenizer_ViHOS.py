import sentencepiece as spm
import os
import torch
import json
from collections import Counter, defaultdict
from builders.vocab_builder import META_VOCAB
from typing import List, Dict
from vocabs.utils import preprocess_sentence



@META_VOCAB.register()
class UnigramTokenizer_ViHOS(object):
    def __init__(self, config, model_type='unigram'):
        self.model_prefix = config.model_prefix
        self.model_type = model_type
        self.sp = None
        self.corpus= []
        self.vocab_size = 1282
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
        
        self.token_to_id = {}
        self.id_to_token = {}
       

        self.specials = [self.pad_token, self.bos_token,
                         self.eos_token, self.unk_token]

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        
        
        self.make_vocab(config)
        
    
    
     
    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        labels = set()
        words_counter = Counter()
        
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for key in data:
                words_split = preprocess_sentence(data[key]["content"])
          
                words_counter.update(words_split)
                
                sentence = data[key]["content"]
                self.corpus.append(sentence)
              
        self.vocab_size = 7403
        self.train()

        vocab_file = f"{self.model_prefix}.vocab"
        vocab = set()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                word = line.split()[0]
                vocab.add(word)


        self.vocab = list(vocab)

        self.itos = {i: token for i, token in enumerate(self.vocab)}
        self.stoi = {token: i for i, token in enumerate(self.vocab)}

    def train(self):
        """
        Args:
            input_file (str): path to .json file containing the data
        """
        # Check if model already exists
        model_file = f"{self.model_prefix}.model"
        if not os.path.exists(model_file):
            text_data = '\n'.join(self.corpus)
            # Write text data to a temporary file
            temp_file = f"{self.model_prefix}_temp.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(text_data)

            spm.SentencePieceTrainer.train(
                f'--input={temp_file} --model_prefix={self.model_prefix} --vocab_size={self.vocab_size} --model_type={self.model_type}')

            # Remove the temporary file
            os.remove(temp_file)
            
            self.load_model()

            print(f"Model trained and saved as {model_file}")
        else:
            
            print(f"Model already exists at {model_file}")
            self.load_model()

    def load_model(self):
        """Load the trained SentencePiece model."""
        model_file = f"{self.model_prefix}.model"
        if os.path.exists(model_file):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_file)
            print(f"Model {model_file} loaded successfully.")
        else:
            # raise FileNotFoundError(
            #     f"Model {model_file} not found. Train the model first.")
            print(f"Model {model_file} not found. Train the model first.")
    
    
    def map_word_to_tokens(self, sentence: str) -> List[List[int]]:
        """
        Map each word in a sentence to its corresponding SentencePiece tokens.
        Returns a list of lists where each sub-list contains token indices within the sentence for each word.
        """
        word_to_token_indices = []
        words = sentence.split()
        start_index = 0

        for word in words:
            tokens = self.sp.encode(word, out_type=str)
            token_indices = list(range(start_index, start_index + len(tokens)))
            word_to_token_indices.append(token_indices)
            start_index += len(tokens)

        return word_to_token_indices
    
    


    def encode_sentence(self, text, max_len=None, pad_token_id=0):
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")
        
        # Encode the text into token IDs
        # text = f"{self.bos_token} {text} {self.eos_token}"

        input_ids = self.sp.encode_as_ids(text)
        
        if max_len is not None:
            if len(input_ids) > max_len:
                # Truncate if too long
                input_ids = input_ids[:max_len]
            else:
                # Pad if too short
                input_ids.extend([pad_token_id] * (max_len - len(input_ids)))
                
        word_to_tokens_mapping = self.map_word_to_tokens(text)
        
        return torch.tensor(input_ids), word_to_tokens_mapping

    def decode_sentence(self, input_ids):
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")
        
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        # Decode the sentence from token IDs
        decoded_sentence = self.sp.decode_ids(input_ids)

        # # Remove the <bos> and <eos> tokens from the decoded sentence
        # if decoded_sentence.startswith(self.bos_token):
        #     decoded_sentence = decoded_sentence[len(self.bos_token):].strip()
        # if decoded_sentence.endswith(self.eos_token):
        #     decoded_sentence = decoded_sentence[:-len(self.eos_token)].strip()

        return decoded_sentence

    def tokenize(self, text,  max_len=None, pad_token_id=0):
        # Tokenize the input text using SentencePiece
        tokens = self.sp.encode(text, out_type=str)

        # Prepend <b> and append <e> for BOS and EOS
        tokens = [self.bos_token] + tokens + [self.eos_token]

        # Map tokens to input_ids, handling unknown tokens with <u>
        input_ids = [self.stoi.get(token, self.stoi[self.unk_token])
                     for token in tokens]
        
        if max_len is not None:
            if len(input_ids) > max_len:
                # Truncate if too long
                input_ids = input_ids[:max_len]
            else:
                # Pad if too short
                input_ids.extend([pad_token_id] * (max_len - len(input_ids)))

        return {"tokens": tokens,
                "input_ids": input_ids}
    
    @property
    def total_labels(self) -> int:
        return len(self.l2i)
    
    @property
    def total_tokens(self) -> int:
        return self.vocab_size
    
    
    def encode_label(self, text: str, indices: list) -> torch.Tensor:
        
        toxic_indices = set(index for span in indices for index in span)
        label = [1 if i in toxic_indices else 0 for i in range(len(text.split()))]
 
        return torch.Tensor([label]).long()
    
    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        """
        label_vecs: (bs)
        """
        
        toxic_indices = []
        current_span = []

        for i, label in enumerate(label_vecs):
            if label == 1:
                # Start or continue a toxic span
                current_span.append(i)
            else:
                # If we hit a non-toxic label, save the current span if it exists
                if current_span:
                    toxic_indices.append(current_span)
                    current_span = []

        # Append any remaining toxic span
        if current_span:
            toxic_indices.append(current_span)
        
        return toxic_indices
    
    
    def Printing_test(self): 
    # Open the file in write mode, creating it if it doesn't exist
        with open("vocab_info.txt", "w", encoding="utf-8") as file:
            # Write Âm đầu details
          
            file.write(f"Vocab size: {len(self.token_to_id)}\n\n")
            file.write(f"Vocab: {self.token_to_id}\n\n")
          
            
            
           
        print("Vocabulary details have been written to vocab_info.txt")

    

