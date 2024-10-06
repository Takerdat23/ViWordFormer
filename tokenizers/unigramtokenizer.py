import sentencepiece as spm
import os
import torch
import json

class UnigramTokenizer:
    def __init__(self, model_prefix='unigram_tokenizer', vocab_size=8000, model_type='unigram'):
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.sp = None
    
    def train(self, input_file):
        """
        Args:
            input_file (str): path to .json file containing the data
        """
        # Check if model already exists
        model_file = f"{self.model_prefix}.model"
        if not os.path.exists(model_file):
            # Read data from JSON file
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text_data = [item['review'] for _, item in data.items()]
            text_data = '\n'.join(text_data)
            
            # Write text data to a temporary file
            temp_file = f"{self.model_prefix}_temp.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(text_data)
            
            spm.SentencePieceTrainer.train(
                f'--input={temp_file} --model_prefix={self.model_prefix} --vocab_size={self.vocab_size} --model_type={self.model_type}')
            
            # Remove the temporary file
            os.remove(temp_file)
            
            print(f"Model trained and saved as {model_file}")
        else:
            print(f"Model already exists at {model_file}")
    
    def load_model(self):
        """Load the trained SentencePiece model."""
        model_file = f"{self.model_prefix}.model"
        if os.path.exists(model_file):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_file)
            print(f"Model {model_file} loaded successfully.")
        else:
            raise FileNotFoundError(f"Model {model_file} not found. Train the model first.")
    
    def encode(self, text, max_len=None, pad_token_id=0):
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

    def decode(self, input_ids):
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")
        
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        
        return self.sp.decode_ids(input_ids)

if __name__ == "__main__":
    tokenizer = UnigramTokenizer(model_prefix="viocd_unigram", vocab_size=2000)
    tokenizer.train(input_file="data/UIT-ViOCD/train.json")

    tokenizer.load_model()
    text = "a nhô a sê ô"
    input_ids = tokenizer.encode(text, max_len=20, pad_token_id=0)

    print("Encoded input as tensor:", input_ids)
    print("Shape of tensor:", input_ids.shape)
    decoded_text = tokenizer.decode(input_ids)
    print("Decoded text:", decoded_text)

