import torch
import json
from collections import Counter, defaultdict
from builders.vocab_builder import META_VOCAB
from typing import List
from vocabs.utils import preprocess_sentence
import os
@META_VOCAB.register()
class BPE_ViSFD(object):
    """Byte-Pair Encoding: Subword-based tokenization algorithm for Vietnamese."""

    def __init__(self, config):
        """Initialize BPE tokenizer."""
        self.model_prefix = config.model_prefix
        self.model_type = model_type
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
       
        self.corpus = []
     
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}
        self.vocab = []
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.make_vocab(config)
    
    
    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
   
        words_counter = Counter()
        
        aspects = set()
        sentiments = set()
        
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for key in data:
              
                words_split = preprocess_sentence(data[key]["comment"])
             
                words_counter.update(words_split)
                tokens = data[key]["comment"]
                
                self.corpus.append(tokens)
                
                
                for label in data[key]["label"]: 
                    aspects.add(label['aspect'])
                    sentiments.add(label['sentiment'])
        self.vocab_size =len(list(words_counter.keys()))
        self.train()
        
        vocab_file = f"{self.model_prefix}.vocab"
        vocab = set()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                word = line.split()[0]
                vocab.add(word)
                
                
        
        aspects = list(aspects)
        self.i2a = {i: label for i, label in enumerate(aspects)}
        self.a2i = {label: i for i, label in enumerate(aspects)}
        
        sentiments = list(sentiments)
        self.i2s = {i: label for i, label in enumerate(sentiments, 1)}
        self.i2s[0] = None
        self.s2i = {label: i for i, label in enumerate(sentiments, 1)}
        self.s2i[None] = 0
        

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
    
    
    def get_vocab_size(self): 
        return len(self.token_to_id)

    
        
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
        
        return torch.tensor(input_ids)

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
    
    def total_labels(self) -> dict:
        return {
                "aspects" : len(self.i2a), 
                "sentiment": len(self.i2s)
               }
    

    def get_aspects_label(self) -> list:
       
        return list(self.a2i.keys() )
    
    
    def encode_label(self, labels: list) -> torch.Tensor:
        label_vector = torch.zeros(self.total_labels()["aspects"])
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
            
            file.write(f"Aspects: {self.i2a}\n")
            file.write(f"length: {len(self.i2a)}\n\n")
            file.write(f"Sentiments: {self.i2s}\n")
            file.write(f"length: {len(self.i2s)}\n\n")
            
            
           
        print("Vocabulary details have been written to vocab_info.txt")
