import torch
import json
from collections import Counter, defaultdict
from builders.vocab_builder import META_VOCAB
from typing import List
from vocabs.utils import preprocess_sentence

@META_VOCAB.register()
class BPE_Vietnamese_VNLI(object):
    """Byte-Pair Encoding: Subword-based tokenization algorithm for Vietnamese."""

    def __init__(self, config):
        """Initialize BPE tokenizer."""
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
        labels = set()
        words_counter = Counter()
        
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for item in data:
                
             
                context =  data[item]["context"]
                sentence1 =  data[item]["sentence_1"]
                sentence2 =  data[item]["sentence_2"]
                sentence = sentence1 + " " + sentence2
          
                
                
                words_split = preprocess_sentence(sentence)
          
                words_counter.update(words_split)
                tokens = sentence
                
                self.corpus.append(tokens)
                labels.add(item["label"])
        self.vocab_size =len(list(words_counter.keys()))
        self.train()
        
        vocab_file = f"{self.model_prefix}.vocab"
        vocab = set()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                word = line.split()[0]
                vocab.add(word)
        
        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

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

    def compute_pair_freqs(self):
        """Compute the frequency of each pair."""
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b):
        """Merge the given pair."""
        new_word = a + b
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [new_word] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits

    def build_vocab_dicts(self):
        """Build token-to-id and id-to-token dictionaries."""
        for i, token in enumerate(self.vocab):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        
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
    
    
    def Printing_test(self): 
    # Open the file in write mode, creating it if it doesn't exist
        with open("vocab_info.txt", "w", encoding="utf-8") as file:
            # Write Âm đầu details
          
            file.write(f"Vocab size: {len(self.token_to_id)}\n\n")
            file.write(f"Vocab: {self.token_to_id}\n\n")
            file.write(f"Labels: {len(self.l2i)}\n\n")
            file.write(f"Labels: {self.l2i}\n\n")
            
            
           
        print("Vocabulary details have been written to vocab_info.txt")
