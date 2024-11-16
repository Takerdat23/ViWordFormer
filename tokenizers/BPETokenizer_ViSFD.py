import torch
import json
from collections import Counter, defaultdict
from builders.vocab_builder import META_VOCAB
from typing import List
from vocabs.utils import preprocess_sentence

@META_VOCAB.register()
class BPE_ViSFD(object):
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
        
        aspects = list(aspects)
        sentiments = list(sentiments)
        self.i2a = {i: label for i, label in enumerate(aspects, 0)}
        self.a2i = {label: i for i, label in enumerate(aspects, 0)}
        self.i2s = {i: label for i, label in enumerate(sentiments, 0)}
        self.i2s[-1] = "None"
        self.s2i = {label: i for i, label in enumerate(sentiments, 0)}
        self.s2i["None"] = -1

    def train(self):
        """Train BPE tokenizer."""

        # Count the frequency
        for text in self.corpus:
            for word in text.split():
                self.word_freqs[word] += 1

        # Compute the base vocabulary of all characters
        alphabet = set()
        for word in self.word_freqs.keys():
            alphabet.update(word)
        alphabet = sorted(alphabet)

        # Add special tokens to the vocabulary
        self.vocab = [self.eos_token, self.unk_token, self.pad_token, self.bos_token, "</w>"] + alphabet.copy()

        # Split each word into individual characters before training
        self.splits = {word: [c for c in word] + ["</w>"] for word in self.word_freqs.keys()}

        while len(self.vocab) < self.vocab_size:
            # Compute the frequency
            pair_freqs = self.compute_pair_freqs()

            # Find the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get, default=None)

            if not best_pair:
                break

            # Merge the most frequent pair
            self.splits = self.merge_pair(*best_pair)
            new_token = ''.join(best_pair) + "</w>"
            self.merges[best_pair] = new_token
            self.vocab.append(new_token)
        self.build_vocab_dicts()
        return self.merges
    
    
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
        """Merge the given pair while handling the `</w>` suffix properly."""
        # Create the merged token
        new_word = a + b

        # Ensure there is no duplicate `</w>` suffix
        if new_word.endswith("</w></w>"):
            new_word = new_word[:-4]  # Remove the extra `</w>`

        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    # Merge the pair and update the split
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

    def tokenize(self, text):
        """Tokenize a given text with trained BPE tokenizer and return a word-to-token mapping."""
        pre_tokenized_text = text.split()  # Initial word-level tokens
        splits_text = [[l for l in word] + ["</w>"] for word in pre_tokenized_text]
        tokens = [self.bos_token]
        word_to_tokens_mapping = []

        # Perform BPE merges based on the trained merges
        for idx, split in enumerate(splits_text):
            word = pre_tokenized_text[idx]
            word_to_tokens = []  # Initialize empty list for token indices
            for pair, merge in self.merges.items():
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
            # Add indices for the merged tokens
            for token in split:
                tokens.append(token)
                word_to_tokens.append(len(tokens) - 1)  # Track token index for the word
            
            word_to_tokens_mapping.append(word_to_tokens)

        tokens.append(self.eos_token)

        # Convert tokens to input IDs
        input_ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]

        return {
            "tokens": tokens,
            "input_ids": input_ids,
        }
        
    def encode_sentence(self, text):
        """Encode text and return input IDs along with a word-token mapping."""
        tokenized_output = self.tokenize(text)

        return torch.Tensor(tokenized_output["input_ids"]).long()

    def decode_sentence(self, input_ids):
        """Decode input IDs back into the original sentence (as close as possible)."""
    
        tokens = [self.id_to_token.get(int(i), self.unk_token) for i in input_ids]

        # Filter out special tokens
        filtered_tokens = [
            token for token in tokens
            if token not in {self.eos_token, self.unk_token, self.pad_token, self.bos_token}
        ]
        decoded_sentence = []
        word = ""
        for token in filtered_tokens:
            if token.endswith("</w>"):
                # Handle the end of a word correctly
                word += token[:-4]  # Remove `</w>`
                decoded_sentence.append(word)
                word = ""  # Reset for the next word
            else:
                word += token

        if word:
            decoded_sentence.append(word)

        return ' '.join(decoded_sentence)
              
    
    @property
    def total_labels(self) -> int:
        return len(self.l2i)
    
    @property
    def total_tokens(self) -> int:
        return len(self.token_to_id)
    
    def total_labels(self) -> dict:
        return {
                "aspects" : len(self.i2a), 
                "sentiment": len(self.i2s)
               }
    

    def get_aspects_label(self) -> list:
       
        return list(self.a2i.keys() )
    
    
    def encode_label(self, labels: list) -> torch.Tensor:
        label_vector = torch.full((self.total_labels()["aspects"],), -1, dtype=torch.long)

        for label in labels: 
            aspect = label['aspect']
            sentiment = label['sentiment']
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
        aspect_list = self.get_aspects_label()
        
        # Iterate over each label vector in the batch
        for vec in label_vecs:
            instance_labels = []
            
            # Iterate over each aspect's sentiment value in the label vector
            for i , label_id in enumerate(vec):
                label_id = label_id.item()  # Get the integer value of the label
                if label_id == -1: 
                    continue
                aspect = self.i2a.get(i)
                sentiment = self.i2s.get(label_id)  
                decoded_label = {"aspect": aspect, "sentiment": sent}
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
