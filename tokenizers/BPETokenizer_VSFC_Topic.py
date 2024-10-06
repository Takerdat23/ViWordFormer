import torch
import json
from collections import Counter, defaultdict
from builders.vocab_builder import META_VOCAB
from typing import List
from vocabs.utils import preprocess_sentence


@META_VOCAB.register()
class BPE_Vietnamese_VSFC_Topic(object):
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

    def tokenize(self, text):
        """Tokenize a given text with trained BPE tokenizer."""
        # Split input text into words
        pre_tokenized_text = text.split()
        splits_text = [[l for l in word] + ["</w>"] for word in pre_tokenized_text]

        # Prepend <b> and append <e> for BOS and EOS
        tokens = [self.bos_token]

        # Perform BPE merges based on the trained merges
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits_text[idx] = split

        tokens += sum(splits_text, [])
        tokens.append(self.eos_token)  # Add <e> for end of sentence

        # Map tokens to input_ids, handling unknown tokens with <u>
        input_ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]

        return {"tokens": tokens,
                "input_ids": input_ids}
        
    def encode_sentence(self, text):
        """Tokenize a given text with trained BPE tokenizer."""
        # Split input text into words
        pre_tokenized_text = text.split()
        splits_text = [[l for l in word] + ["</w>"] for word in pre_tokenized_text]

        # Prepend <b> and append <e> for BOS and EOS
        tokens = [self.bos_token]

        # Perform BPE merges based on the trained merges
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits_text[idx] = split

        tokens += sum(splits_text, [])
        tokens.append(self.eos_token)  # Add <e> for end of sentence

        # Map tokens to input_ids, handling unknown tokens with <u>
        input_ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]

        return torch.Tensor(input_ids).long()

    
    def decode_sentence(self, input_ids):
        """Decode input_ids back into the original sentence (as close as possible)."""
        # Convert input IDs back to tokens
        tokens = [self.id_to_token.get(i, self.unk_token) for i in input_ids]

        # Remove special tokens
        tokens = [token for token in tokens if token not in {self.eos_token, self.unk_token, self.pad_token, self.bos_token}]

        # Remove the </w> that represents word boundaries and join characters into words
        decoded_sentence = []
        word = ""

        for token in tokens:
            if token.endswith("</w>"):
                word += token[:-4]  # Remove the "</w>" and finish the word
                decoded_sentence.append(word)
                word = ""  # Reset for the next word
            else:
                word += token  # Accumulate characters

        return ' '.join(decoded_sentence)

              
    
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
