import torch
import json
from collections import Counter, defaultdict
from builders.vocab_builder import META_VOCAB
from typing import List, Dict
from vocabs.utils import preprocess_sentence

@META_VOCAB.register()
class BPE_ViHOS(object):
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
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for key in data:
                words_split = preprocess_sentence(data[key]["content"])
                words_counter.update(words_split)
                tokens = data[key]["content"]
                self.corpus.append(tokens)
        
        self.vocab_size = len(list(words_counter.keys()))
        self.train()
     
    def train(self):
        """Train BPE tokenizer."""
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
            pair_freqs = self.compute_pair_freqs()
            best_pair = max(pair_freqs, key=pair_freqs.get, default=None)
            if not best_pair:
                break

            # Merge the most frequent pair
            self.splits = self.merge_pair(*best_pair)
            new_token = ''.join(best_pair)

            # Ensure no duplicate `</w>`
            if new_token.endswith("</w></w>"):
                new_token = new_token[:-4] + "</w>"

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
            "word_to_tokens_mapping": word_to_tokens_mapping
        }
    
    @property
    def total_labels(self) -> int:
        return 2
    
    @property
    def total_tokens(self) -> int:
        return len(self.token_to_id)
    

    def encode_sentence(self, text):
        """Encode text and return input IDs along with a word-token mapping."""
        tokenized_output = self.tokenize(text)

        return torch.Tensor(tokenized_output["input_ids"]).long(), tokenized_output["word_to_tokens_mapping"]

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
        with open("vocab_info.txt", "w", encoding="utf-8") as file:
            file.write(f"Vocab size: {len(self.token_to_id)}\n\n")
            file.write(f"Vocab: {self.token_to_id}\n\n")
        print("Vocabulary details have been written to vocab_info.txt")
