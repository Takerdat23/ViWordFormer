import os
import json
import torch
from collections import Counter
from typing import List

from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Whitespace

from builders.vocab_builder import META_VOCAB
from .utils.utils import preprocess_sentence


@META_VOCAB.register()
class WordPieceTokenizer:
    """
    A WordPiece-based tokenizer that uses HuggingFace's `tokenizers` library.
    Handles label mapping, vocabulary building, and sequence encoding/decoding.
    """

    def __init__(self, config):
        """
        Initializes the WordPiece tokenizer.

        Args:
            config: A configuration object, which should contain:
                - model_prefix (str): Prefix for saving/loading the tokenizer files.
                - path.train, path.dev, path.test (str): Paths to JSON files with data.
                - pad_token, bos_token, eos_token, unk_token (str): Special tokens.
                - schema (int): Determines how vocab_size is set (1 or 2).
                - vocab_size (int, optional): If schema=1, use this for the vocab size.
        """
        # Model prefix to name the tokenizer files
        self.model_prefix = config.model_prefix

        # The underlying HF Tokenizer object will be stored here
        self.tokenizer = None

        # In-memory corpus of strings for training
        self.corpus = []
        self.config = config

        # Special tokens and their IDs
        self.unk_piece = config.unk_piece
        self.bos_piece = config.bos_piece
        self.eos_piece = config.eos_piece
        self.pad_piece = config.pad_piece

        self.unk_id = config.unk_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id
        
        self.specials = [self.pad_piece, self.bos_piece, self.eos_piece, self.unk_piece]

        # Label mappings
        self.i2l = {}
        self.l2i = {}

        # Python-level vocab list derived from the saved vocab JSON
        self.vocab = []

        # Build vocabulary (and train tokenizer if needed)
        self.make_vocab(config)

    def make_vocab(self, config):
        """
        Reads data from JSON files, preprocesses sentences, collects them into `self.corpus`,
        determines vocab size based on `schema`, then trains the tokenizer.
        Finally, loads the vocab JSON to populate Python-level vocab & label maps.
        """
        json_paths = [config.path.train, config.path.dev, config.path.test]
        words_counter = Counter()
        labels = set()
        aspects = set()
        sentiments = set()

        # Validate file paths
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")

        # Gather data
        for path in json_paths:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                if isinstance(item[config.text], list):
                    tokens = preprocess_sentence(" ".join(item[config.text]))
                    words_counter.update(tokens)
                    # Keep original text for training
                    self.corpus.append(" ".join(item[config.text]))
                else: 
                    tokens = preprocess_sentence(item[config.text])
                    words_counter.update(tokens)
                    # Keep original text for training
                    self.corpus.append(item[config.text])
                
                if self.config.get("task_type", None) == "seq_labeling":
                    for label in item[config.label]:
                        label = label.split("-")[-1]
                        labels.add(label)
                elif self.config.get("task_type", None) == "aspect_based":
                    for label in item["label"]: 
                        aspects.add(label['aspect'])
                        sentiments.add(label['sentiment'])
                else:
                    labels.add(item[config.label])

        # Decide how to set vocab_size
        if config.schema == 2:
            self.vocab_size = len(words_counter.keys())
        elif config.schema == 1:
            self.vocab_size = config.vocab_size
        else:
            raise ValueError(f"Unsupported schema setting: {config.schema}")

        # Train the tokenizer (only if it doesn't already exist)
        self.train()

        # Load the vocab from the JSON file (created during training)
        vocab_file = f"{self.model_prefix}-vocab.json"
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        with open(vocab_file, "r", encoding="utf-8") as vf:
            vocab_dict = json.load(vf)
            self.vocab = list(vocab_dict.keys())

        # Create label <-> index maps (sorted for consistent ordering)
        if self.config.get("task_type", None) == "aspect_based":

            aspects = sorted({a for a in aspects if a is not None}, key=lambda x: x.lower())
            sentiments = sorted({s for s in sentiments if s is not None}, key=lambda x: x.lower())

            self.i2a = {i: label for i, label in enumerate(aspects)}
            self.a2i = {label: i for i, label in enumerate(aspects)}

            self.i2s = {i: label for i, label in enumerate(sentiments, 1)}
            self.i2s[0] = None  

            self.s2i = {label: i for i, label in enumerate(sentiments, 1)}
            self.s2i[None] = 0 
        else:
            # Create label <-> index maps (sorted for consistent ordering)
            labels = sorted(list(labels))
            self.i2l = {i: label for i, label in enumerate(labels)}
            self.l2i = {label: i for i, label in enumerate(labels)}
        
        # Optionally, save these label mappings
        self.save_labels()
        
        # Build basic Python-level mappings (token <-> index).
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        self.stoi = {token: i for i, token in enumerate(self.vocab)}

    def train(self):
        """
        Trains a WordPiece tokenizer on self.corpus if the model file does not already exist.
        Saves both the tokenizer JSON and a vocab JSON.
        """
        model_file = f"{self.model_prefix}-tokenizer.json"
        if os.path.exists(model_file):
            print(f"Tokenizer model already exists at {model_file}")
            self.load_model()
            return

        # Initialize the WordPiece tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_piece))
        tokenizer.pre_tokenizer = Whitespace()

        # Define a WordPieceTrainer
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.specials
        )

        # Train from the in-memory corpus
        tokenizer.train_from_iterator(self.corpus, trainer)
        tokenizer.save(model_file)

        # Save the vocab separately for Python-level usage
        vocab_file = f"{self.model_prefix}-vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as vf:
            json.dump(tokenizer.get_vocab(), vf, indent=4, ensure_ascii=False)

        self.tokenizer = tokenizer
        print(f"Tokenizer model saved to {model_file}")


    def load_model(self):
        """
        Loads the trained tokenizer from disk.
        """
        model_file = f"{self.model_prefix}-tokenizer.json"
        if not os.path.exists(model_file):
            print(f"Model {model_file} not found. Train the model first.")
            return

        self.tokenizer = Tokenizer.from_file(model_file)
        print(f"Model {model_file} loaded successfully.")
    
    
    def save_labels(self):
        """
        Save the label dictionaries (i2l, l2i) into a JSON file.
        """
        labels_file = f"{self.model_prefix}_labels.json"
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump({"i2l": self.i2l, "l2i": self.l2i}, f, ensure_ascii=False)
        print(f"Labels saved to {labels_file}")
    
    
    def encode_sequence_labeling(self, text: str, max_len: int = None) -> (List[int], List[int]):
        """
        Tokenize a sentence and return input IDs with a mapping from words to subwords.

        Args:
            text (str): The input text to tokenize.

        Returns:
            A tuple containing:
            - List[int]: Token IDs for the entire text.
            - List[int]: A list mapping each subword token ID to its original word index.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        words = preprocess_sentence(text)  # Split text into words using your preprocess_sentence method
        input_ids = []
        word_to_subword_mapping = []

        for word_idx, word in enumerate(words):
            # Encode each word into subwords
            subword_ids = self.tokenizer.encode(word).ids
            input_ids.extend(subword_ids)
            word_to_subword_mapping.extend([word_idx] * len(subword_ids))
        
        
        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids.extend([self.pad_id] * (max_len - len(input_ids)))


        return torch.tensor(input_ids, dtype=torch.long), word_to_subword_mapping
    
    
    def align_labels_with_subwords(self, labels: List[str], word_to_subword_mapping: List[int]) -> List[str]:
        """
        Align word-level labels with subword tokens.

        Args:
            labels (List[str]): Word-level labels.
            word_to_subword_mapping (List[int]): Mapping from subword tokens to word indices.

        Returns:
            List[str]: Subword-level labels.
        """
        
        ###### Not yet finished #######
        subword_labels = []
        for subword_idx in word_to_subword_mapping:
            if subword_idx < len(labels):
                subword_labels.append(labels[subword_idx])
            else:
                print(f"Warning: subword_idx ({subword_idx}) exceeds labels length ({len(labels)})")
                subword_labels.append(labels[-1]) 
        return subword_labels
    
    

    def encode_sentence(self, text: str, max_len: int = None, pad_token_id: int = 0) -> torch.Tensor:
        """
        Encode a sentence into token IDs. Optionally truncate/pad to max_len.

        Args:
            text (str): The input text to encode.
            max_len (int, optional): Max sequence length.
            pad_token_id (int): The ID to use for padding (default 0).

        Returns:
            torch.Tensor of shape (sequence_length,).
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        encoding = self.tokenizer.encode(text)
        input_ids = encoding.ids

        # Truncate or pad
        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids += [pad_token_id] * (max_len - len(input_ids))

        return torch.tensor(input_ids, dtype=torch.long)

    def decode_sentence(self, input_ids) -> str:
        """
        Decode token IDs back into a string.

        Args:
            input_ids (List[int] or torch.Tensor): Token IDs.

        Returns:
            str: Decoded text (special tokens are skipped).
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        # Handle torch.Tensor
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        # The HF Tokenizer can decode lists of IDs directly
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def tokenize(self, text: str, max_len: int = None, pad_token_id: int = 0) -> dict:
        """
        Tokenize the input text into subwords and return both tokens and IDs.

        Args:
            text (str): The input text to tokenize.
            max_len (int, optional): Max sequence length for truncation/padding.
            pad_token_id (int): ID to use for padding.

        Returns:
            dict with:
                "tokens": List[str] of subword tokens
                "input_ids": List[int] of token IDs
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        encoding = self.tokenizer.encode(text)
        tokens = encoding.tokens
        input_ids = encoding.ids

        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids += [pad_token_id] * (max_len - len(input_ids))

        return {"tokens": tokens, "input_ids": input_ids}

    @property
    def total_labels(self) -> int:
        """
        Returns:
            int: Number of distinct labels in the dataset.
        """
        return len(self.l2i)
    
    @property
    def total_aspects_labels(self) -> dict:
        return {
                "aspects" : len(self.i2a), 
                "sentiment": len(self.i2s)
               }

    @property
    def total_tokens(self) -> int:
        """
        Returns:
            int: Total number of tokens in the Python-level vocab (from the saved vocab JSON).
        """
        return len(self.vocab)
    
    @property
    def get_pad_idx(self) -> int:
        """Get the ID of the padding token."""
        return self.pad_id

    def encode_label(self, label: str) -> torch.Tensor:
        """
        Convert a string label to an integer label ID.

        Args:
            label (str): The label to encode.

        Returns:
            A torch.Tensor with the label ID.
        """
        
        if self.config.get("task_type", None) == "seq_labeling":
            
            labels = [self.l2i[l] for l in label]
 
            return torch.Tensor(labels).long()
        elif self.config.get("task_type", None) == "aspect_based":
            
            label_vector = torch.zeros(self.total_aspects_labels["aspects"])
            for l in label: 
                aspect = l['aspect']
                sentiment = l['sentiment']
                # active the OTHERS case
                if aspect == "OTHERS":
                    sentiment = "Positive"
                label_vector[self.a2i[aspect]] = self.s2i[sentiment]
            
            return torch.Tensor(label_vector).long() 
        else:
            return torch.tensor([self.l2i[label]], dtype=torch.long)
        
        
    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        """
        Convert integer label IDs back to string labels.

        Args:
            label_vecs (torch.Tensor): A 1D tensor of label IDs.

        Returns:
            A list of string labels.
        """
        if self.config.get("task_type", None) == "seq_labeling":
            results = []
            batch_labels = label_vecs.tolist()  
            for labels in batch_labels:
                result = []
                for label in labels:
                    result.append(self.i2l[label])
                results.append(result)
            
            return results
        elif self.config.get("task_type", None) == "aspect_based":
            
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
        
        else:
            return [self.i2l[label_id.item()] for label_id in label_vecs]


    def Printing_test(self):
        """
        Write vocabulary and label info to a file for debugging.
        """
        with open("vocab_info.txt", "w", encoding="utf-8") as file:
            file.write(f"Vocab size: {len(self.vocab)}\n\n")
            file.write(f"Vocab: {self.vocab}\n\n")
            file.write(f"Label count: {len(self.l2i)}\n\n")
            file.write(f"Labels: {self.l2i}\n\n")

        print("Vocabulary details have been written to vocab_info.txt")
