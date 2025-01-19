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

        # Validate file paths
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")

        # Gather data
        for path in json_paths:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                # Preprocess the text into tokens
                tokens = preprocess_sentence(item[config.text])
                # Update word frequencies if schema=2 uses dynamic vocab size
                words_counter.update(tokens)
                # Rejoin tokens so the training corpus has consistent spacing
                self.corpus.append(" ".join(tokens))
                # Collect labels
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

        # Build basic Python-level mappings (token <-> index).
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        self.stoi = {token: i for i, token in enumerate(self.vocab)}

        # Build label <-> index maps
        labels = sorted(list(labels))
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

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
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

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
            torch.Tensor: [label_id].
        """
        return torch.tensor([self.l2i[label]], dtype=torch.long)

    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        """
        Convert integer label IDs back to string labels.

        Args:
            label_vecs (torch.Tensor): A 1D tensor of label IDs.

        Returns:
            List[str]: Decoded label strings.
        """
        labels = []
        for vec in label_vecs:
            label_id = vec.item()
            labels.append(self.i2l[label_id])
        return labels

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

