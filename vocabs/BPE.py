import os
import json
import torch
import sentencepiece as spm

from typing import List
from collections import Counter
from builders.vocab_builder import META_VOCAB
from .utils.utils import preprocess_sentence


@META_VOCAB.register()
class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer for Vietnamese text using SentencePiece.
    """

    def __init__(self, config, model_type: str = 'bpe'):
        """
        Initialize the BPE tokenizer with the given configuration.
        
        Args:
            config: A configuration object containing paths, special tokens, etc.
            model_type (str): The SentencePiece model type (default is 'bpe').
        """
        self.model_prefix = config.model_prefix
        self.model_type = model_type

        # Special tokens and their IDs
        self.unk_piece = config.unk_piece
        self.bos_piece = config.bos_piece
        self.eos_piece = config.eos_piece
        self.pad_piece = config.pad_piece

        self.unk_id = config.unk_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id

        # Will hold your trained SentencePieceProcessor
        self.sp = None

        # A place to store text data for SentencePiece training
        self.corpus = []

        # Vocab size will be determined based on the config
        self.vocab_size = 0

        # Label mappings
        self.i2l = {}
        self.l2i = {}

        # Build vocabulary and train/load the model
        self.make_vocab(config)

    def make_vocab(self, config):
        """
        Gather text from JSON files, determine vocab size, and train the model if needed.
        """
        json_paths = [config.path.train, config.path.dev, config.path.test]
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")

        words_counter = Counter()
        labels = set()

        # Collect text data and labels
        for path in json_paths:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                tokens = preprocess_sentence(item[config.text])
                words_counter.update(tokens)

                # Keep original text for training
                self.corpus.append(item[config.text])
                labels.add(item[config.label])

        # Decide vocab size based on schema
        if config.schema == 2:
            self.vocab_size = len(words_counter.keys())
        elif config.schema == 1:
            self.vocab_size = config.vocab_size
        else:
            # If there's a chance of an unknown schema, handle or raise
            raise ValueError(f"Unsupported schema type: {config.schema}")

        # Train the SentencePiece model if needed
        self.train()

        # Create label <-> index maps (sorted for consistent ordering)
        labels = sorted(list(labels))
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

        # Optionally, save these label mappings
        self.save_labels()

    def train(self):
        """
        Train a SentencePiece model on the collected corpus if it doesn't already exist.
        """
        model_file = f"{self.model_prefix}.model"
        if os.path.exists(model_file):
            print(f"Model already exists at {model_file}")
            self.load_model()
            return

        # Write the corpus to a temporary file
        temp_file = f"{self.model_prefix}_temp.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.corpus))

        # Train the SentencePiece model
        cmd = " ".join([
            f"--input={temp_file}",
            f"--model_prefix={self.model_prefix}",
            f"--vocab_size={self.vocab_size}",
            f"--model_type={self.model_type}",
            f"--unk_id={self.unk_id}",
            f"--bos_id={self.bos_id}",
            f"--eos_id={self.eos_id}",
            f"--pad_id={self.pad_id}",
            f"--unk_piece={self.unk_piece}",
            f"--bos_piece={self.bos_piece}",
            f"--eos_piece={self.eos_piece}",
            f"--pad_piece={self.pad_piece}",
        ])
        spm.SentencePieceTrainer.train(cmd)

        # Clean up temp file
        os.remove(temp_file)
        self.load_model()
        print(f"Model trained and saved as {model_file}")

    def load_model(self):
        """
        Load the trained SentencePiece model from disk.
        """
        model_file = f"{self.model_prefix}.model"
        if not os.path.exists(model_file):
            print(f"Model {model_file} not found. Train the model first.")
            return

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        print(f"Model {model_file} loaded successfully.")

        # If you have a labels file, you can load it here as well
        self.load_labels()

    def save_labels(self):
        """
        Save the label dictionaries (i2l, l2i) into a JSON file.
        """
        labels_file = f"{self.model_prefix}_labels.json"
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump({"i2l": self.i2l, "l2i": self.l2i}, f, ensure_ascii=False)
        print(f"Labels saved to {labels_file}")

    def load_labels(self):
        """
        Load the label dictionaries (i2l, l2i) from a JSON file, if it exists.
        """
        labels_file = f"{self.model_prefix}_labels.json"
        if not os.path.exists(labels_file):
            print(f"Labels file not found: {labels_file}.")
            return

        with open(labels_file, "r", encoding="utf-8") as f:
            label_data = json.load(f)
            # Convert keys of i2l back to integers
            self.i2l = {int(k): v for k, v in label_data["i2l"].items()}
            self.l2i = label_data["l2i"]

        print(f"Labels loaded successfully from {labels_file}")

    def encode_sentence(self, text: str, max_len: int = None) -> torch.Tensor:
        """
        Encode a sentence into token IDs using SentencePiece. Truncate/pad if max_len is given.

        Args:
            text (str): The input text to encode.
            max_len (int, optional): Maximum length of the output token IDs.

        Returns:
            A 1D torch.Tensor of token IDs.
        """
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        input_ids = self.sp.encode_as_ids(text)

        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids.extend([self.pad_id] * (max_len - len(input_ids)))

        return torch.tensor(input_ids, dtype=torch.long)

    def decode_sentence(self, input_ids) -> str:
        """
        Decode token IDs back to text.

        Args:
            input_ids (List[int] or torch.Tensor): The token IDs to decode.

        Returns:
            A decoded string.
        """
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        return self.sp.decode_ids(input_ids)

    def encode_label(self, label: str) -> torch.Tensor:
        """
        Convert a string label to an integer label ID.

        Args:
            label (str): The label to encode.

        Returns:
            A torch.Tensor with the label ID.
        """
        return torch.tensor([self.l2i[label]], dtype=torch.long)

    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        """
        Convert integer label IDs back to string labels.

        Args:
            label_vecs (torch.Tensor): A 1D tensor of label IDs.

        Returns:
            A list of string labels.
        """
        return [self.i2l[label_id.item()] for label_id in label_vecs]

    @property
    def total_labels(self) -> int:
        """Number of distinct labels."""
        return len(self.l2i)

    @property
    def total_tokens(self) -> int:
        """Vocabulary size determined at training time."""
        return self.vocab_size

    @property
    def get_vocab_size(self) -> int:
        """
        Returns the vocab size. Named differently to avoid confusion
        if you want to call it like a function: tokenizer.get_vocab_size.
        """
        return self.vocab_size

    @property
    def get_pad_idx(self) -> int:
        """Get the ID of the padding token."""
        return self.pad_id

    
    # def tokenize(self, text,  max_len=None, pad_token_id=0):
    #     # Tokenize the input text using SentencePiece
    #     tokens = self.sp.encode(text, out_type=str)

    #     # Prepend <b> and append <e> for BOS and EOS
    #     tokens = [self.bos_token] + tokens + [self.eos_token]

    #     # Map tokens to input_ids, handling unknown tokens with <u>
    #     input_ids = [self.stoi.get(token, self.stoi[self.unk_token])
    #                  for token in tokens]
        
    #     if max_len is not None:
    #         if len(input_ids) > max_len:
    #             # Truncate if too long
    #             input_ids = input_ids[:max_len]
    #         else:
    #             # Pad if too short
    #             input_ids.extend([pad_token_id] * (max_len - len(input_ids)))

    #     return {"tokens": tokens,
    #             "input_ids": input_ids}
    
    # def Printing_test(self): 
    # # Open the file in write mode, creating it if it doesn't exist
    #     with open("vocab_info.txt", "w", encoding="utf-8") as file:
    #         # Write Âm đầu details
    #         file.write(f"Vocab size: {len(self.token_to_id)}\n\n")
    #         file.write(f"Vocab: {self.token_to_id}\n\n")
    #         file.write(f"Labels: {len(self.l2i)}\n\n")
    #         file.write(f"Labels: {self.l2i}\n\n")
        
    #     print("Vocabulary details have been written to vocab_info.txt")
