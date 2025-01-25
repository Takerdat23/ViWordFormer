import os
import json
import torch
import sentencepiece as spm

from typing import List
from collections import Counter
from builders.vocab_builder import META_VOCAB
from .utils.utils import preprocess_sentence


@META_VOCAB.register()
class UnigramTokenizer:
    """
    Unigram tokenizer for Vietnamese text using SentencePiece.
    """

    def __init__(self, config, model_type: str = 'unigram'):
        """
        Initialize the Unigram tokenizer with the given configuration.

        Args:
            config: A configuration object containing paths, special tokens, etc.
            model_type (str): The SentencePiece model type (default is 'unigram').
        """
        self.model_prefix = config.model_prefix
        self.model_type = model_type
        self.sp = None
        self.config = config
        # Collect text data for training
        self.corpus = []

        # Decide vocab size from config
        if config.schema == 2:
            # A default fallback, but you can customize this logic as needed
            self.vocab_size = 1282  
        elif config.schema == 1:
            self.vocab_size = config.vocab_size
        else:
            # If there's a chance of an unknown schema, handle or raise
            raise ValueError(f"Unsupported schema type: {config.schema}")

        # Special tokens (strings)
        self.unk_piece = config.unk_piece
        self.bos_piece = config.bos_piece
        self.eos_piece = config.eos_piece
        self.pad_piece = config.pad_piece

        self.unk_id = config.unk_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id

        self.specials = [self.unk_piece, self.bos_piece, self.eos_piece, self.pad_piece]

        # Label mappings
        self.i2l = {}
        self.l2i = {}

        # After training, we will parse the SentencePiece .vocab file
        self.vocab = []
        self.itos = {}  # index -> token
        self.stoi = {}  # token -> index

        # Build vocab from data and train/load model
        self.make_vocab(config)

    def make_vocab(self, config):
        """
        Read data from JSON, build the corpus, train the model, 
        and build label+vocab dictionaries.
        """
        json_paths = [config.path.train, config.path.dev, config.path.test]

        # Check JSON file existence
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
                else:
                    labels.add(item[config.label])

        # Train the SentencePiece model if needed
        self.train()

        # After training, parse the .vocab file to build a Python-level vocab
        vocab_file = f"{self.model_prefix}.vocab"
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        vocab_set = set()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Each line typically: <token>\t<freq>\t[extra fields...]
                token = line.split()[0]
                vocab_set.add(token)
        
        
        # Decide vocab size based on schema
        if config.schema == 2:
            self.vocab_size = len(words_counter.keys())
        elif config.schema == 1:
            self.vocab_size = config.vocab_size
        else:
            # If there's a chance of an unknown schema, handle or raise
            raise ValueError(f"Unsupported schema type: {config.schema}")

        # Build label dictionaries (sorted if you need consistency)
        # Create label <-> index maps (sorted for consistent ordering)
        labels = sorted(list(labels))
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

        # Optionally, save these label mappings
        self.save_labels()
        
        # Build Python vocab from the SentencePiece vocab file
        self.vocab = list(vocab_set)
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        self.stoi = {token: i for i, token in enumerate(self.vocab)}

    def train(self):
        """
        Train the SentencePiece model using the unigram algorithm if it isn't already trained.
        """
        model_file = f"{self.model_prefix}.model"
        if os.path.exists(model_file):
            print(f"Model already exists at {model_file}")
            self.load_model()
            return

        # Write corpus to a temporary file
        temp_file = f"{self.model_prefix}_temp.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.corpus))

        # Build the training command for SentencePiece
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

        # Clean up
        os.remove(temp_file)

        # Load the newly created model
        self.load_model()
        print(f"Model trained and saved as {model_file}")
    
    def save_labels(self):
        """
        Save the label dictionaries (i2l, l2i) into a JSON file.
        """
        labels_file = f"{self.model_prefix}_labels.json"
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump({"i2l": self.i2l, "l2i": self.l2i}, f, ensure_ascii=False)
        print(f"Labels saved to {labels_file}")
        
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
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        words = preprocess_sentence(text)  # Split text into words using your preprocess_sentence method
        input_ids = []
        word_to_subword_mapping = []

        for word_idx, word in enumerate(words):
            # Encode each word into subwords
            subword_ids = self.sp.encode(word, out_type=int)
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
                subword_labels.append(labels[subword_idx-1])
        return subword_labels
    
    
    def encode_sentence(self, text: str, max_len: int = None) -> torch.Tensor:
        """
        Encode a sentence into token IDs using SentencePiece. Truncate/pad if max_len is given.
        
        Args:
            text (str): The input text to encode.
            max_len (int, optional): Maximum length of the output token IDs.

        Returns:
            torch.Tensor: A 1D tensor of token IDs.
        """
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        # Encode text to IDs
        input_ids = self.sp.encode_as_ids(text)

        # Truncate or pad as needed
        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids += [self.pad_id] * (max_len - len(input_ids))

        return torch.tensor(input_ids, dtype=torch.long)

    def decode_sentence(self, input_ids) -> str:
        """
        Decode token IDs back to a string using the loaded SentencePiece model.

        Args:
            input_ids (List[int] or torch.Tensor): The token IDs to decode.

        Returns:
            str: The decoded string.
        """
        if not self.sp:
            raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        decoded_sentence = self.sp.decode_ids(input_ids)
        return decoded_sentence

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
            int: The vocab size as specified/used by SentencePiece.
        """
        return self.vocab_size
    
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
        else:
            return [self.i2l[label_id.item()] for label_id in label_vecs]

    # def tokenize(self, text: str, max_len: int = None, pad_token_id: int = 0) -> dict:
    #     """
    #     Tokenize the input text into tokens (string pieces), prepend BOS, append EOS,
    #     then map those tokens to IDs using self.stoi.

    #     Args:
    #         text (str): The text to tokenize.
    #         max_len (int, optional): Max sequence length for truncation/padding.
    #         pad_token_id (int): ID to use for padding.

    #     Returns:
    #         dict: A dictionary with:
    #               "tokens": List[str] of token pieces
    #               "input_ids": List[int] of token IDs
    #     """
    #     if not self.sp:
    #         raise ValueError("Tokenizer model is not loaded. Call load_model() first.")

    #     # Encode text into string tokens via SentencePiece
    #     tokens = self.sp.encode(text, out_type=str)

    #     # Insert BOS/EOS if desired
    #     tokens = [self.bos_token] + tokens + [self.eos_token]

    #     # Convert string tokens to IDs, using the stoi dict
    #     # For unknown tokens, fall back to self.unk_token
    #     input_ids = [self.stoi.get(t, self.stoi.get(self.unk_token, self.unk_idx)) for t in tokens]

    #     # Apply optional truncation/padding
    #     if max_len is not None:
    #         if len(input_ids) > max_len:
    #             input_ids = input_ids[:max_len]
    #         else:
    #             input_ids += [pad_token_id] * (max_len - len(input_ids))

    #     return {
    #         "tokens": tokens,
    #         "input_ids": input_ids
    #     }