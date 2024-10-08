import sentencepiece as spm
import os
import torch
import json
import torch
import json
from collections import Counter, defaultdict
# from builders.vocab_builder import META_VOCAB
from vocabs.vocab import Vocab
from typing import List
from vocabs.utils import preprocess_sentence


class UnigramTokenizer(object):
    def __init__(self, config, model_type='unigram'):
        self.model_prefix = f"{
            config.training.checkpoint_path}/{config.model_prefix}"
        self.vocab_size = config.vocab_size
        self.model_type = model_type
        self.sp = None

        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
        self.space_token = config.space_token

        self.specials = [self.pad_token, self.bos_token,
                         self.eos_token, self.unk_token, self.space_token]

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        self.space_idx = 4

    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        labels = set()

        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for item in data:
                sentence = item["sentence"]
                self.corpus.append(sentence)
                labels.add(item["topic"])

        self.train()

        vocab_file = f"{self.model_prefix}.vocab"
        vocab = set()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                word = line.split()[0]
                vocab.add(word)

        labels = list(labels)
        self.vocab = list(vocab)

        self.itos = {i: token for i, token in enumerate(self.vocab)}
        self.stoi = {token: i for i, token in enumerate(self.vocab)}

        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

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
            raise FileNotFoundError(
                f"Model {model_file} not found. Train the model first.")

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length """
        sentence = preprocess_sentence(sentence)
        vec = [self.bos_idx] + [self.stoi[token]
                                if token in self.stoi else self.unk_idx for token in sentence] + [self.eos_idx]
        vec = torch.Tensor(vec).long()

        return vec

    def decode_sentence(self, sentence_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            sentence_vecs: (bs, max_length)
        '''
        sentences = []
        for vec in sentence_vecs:
            question = " ".join(
                [self.itos[idx] for idx in vec.tolist() if self.itos[idx] not in self.specials])
            if join_words:
                sentences.append(question)
            else:
                sentences.append(question.strip().split())

        return sentences

    def tokenize(self, text):
        self.load_model()

        # Tokenize the input text using SentencePiece
        tokens = self.sp.encode(text, out_type=str)

        # Prepend <b> and append <e> for BOS and EOS
        tokens = [self.bos_token] + tokens + [self.eos_token]

        # Map tokens to input_ids, handling unknown tokens with <u>
        input_ids = [self.stoi.get(token, self.stoi[self.unk_token])
                     for token in tokens]

        return {"tokens": tokens,
                "input_ids": input_ids}


if __name__ == "__main__":
    tokenizer = UnigramTokenizer(model_prefix="viocd_unigram", vocab_size=2000)
    tokenizer.train()

    tokenizer.load_model()
    text = "a nhô a sê ô"
    input_ids = tokenizer.encode(text, max_len=20, pad_token_id=0)

    print("Encoded input as tensor:", input_ids)
    print("Shape of tensor:", input_ids.shape)
    decoded_text = tokenizer.decode(input_ids)
    print("Decoded text:", decoded_text)
