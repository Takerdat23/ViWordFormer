import torch
from .utils import preprocess_sentence
from transformers import AutoTokenizer

class PhoBERT_Vocab:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_name)

        self.stoi = self.tokenizer.get_vocab()
        self.itos = {i: s for s, i in self.stoi.items()}

        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token

        self.pad_idx = self.tokenizer.pad_token_id
        self.unk_idx = self.tokenizer.unk_token_id

    def make_vocab(self, config):
        raise NotImplementedError("The abstract Vocab class must be inherited and implement!")
    
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length """
        sentence = sentence.split()
        vec = [self.stoi[token] if token in self.stoi else self.unk_idx for token in sentence]
        vec = torch.Tensor(vec).long()

        return vec

    def decode_sentence(self, sentence_vecs: torch.Tensor, join_words=True) -> list[str]:
        '''
            sentence_vecs: (bs, max_length)
        '''
        sentences = self.tokenizer.batch_decode(sentence_vecs)

        if not join_words:
            sentences = [sentence.split() for sentence in sentences]

        return sentences

    