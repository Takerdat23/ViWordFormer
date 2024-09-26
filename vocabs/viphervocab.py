import torch
import unicodedata
import json
from .utils import preprocess_sentence
from .word_decomposation import is_Vietnamese
from vocabs.vocab import Vocab

class ViPherVocab(Vocab):
    """
    A base Vocab class that is used to create vocabularies for particular tasks.
    """

    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token

        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

    def make_vocab(self, config):
        raise NotImplementedError("The abstract Vocab class must be inherited and implement!")

    def decompose_word(self, word):
        """Normalize and decompose Vietnamese word into its Unicode components."""
        return unicodedata.normalize('NFD', word)

    def compose_word(self, onset: str, medial: str, nucleus: str, coda: str, tone: str) -> str:
        tone_map = {
            '<`>': '\u0300',
            '</>': '\u0301',
            '<~>': '\u0303',
            '<?>': '\u0309',
            '<.>': '\u0323'
        }
        tone = tone_map[tone]
        if coda is None:
            nucleus = tone + nucleus
        else:
            nucleus = nucleus + tone

        word = ""
        if onset:
            word += onset
        if medial:
            word += medial
        if nucleus:
            word += nucleus
        if coda:
            word += coda

        return word

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """Turn a sentence into a vector of triplets (onset, tone, rhyme)."""
        pass

    def decode_sentence(self, encoded_sentence: torch.Tensor) -> str:
        """
        Decode a vector of triplets back into the original sentence.

        :param encoded_sentence: Tensor of shape (n_words, 3), where each triplet represents (onse index, tone index, rhyme index)
        :return: Decoded sentence as a string
        """
        pass

    def compose_word(self, onset, tone, rhyme):
        """
        Recompose a Vietnamese word from its components (âm đầu, tone, vần).

        Args:
            onset (str): The onset of the word.
            tone (str): The tone mark of the word.
            rhyme (str): The rhyme of the word.

        Returns:
            str: The fully composed word with tone marks correctly placed.
        """
        word = onset + rhyme

        if tone in self.TONE_MARKS:

            return self.insert_tone_mark(word, tone)
        else:
            return word

