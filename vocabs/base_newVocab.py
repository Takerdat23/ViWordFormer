import torch
import unicodedata
from typing import List
from .utils import preprocess_sentence
from collections import Counter
import regex as re
from .word_decomposation import split_word



class NewVocab(object):
    """
    A base Vocab class that is used to create vocabularies for particular tasks.
    """

    def __init__(self, config):
        self.ONSET_SET = ["m", "b", "v",  "t",  "đ", "n", "x", "s", "l", "h", "r", "g", "d", "k", "q", "c",
                          "ph", "th", "nh", "tr", "ch", "kh", "gh", "gi", "ng", "ngh"]

        self.uncommon_letters = {'w', 'z', 'f', 'j'}

        self.TONE_MARKS = {
            '\u0300': '<grave>',  # Huyền
            '\u0301': '<acute>',  # Sắc
            '\u0303': '<tilde>',  # ngã
            '\u0309': '<hook_above>',  # dấu hỏi
            '\u0323': '<dot_below>',  # dấu chấm
        }

        # Set of vowels in Vietnamese
        self.vowels = "aieuoy"

        self.VOWELS_LIST = ["ê", "e", "ư", "u", "ô", "i", "y", "o", "ơ", "â", "a", "o", "ă",
                            "ưo", "ươ", "uô", "ua", "iê", "yê", "ia", "ya"]

        self.FINAL_CONSONANTS = ["m", "n", "p", "t",
                                 "ng", "nh", "c", "ch", "u", "o", "i", "y"]

        self.special_tokens = ["!", "?", ":", ";", ",", "\"", "'", "%", "^", "`", "~",
                               "(", ")", "[", "]", "/", ".", "-",  "&", "$", "*", "_",
                               "=", "+", "<", ">", "{", "}", "|", "=", "•"]

        self.english_final_consonants = [
            "b", "d", "f", "k", "l", "r", "s", "v", "z"]
        self.english_charactor = ["w", "j", "z", "f"]
        self.vocab = {}
        self.index = 0
        self.initialize_special_tokens(config)
        self.make_vocab(config)

    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
        self.space_token = config.space_token

        self.specials = [self.pad_token, self.bos_token,
                         self.eos_token, self.unk_token, self.space_token]

        self.pad_idx = (0, 0, 0)
        self.bos_idx = (1, 1, 1)
        self.eos_idx = (2, 2, 2)
        self.unk_idx = (3, 3, 3)
        self.space_idx = (4, 4, 4)

    def make_vocab(self, config):
        raise NotImplementedError(
            "The abstract Vocab class must be inherited and implement!")

    def decompose_word(self, word):
        """Normalize and decompose Vietnamese word into its Unicode components."""
        return unicodedata.normalize('NFD', word)

    def recompose_word(self, onset, tone, rhyme):
        """
        Recompose a Vietnamese word from its components (âm đầu, tone, vần).

        Args:
            onset (str): The onset of the word.
            tone (str): The tone mark of the word.
            rhyme (str): The rhyme of the word.

        Returns:
            str: The fully composed word with tone marks correctly placed.
        """
        # Combine onset and rhyme
        word = onset + rhyme

        # Insert tone mark correctly in the word
        if tone in self.TONE_MARKS:
            # This assumes tone should be placed on the first main vowel
            return self.insert_tone_mark(word, tone)
        else:
            return word

    def remove_tone_marks(self, decomposed_word):
        """Remove only tone marks from the decomposed word while keeping accents."""
        TONE_MARKS = {"\u0301", "\u0300", "\u0309", "\u0303", "\u0323"}
        return ''.join([c for c in decomposed_word if unicodedata.combining(c) == 0 or c not in TONE_MARKS])

  
    def split_non_vietnamese_word(self, word):
      
        decomposed_character = self.decompose_word(word)

        onset = ""
        if decomposed_character in self.ONSET_SET:
            onset = decomposed_character
            return onset, "", ""
        else:
            return "", "", decomposed_character
  

    def encode_test(self, sentence: str):
        tokens = preprocess_sentence(sentence)

        vec = []
        for token in tokens:
            word_dict = split_word(token)
            if word_dict['is_vietnamese']:
             
                onset = word_dict['onset']
                tone = word_dict['tone']
                rhyme = ''.join([word_dict['medial'], word_dict['nucleus'], word_dict['coda']])
                vec.append((onset, tone, rhyme))
            else:
              
                for char in token:
                    onset, tone, rhyme = self.split_non_vietnamese_word(char)
                    vec.append((onset, tone, rhyme))  # Append the triplet
        return vec

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """Turn a sentence into a vector of triplets (onset, tone, rhyme)."""
        tokens = preprocess_sentence(sentence)

        vec = [self.bos_idx]

        for token in tokens:
            word_dict = split_word(token)
            if word_dict['is_vietnamese']:
              
                onset = word_dict['onset']
                tone = word_dict['tone']
                rhyme = ''.join([word_dict['medial'], word_dict['nucleus'], word_dict['coda']])
            else:
                vec.append(
                    (self.space_idx[0], self.space_idx[1], self.space_idx[2]))
                for char in token:
                    onset, tone, rhyme = self.split_non_vietnamese_word(char)

                    onset_idx = self.stoi_onset.get(onset, self.unk_idx[0])
                    tone_idx = self.stoi_tone.get(tone, self.unk_idx[1])
                    rhyme_idx = self.stoi_rhyme.get(rhyme, self.unk_idx[2])

                    vec.append((onset_idx, tone_idx, rhyme_idx))
                vec.append(
                    (self.space_idx[0], self.space_idx[1], self.space_idx[2]))
                continue

            onset_idx = self.stoi_onset.get(onset, self.unk_idx[0])
            tone_idx = self.stoi_tone.get(tone, self.unk_idx[1])
            rhyme_idx = self.stoi_rhyme.get(rhyme, self.unk_idx[2])

            vec.append((onset_idx, tone_idx, rhyme_idx))  # Append the triplet

        vec = vec + [self.eos_idx]
        return torch.Tensor(vec).long()

    def decode_sentence(self, encoded_sentence: torch.Tensor) -> str:
        """
        Decode a vector of triplets back into the original sentence.

        :param encoded_sentence: Tensor of shape (n_words, 3), where each triplet represents (onse index, tone index, rhyme index)
        :return: Decoded sentence as a string
        """
        words = []
        current_word = []
        i = 0  # Main loop index

        while i < len(encoded_sentence):

            onset_idx, tone_idx, rhyme_idx = encoded_sentence[i].tolist()

            # Decode the indices back to the actual components (onset, tone, rhyme)
            onset = self.itos_onset.get(onset_idx, "")
            tone = self.itos_tone.get(tone_idx, "")
            rhyme = self.itos_rhyme.get(rhyme_idx, "")

            tone_char = [k for k, v in self.TONE_MARKS.items() if v == tone]
            tone_char = tone_char[0] if tone_char else ""

            if onset == self.eos_token and tone == self.eos_token and rhyme == self.eos_token:

                words.append(self.eos_token)

            elif onset == self.bos_token and tone == self.bos_token and rhyme == self.bos_token:

                words.append(self.bos_token)

            elif onset == self.space_token and tone == self.space_token and rhyme == self.space_token:
                # Handle non vietnamese words

                current_word = []
                i += 1

                while i < len(encoded_sentence):

                    onset_idx, tone_idx, rhyme_idx = encoded_sentence[i].tolist(
                    )

                    # Decode each character
                    onset = self.itos_onset.get(onset_idx, "")
                    tone = self.itos_tone.get(tone_idx, "")
                    rhyme = self.itos_rhyme.get(rhyme_idx, "")

                    # Stop at the next space token, indicating end of non-Vietnamese word
                    if onset == self.space_token and tone == self.space_token and rhyme == self.space_token:
                        i += 1
                        break

                    if onset != "":
                        current_word.append(onset)
                    else:
                        current_word.append(rhyme)

                    i += 1

                if current_word:

                    words.append("".join(current_word))
                current_word = []
                continue

            else:
                # Handle Vietnamese words
                word = self.recompose_word(onset, tone_char, rhyme)
                words.append(word)

            i += 1

        if current_word:
            words.append("".join(current_word))

        return ' '.join(words)

    def recompose_word(self, onset, tone, rhyme):
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

    import unicodedata

    def insert_tone_mark(self, word, tone):
        """
        Insert the tone mark into the correct position within the Vietnamese word.

        Args:
            word (str): The word without the tone mark.
            tone (str): The tone mark to be inserted.

        Returns:
            str: The word with the tone mark inserted.
        """
        # Normalize the word to NFC form
        word = unicodedata.normalize('NFC', word)

        vowels_priority = ['a', 'â', 'ă', 'e','u', 'ư',
                           'ê', 'i', 'o', 'ô', 'ơ', 'y']

        for vowel in vowels_priority:

            index = word.find(vowel)
            if index != -1:

                combined_char = vowel + tone
                return word[:index] + combined_char + word[index + 1:]

        return word

    def __eq__(self, other: "NewVocab"):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)
