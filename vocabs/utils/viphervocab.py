import torch
import unicodedata
import json
from .utils import preprocess_sentence
from .word_decomposation import is_Vietnamese, split_non_vietnamese_word
from vocabs.utils.vocab import Vocab

class ViPherVocab(Vocab):
    """
    A base Vocab class that is used to create vocabularies for particular tasks.
    """

    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
        self.space_token = config.space_token

        self.specials = [self.pad_token, self.bos_token,
                         self.eos_token, self.unk_token, self.space_token]

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
        tokens = preprocess_sentence(sentence)

        vec = [self.bos_idx]

        for token in tokens:
            isVietnamese, wordsplit = is_Vietnamese(token)
            if isVietnamese:
              
                onset, medial, nucleus, coda, tone = wordsplit
                rhyme = ''.join(parts for parts in [medial, nucleus, coda] if parts is not None)
            else:
                vec.append(
                    (self.space_idx[0], self.space_idx[1], self.space_idx[2]))
                for char in token:
                    onset, tone, rhyme = split_non_vietnamese_word(char)

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
        
        tone_map =  {
            '\u0300': '<`>',
            '\u0301': '</>',
            '\u0303': '<~>',
            '\u0309': '<?>',
            '\u0323': '<.>',
        }
        words = []
        current_word = []
        i = 0  # Main loop index
      
        encoded_sentence = encoded_sentence[0]
        
        while i < len(encoded_sentence):

            onset_idx, tone_idx, rhyme_idx = encoded_sentence[i].tolist()

            # Decode the indices back to the actual components (onset, tone, rhyme)
            onset = self.itos_onset.get(onset_idx, "")
            tone = self.itos_tone.get(tone_idx, "")
            rhyme = self.itos_rhyme.get(rhyme_idx, "")
            
          

            tone_char = [k for k, v in tone_map.items() if v == tone]
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
        
        TONE_MARKS = {
        '\u0300': '<`>',
        '\u0301': '</>',
        '\u0303': '<~>',
        '\u0309': '<?>',
        '\u0323': '<.>',
        }

        # Combine onset and rhyme
        word = onset + rhyme

        # Insert tone mark correctly in the word
        if tone in TONE_MARKS:
            # This assumes tone should be placed on the first main vowel
            return self.insert_tone_mark(word, tone)
        else:
            return word