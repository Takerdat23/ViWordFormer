import torch
import unicodedata
from typing import List
from .utils import preprocess_sentence

class NewVocab(object):
    """
    A base Vocab class that is used to create vocabularies for particular tasks.
    """
    def __init__(self, config):
        self.INITIALS = ["m", "b", "v", "ph", "t", "th", "đ", "n", "x", "s", "l", "ch", "tr", "nh", 
                         "kh", "h", "r", "gi", "d", "k", "q", "c", "gh", "g", "ngh", "ng"]
        
        self.TONE_MARKS = {
            "\u0301": "Sắc",    # ́  (U+0301)
            "\u0300": "Huyền",  # ̀  (U+0300)
            "\u0309": "Hỏi",    # ̉  (U+0309)
            "\u0303": "Ngã",    # ̃  (U+0303)
            "\u0323": "Nặng"    # ̣  (U+0323)
        }
        
        self.AM_CHINH_LIST = ["ê", "e", "ư", "u", "ô", "i", "y", "o", "oo", "ơ", "â", "a", "o", "ă", 
                              "ưo", "ươ", "uô", "ua", "iê", "yê", "ia", "ya"]
        
        self.FINAL_CONSONANTS = ["m", "n", "p", "t", "ng", "nh", "c", "ch", "u", "o", "i", "y"]
        
        self.initialize_special_tokens(config)
        self.make_vocab(config)

    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token

        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_idx = (0, 0, 0, 0, 0)
        self.cls_idx = (1, 1, 1, 1, 1)
        self.eos_idx = (2, 2, 2, 2, 2)
        self.unk_idx = (3, 3, 3, 3, 3)

    def make_vocab(self, config):
        raise NotImplementedError("The abstract Vocab class must be inherited and implement!")

    def decompose_word(self, word):
        """Normalize and decompose Vietnamese word into its Unicode components."""
        return unicodedata.normalize('NFD', word)

    def remove_tone_marks(self, decomposed_word):
        """Remove only tone marks from the decomposed word while keeping accents."""
        TONE_MARKS = {"\u0301", "\u0300", "\u0309", "\u0303", "\u0323"}
        return ''.join([c for c in decomposed_word if unicodedata.combining(c) == 0 or c not in TONE_MARKS])

    def split_vietnamese_word(self, word):
        """
        Split a Vietnamese word into Âm đầu (initial consonants), Thanh Điệu (tone marks),
        and Vần (rhyme, including accents but without tone).
        """

        special_tokens = ["!", "?", ":", ";", ",", "\"", "'", "%", "^", "`", "~",
                      "(", ")", "[", "]", "/", ".", "-", "$", "&", "*"]
        if word in special_tokens:
            return self.eos_token, self.eos_token,self.eos_token, self.eos_token,self.eos_token 
    
        decomposed_word = self.decompose_word(word)

        # 1. Identify Âm đầu (initial consonant)
        am_dau = ""
        for initial in sorted(self.INITIALS, key=len, reverse=True):
            if decomposed_word.startswith(initial):
                am_dau = initial
                decomposed_word = decomposed_word[len(initial):]
                break

        # 2. Identify Thanh Điệu (tone mark)
        tone = "No mark"
        for char in decomposed_word:
            if unicodedata.combining(char):
                if char in self.TONE_MARKS:
                    tone = self.TONE_MARKS[char]
                    break

        # 3. Identify Âm cuối (final consonant)
        am_cuoi = ""
       
        for consonant in sorted(self.FINAL_CONSONANTS, key=len, reverse=True):
            if decomposed_word.endswith(consonant):
                am_cuoi = consonant
                decomposed_word = decomposed_word[:-len(consonant)]
                break

        # Remove tone marks from the word
        van = self.remove_tone_marks(decomposed_word)
        van = unicodedata.normalize('NFC', van)  # Recompose the word after removing tones

        # 4. Further processing of van
        if van in self.AM_CHINH_LIST:
            # Case 1: If the whole `vần` is included in âm chính list
            am_dem = '0'
            am_chinh = van
        elif not van:
            # Case 2: If the `vần` is empty, classify as unknown
            if len(am_dau) > 1:
                am_dem = '0'
                am_chinh = am_dau[1:]
                am_dau = am_dau[0]  # No initial consonant left
            else:
                am_dem = '0'
                am_chinh = self.unk_token
              
        else:
            # Case 3: Normal splitting
            am_dem, am_chinh, am_cuoi = self.split_van(van)

        return am_dau, tone, am_dem, am_chinh, am_cuoi

    def split_van(self, van: str):
            """
            Split the Vietnamese van into âm đệm (medial sound), âm chính (main vowel),
            and âm cuối (final consonant).
            
            :param van: The rhyme part of the word without tone marks.
            :return: A tuple (âm đệm, âm chính, âm cuối)
            """
            # 1. Identify âm đệm (medial sound)
            am_dem = '0'  # Default value if no medial sound is found
            if len(van) > 1 and (van[0] == 'u' or van[0] == 'o'):
                am_dem = van[0]
                van = van[1:]  # Remove the medial sound from van

            # 2. Identify âm cuối (final consonant)
            am_cuoi = ''
            for consonant in sorted(self.FINAL_CONSONANTS, key=len, reverse=True):
                if van.endswith(consonant):
                    am_cuoi = consonant
                    van = van[:-len(consonant)]  # Remove final consonant from van
                    break

            # 3. The rest is âm chính (main vowel)
            am_chinh = van

            return am_dem, am_chinh, am_cuoi

    

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """Turn a sentence into a vector of triplets (âm đầu, tone, vần)."""
        tokens = preprocess_sentence(sentence)  # Assume this function tokenizes the sentence
        vec = [(self.cls_idx[0], self.cls_idx[1], self.cls_idx[2])]  # Add CLS token triplet
        
        for token in tokens:
            am_dau, tone, van = self.split_vietnamese_word(token)

            # Get the corresponding indices for âm đầu, tone, and vần
            am_dau_idx = self.stoi_am_dau.get(am_dau, self.unk_idx[0])
            tone_idx = self.stoi_tone.get(tone, self.unk_idx[1])
            van_idx = self.stoi_van.get(van, self.unk_idx[2])
            
            vec.append((am_dau_idx, tone_idx, van_idx))  # Append the triplet

        return torch.Tensor(vec).long()
    
    def decode_sentence(self, encoded_sentence: torch.Tensor) -> str:
        """
        Decode a vector of triplets back into the original sentence.
        
        :param encoded_sentence: Tensor of shape (n_words, 3), where each triplet represents (âm đầu index, tone index, vần index)
        :return: Decoded sentence as a string
        """
        words = []
        
        for triplet in encoded_sentence:
            am_dau_idx, tone_idx, van_idx = triplet.tolist()
            
            # Decode the indices back to the actual components (âm đầu, tone, vần)
            am_dau = self.itos_am_dau.get(am_dau_idx, "")
            tone = self.itos_tone.get(tone_idx, "")
            van = self.itos_van.get(van_idx, "")

            # Reconstruct the word with tone marks
            word = self.recompose_word(am_dau, tone, van)
            words.append(word)
        
        # Join the words to form the full sentence
        return ' '.join(words)

    

    def recompose_word(self, am_dau: str, tone: str, van: str) -> str:
        """
        Recompose a word from its components (âm đầu, tone, vần) into a normalized form.
        
        :param am_dau: The initial consonant (âm đầu)
        :param tone: The tone mark (thanh điệu)
        :param van: The remaining part of the word (vần)
        :return: The recomposed word as a string
        """
        # Mapping between tone names and Unicode combining characters
        tone_to_unicode = {
            "Sắc": "\u0301",   # Combining acute accent
            "Huyền": "\u0300", # Combining grave accent
            "Hỏi": "\u0309",   # Combining hook above
            "Ngã": "\u0303",   # Combining tilde
            "Nặng": "\u0323",  # Combining dot below
            "No mark": ""      # No tone mark
        }

        # Combine âm đầu and vần
        word = am_dau + van
        
        # If there's a tone, find the appropriate letter to attach it to
        if tone != "No mark" and tone != "" and tone not in self.specials:
            decomposed_word = unicodedata.normalize('NFD', word)
            tone_char = tone_to_unicode.get(tone, "")
            
            # Attach the tone mark to the first non-initial consonant character
            for i, char in enumerate(decomposed_word):
                if unicodedata.combining(char) == 0 and char not in self.INITIALS:
                    word = decomposed_word[:i+1] + tone_char + decomposed_word[i+1:]
                    break
        
        # Recompose and normalize the word
        word = unicodedata.normalize('NFC', word)
        return word

    def __eq__(self, other: "NewVocab"):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)
