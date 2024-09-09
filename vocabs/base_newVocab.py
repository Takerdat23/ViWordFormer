import torch
import unicodedata
from typing import List
from .utils import preprocess_sentence
from collections import Counter
import regex as re 
class NewVocab(object):
    """
    A base Vocab class that is used to create vocabularies for particular tasks.
    """
    def __init__(self, config):
        self.ONSET_SET = ["m", "b", "v", "ph", "t", "th", "đ", "n", "x", "s", "l", "ch", "tr", "nh", 
                         "kh", "h", "r", "gi", "d", "k", "q", "c", "gh", "g", "ngh", "ng"]
        
        self.uncommon_letters = {'w', 'z', 'f', 'j'}
        
        self.TONE_MARKS = {
            '\u0300': '<grave>', # Huyền
            '\u0301': '<acute>', # Sắc
            '\u0303': '<tilde>',  # ngã
            '\u0309': '<hook_above>', # dấu hỏi
            '\u0323': '<dot_below>', # dấu chấm
        }
        
        # Set of vowels in Vietnamese
        self.vowels = "aieuoy"
        
        self.VOWELS_LIST = ["ê", "e", "ư", "u", "ô", "i", "y", "o", "ơ", "â", "a", "o", "ă", 
                              "ưo", "ươ", "uô", "ua", "iê", "yê", "ia", "ya"]
        
        self.FINAL_CONSONANTS = ["m", "n", "p", "t", "ng", "nh", "c", "ch", "u", "o", "i", "y"]
        
        self.special_tokens = ["!", "?", ":", ";", ",", "\"", "'", "%", "^", "`", "~",
                      "(", ")", "[", "]", "/", ".", "-", "$", "&", "*"]
        

        self.vocab = {}
        self.index = 0
        self.initialize_special_tokens(config)
        self.make_vocab(config)

    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token

        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_idx = (0, 0, 0)
        self.cls_idx = (1, 1, 1)
        self.eos_idx = (2, 2, 2)
        self.unk_idx = (3, 3, 3)

    def make_vocab(self, config):
        raise NotImplementedError("The abstract Vocab class must be inherited and implement!")

    def decompose_word(self, word):
        """Normalize and decompose Vietnamese word into its Unicode components."""
        return unicodedata.normalize('NFD', word)

    def remove_tone_marks(self, decomposed_word):
        """Remove only tone marks from the decomposed word while keeping accents."""
        TONE_MARKS = {"\u0301", "\u0300", "\u0309", "\u0303", "\u0323"}
        return ''.join([c for c in decomposed_word if unicodedata.combining(c) == 0 or c not in TONE_MARKS])
    
    def is_vietnamese_word(self, word: str) -> bool:
        #This exception may contain english stopwords
        EXCEPTION_WORD = [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
            'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
            "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
            "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
            'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', 
            "wouldn't", "pet", "copy", "play" , "unit", "avt", "pubg", ">", "<", ".", "/", "hada", "covid", 
            "tiktoker", "pokestops", "pokémon", "oke", "hope", "item", "level", "never", "code", "kil", "ks", "opo", "lsv", "net", 
            "pink", "kpop", "mod", "gamer", "lock", "mail", "sms", "aply","pay"
        ]
        
        EXCEPTION_RHYME = ["hope", "ada", "oke", "ina", "tar", "star", "ogle", "ode", "ick", "one", "ale", "ade", "adm" 
                           "ude", "eal", "mail", "lock", "uper", "eve", "ever", "ite", "rab", "eta", "oca", "ack", "ed", "ad"]
        if word.lower() in EXCEPTION_WORD:
            return False
        
        japanese_pattern = re.compile(
                "[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", flags=re.UNICODE
            )

        if japanese_pattern.search(word):
            return False
                
        if any(char in self.special_tokens for char in word):
            return False
        
        if any(char.isdigit() for char in word):
            return False

        
        emoji_pattern = re.compile(
                "[\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F700-\U0001F77F"  # alchemical symbols
                "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "\U00002702-\U000027B0"  # Dingbats
                "\U000024C2-\U0001F251" 
                "]+", flags=re.UNICODE
            )

        if emoji_pattern.search(word):
            return False
    
        # Handle very long words
        if len(unicodedata.normalize('NFC', word) ) > 7:
            return False
        
        # onset_count = sum(1 for onset_candidate in self.ONSET_SET if onset_candidate in word)
        
        # if onset_count > 2:
        #     return False
        

        # Normalize and check for uncommon letters
        decomposed = unicodedata.normalize('NFD', word)
        removed_tone = self.remove_tone_marks(word)
        
        if any(char in self.uncommon_letters for char in decomposed):
       
            return False
            
        onset, tone, rhyme = self.split_vietnamese_word(word)
        

        if rhyme in EXCEPTION_RHYME : 
            return False
        
        if tone == '' and len(unicodedata.normalize('NFC', rhyme) ) > 4: 
            return False 
        
        # if len(unicodedata.normalize('NFC', removed_tone) ) > 2  and removed_tone[0] == 'p' and removed_tone[1] != 'h': 
        #     return False
        if onset:
            
            temp = decomposed
            temp = self.remove_tone_marks(temp)
            temp = temp[len(onset):]
        
            if len(temp) == 1 and temp in self.vowels: 
                return True
            
    
            
            if len(unicodedata.normalize('NFC', temp) ) == 0 :
                return False
              
            
            for consonant in sorted(self.FINAL_CONSONANTS, key=len, reverse=True):
                if temp.endswith(consonant):
                    
                    temp = temp[:-len(consonant)].strip()
                 
                  
                    if len(unicodedata.normalize('NFC', temp) ) == 1  and temp in self.vowels:
               
                        return True
                    
                    else:
                        for o in self.ONSET_SET:
                            
                            if o in temp: 
                                return False
            else :
                    
                if len(unicodedata.normalize('NFC', temp) ) == 0 :
                        return False
            
                    # for o in self.ONSET_SET:
                    #         if o in  temp[ :-1]: 
                    #             return False 
                 
                  
                    # if len(unicodedata.normalize('NFC', temp) ) != 1 and  temp[-1] in self.ONSET_SET:
                    #     return False
                 
        return True
 
    
    
   


    def split_vietnamese_word(self, word):
        """
        Split a Vietnamese word into Âm đầu (initial consonants), Thanh Điệu (tone marks),
        and Vần (rhyme, including accents but without tone).
        """

        special_tokens = ["!", "?", ":", ";", ",", "\"", "'", "%", "^", "`", "~",
                        "(", ")", "[", "]", "/", ".", "-", "$", "&", "*"]
        if word in special_tokens:
            return self.eos_token, self.eos_token, self.eos_token

        # Normalize both the input word and the comparison strings
        word = unicodedata.normalize('NFC', word)
        
        if word == unicodedata.normalize('NFC', 'giếng'):
            return 'g', '<acute>', 'iêng'
        
        if word == unicodedata.normalize('NFC', 'gìn'):
            return 'g', '<grave>', 'in'
        
        if word == unicodedata.normalize('NFC', 'pin'):
            return 'p', '', 'in'


        decomposed_word = self.decompose_word(word)

        onset = ""
        for initial in sorted(self.ONSET_SET, key=len, reverse=True):
            if decomposed_word.startswith(initial):
                onset = initial
                decomposed_word = decomposed_word[len(initial):]
                break

        tone = ""
        for char in decomposed_word:
            if unicodedata.combining(char):
                if char in self.TONE_MARKS:
                    tone = self.TONE_MARKS[char]
                    break

        rhyme = self.remove_tone_marks(decomposed_word)
        
        return onset, tone, rhyme
    
    def split_non_vietnamese_word(self, word):
        """
            Split a Vietnamese word into Âm đầu (initial consonants), Thanh Điệu (tone marks),
            and Vần (rhyme, including accents but without tone).
        """
        
        special_tokens = ["!", "?", ":", ";", ",", "\"", "'", "%", "^", "`", "~",
                        "(", ")", "[", "]", "/", ".", "-", "$", "&", "*"]
        if word in special_tokens:
                return self.eos_token, self.eos_token,self.eos_token
        
        decomposed_character = self.decompose_word(word)

        onset = ""
        if decomposed_character in self.ONSET_SET: 
            onset = decomposed_character
            return onset , "", ""
        elif decomposed_character.isalpha() : 
            return "", "", decomposed_character
        else: 
            return self.eos_token, self.eos_token,self.eos_token
     
     
    def encode_test(self, sentence: str): 
        tokens = preprocess_sentence(sentence)  
      
        vec = []  
        for token in tokens:
            if self.is_vietnamese_word(token):
                print("vietnamese") 
                onset, tone, rhyme = self.split_vietnamese_word(token)

                vec.append((onset, tone, rhyme))  # Append the triplet
            else: 
                print("non vietnamese")
                for char in token:
                    onset, tone, rhyme = self.split_non_vietnamese_word(char)
                    vec.append((onset, tone, rhyme))  # Append the triplet
        
        
        return vec
 
    
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """Turn a sentence into a vector of triplets (âm đầu, tone, vần)."""
        tokens = preprocess_sentence(sentence)  
        vec = [(self.cls_idx[0], self.cls_idx[1], self.cls_idx[2])]  
        
        for token in tokens:
            onset, tone, rhyme = self.split_vietnamese_word(token)

            onset_idx = self.stoi_onset.get(onset, self.unk_idx[0])
            tone_idx = self.stoi_tone.get(tone, self.unk_idx[1])
            rhyme_idx = self.stoi_rhyme.get(rhyme, self.unk_idx[2])
            
            vec.append((onset_idx, tone_idx, rhyme_idx))  # Append the triplet

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
            am_dau = self.itos_onset.get(am_dau_idx, "")
            tone = self.itos_tone.get(tone_idx, "")
            van = self.itos_rhyme.get(van_idx, "")

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
    
    def split_van(self, van: str):
            """
            Split the Vietnamese van into âm đệm (medial sound), âm chính (main vowel),
            and âm cuối (final consonant).
            
            :param van: The rhyme part of the word without tone marks.
            :return: A tuple (âm đệm, âm chính, âm cuối)
            
            Not in useful (for now)
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

    def __eq__(self, other: "NewVocab"):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)
