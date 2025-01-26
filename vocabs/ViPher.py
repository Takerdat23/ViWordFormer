import os
import json
import torch
import unicodedata
from typing import List
from collections import Counter
from builders.vocab_builder import META_VOCAB
from .utils.utils import preprocess_sentence
from .utils.word_decomposation import is_Vietnamese, split_non_vietnamese_word

@META_VOCAB.register()
class VipherTokenizer:
    def __init__(self, config):
        """
        Initialize the tokenizer and build the vocabulary.

        Args:
            config: A configuration object with fields like:
                - config.pad_token, bos_token, eos_token, unk_token, space_token
                - config.path.train, dev, test: JSON data files
                - config.text, config.label: the JSON keys for text/label
                - config.schema, config.min_freq, etc. if needed
        """
        self._initialize_special_tokens(config)
        self.nonvietnamese = []
        self.vietnamese = []
        self.config = config

        # Vocab dictionaries for onset, tone, rhyme
        self.itos_onset = {}
        self.stoi_onset = {}
        self.itos_tone = {}
        self.stoi_tone = {}
        self.itos_rhyme = {}
        self.stoi_rhyme = {}

        # Label mappings
        self.i2l = {}
        self.l2i = {}

        # Build the vocab from JSON data
        self._make_vocab(config)

    def _initialize_special_tokens(self, config) -> None:
        """
        Define special tokens and their corresponding tuple-index forms 
        (onset_idx, tone_idx, rhyme_idx).
        """
        self.pad_token = config.unk_piece
        self.bos_token = config.bos_piece
        self.eos_token = config.eos_piece
        self.unk_token = config.unk_piece
        self.space_token = config.space_token

        # The base list of special tokens
        self.specials = [
            self.pad_token,  # Index 0
            self.bos_token,  # Index 1
            self.eos_token,  # Index 2
            self.unk_token,  # Index 3
            self.space_token # Index 4
        ]

        # Because we store each token as a triplet (onset, tone, rhyme),
        # we treat each special token as the same index for all three parts.
        self.pad_idx   = (config.pad_id, config.pad_id, config.pad_id)
        self.bos_idx   = (config.bos_id, config.bos_id, config.bos_id)
        self.eos_idx   = (config.eos_id, config.eos_id, config.eos_id)
        self.unk_idx   = (config.unk_id, config.unk_id, config.unk_id)
        self.space_idx = (config.space_id, config.space_id, config.space_id)

    def _make_vocab(self, config):
        """
        Build the onset/tone/rhyme vocabulary from the JSON data,
        differentiating Vietnamese from non-Vietnamese text.
        """
        json_paths = [config.path.train, config.path.dev, config.path.test]
        counter_onset = Counter()
        counter_tone  = Counter()
        counter_rhyme = Counter()
        labels = set()

        # Collect token stats from each JSON
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                if isinstance(item[config.text], list):
                    token_list = preprocess_sentence(" ".join(item[config.text]))
                else: 
                    text = item[config.text]
                    token_list = preprocess_sentence(text)

                for token in token_list:
                    is_vn, word_split = is_Vietnamese(token)
                    if is_vn:
                        # Vietnamese word
                        if token not in self.vietnamese:
                            self.vietnamese.append(token)

                        # word_split = (onset, medial, nucleus, coda, tone)
                        onset, medial, nucleus, coda, tone = word_split
                        onset  = onset  if onset  else ""
                        medial = medial if medial else ""
                        nucleus= nucleus if nucleus else ""
                        coda   = coda   if coda   else ""
                        tone   = tone   if tone   else ""

                        # Rhyme is the concatenation of medial/nucleus/coda
                        rhyme = ''.join([medial, nucleus, coda])

                        # Update counters (skip if token is a special token)
                        if onset  not in self.specials: counter_onset.update([onset])
                        if tone   not in self.specials: counter_tone.update([tone])
                        if rhyme  not in self.specials: counter_rhyme.update([rhyme])

                    else:
                        # Non-Vietnamese word, split into characters
                        if token not in self.nonvietnamese:
                            self.nonvietnamese.append(token)

                        for char in token:
                            onset_char, tone_char, rhyme_char = split_non_vietnamese_word(char)
                            if onset_char not in self.specials:
                                counter_onset.update([onset_char])
                            if tone_char not in self.specials:
                                counter_tone.update([tone_char])
                            if rhyme_char not in self.specials:
                                counter_rhyme.update([rhyme_char])


                if self.config.get("task_type", None) == "seq_labeling":
                    for label in item[config.label]:
                        label = label.split("-")[-1]
                        labels.add(label)
                else:
                    labels.add(item[config.label])

        # Sort the counters to build final vocab lists
        sorted_onset = sorted(counter_onset)
        sorted_tone  = sorted(counter_tone)
        sorted_rhyme = sorted(counter_rhyme)

        # Build itos/stoi for onset, tone, rhyme
        # Prepend the specials at the start
        self.itos_onset = {
            i: tok for i, tok in enumerate(self.specials + sorted_onset)
        }
        self.stoi_onset = {
            tok: i for i, tok in enumerate(self.specials + sorted_onset)
        }

        self.itos_tone = {
            i: tok for i, tok in enumerate(self.specials + sorted_tone)
        }
        self.stoi_tone = {
            tok: i for i, tok in enumerate(self.specials + sorted_tone)
        }

        self.itos_rhyme = {
            i: tok for i, tok in enumerate(self.specials + sorted_rhyme)
        }
        self.stoi_rhyme = {
            tok: i for i, tok in enumerate(self.specials + sorted_rhyme)
        }

        # Build label <-> index maps
        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}
        
        
        
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
    
        words = preprocess_sentence(text)  # Split text into words using your preprocess_sentence method
        word_to_subword_mapping = []
        
        vec =[]

        for word_idx, word in enumerate(words):
            # Encode each word into subwords
            
            input_ids = []
            
            is_vn, word_split = is_Vietnamese(word)
            if is_vn:
                onset, medial, nucleus, coda, tone = word_split
                onset  = onset  if onset  else ""
                medial = medial if medial else ""
                nucleus= nucleus if nucleus else ""
                coda   = coda   if coda   else ""
                tone   = tone   if tone   else ""

                rhyme = ''.join([medial, nucleus, coda])

                onset_idx = self.stoi_onset.get(onset, self.unk_idx[0])
                tone_idx  = self.stoi_tone.get(tone, self.unk_idx[1])
                rhyme_idx = self.stoi_rhyme.get(rhyme, self.unk_idx[2])
                input_ids.append((onset_idx, tone_idx, rhyme_idx))
                vec.append((onset_idx, tone_idx, rhyme_idx))

            else:
                # For non-Vietnamese, we mark a "space" triplet, then each char
                # individually, then a closing "space" triplet
                vec.append(self.space_idx)
                input_ids.append(self.space_idx)
                for char in word:
                    onset_c, tone_c, rhyme_c = split_non_vietnamese_word(char)
                    o_idx = self.stoi_onset.get(onset_c, self.unk_idx[0])
                    t_idx = self.stoi_tone.get(tone_c, self.unk_idx[1])
                    r_idx = self.stoi_rhyme.get(rhyme_c, self.unk_idx[2])
                    vec.append((o_idx, t_idx, r_idx))
                    input_ids.append((o_idx, t_idx, r_idx))
                vec.append(self.space_idx)
                input_ids.append(self.space_idx)
                
            word_to_subword_mapping.extend([word_idx] * len(input_ids))
        
        
        if max_len is not None:
            if len(vec) > max_len:
                vec = vec[:max_len]
            else:
                vec.extend([self.pad_id] * (max_len - len(vec)))


        return torch.tensor(vec, dtype=torch.long), word_to_subword_mapping
    
    
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
                print(f"Warning: subword_idx ({subword_idx}) exceeds labels length ({len(labels)})")
                subword_labels.append(labels[-1]) 
        return subword_labels
    

    def encode_sentence(self, sentence: str, max_len: int = None) -> torch.Tensor:
        """
        Turn a sentence into a list of triplets (onset_idx, tone_idx, rhyme_idx),
        including BOS at the start and EOS at the end.

        Args:
            sentence (str): The raw sentence to encode.

        Returns:
            torch.Tensor of shape (sequence_length, 3).
        """
        tokens = preprocess_sentence(sentence)
        vec = [self.bos_idx]  # Start with BOS triplet

        for token in tokens:
            is_vn, word_split = is_Vietnamese(token)
            if is_vn:
                onset, medial, nucleus, coda, tone = word_split
                onset  = onset  if onset  else ""
                medial = medial if medial else ""
                nucleus= nucleus if nucleus else ""
                coda   = coda   if coda   else ""
                tone   = tone   if tone   else ""

                rhyme = ''.join([medial, nucleus, coda])

                onset_idx = self.stoi_onset.get(onset, self.unk_idx[0])
                tone_idx  = self.stoi_tone.get(tone, self.unk_idx[1])
                rhyme_idx = self.stoi_rhyme.get(rhyme, self.unk_idx[2])
                vec.append((onset_idx, tone_idx, rhyme_idx))

            else:
                # For non-Vietnamese, we mark a "space" triplet, then each char
                # individually, then a closing "space" triplet
                vec.append(self.space_idx)
                for char in token:
                    onset_c, tone_c, rhyme_c = split_non_vietnamese_word(char)
                    o_idx = self.stoi_onset.get(onset_c, self.unk_idx[0])
                    t_idx = self.stoi_tone.get(tone_c, self.unk_idx[1])
                    r_idx = self.stoi_rhyme.get(rhyme_c, self.unk_idx[2])
                    vec.append((o_idx, t_idx, r_idx))
                vec.append(self.space_idx)

        vec.append(self.eos_idx)

        if max_len is not None:
            if len(vec) > max_len:
                vec = vec[:max_len]
                vec[-1] = self.eos_idx
            elif len(vec) < max_len:
                # pad with the pad triplet
                vec += [self.pad_idx] * (max_len - len(vec))
        return torch.tensor(vec, dtype=torch.long)

    def decode_sentence(self, encoded_sentence: torch.Tensor) -> str:
        """
        Decode a tensor of triplets back into the original sentence. 
        Splits out Vietnamese vs. non-Vietnamese logic.

        Args:
            encoded_sentence: (sequence_length, 3) or (batch_size, seq_len, 3).
                              If there's a batch dimension, we'll decode just the first item 
                              or you can adapt to handle multiple.

        Returns:
            str: The reconstructed sentence.
        """
        # If there's a batch dimension, pick the first item
        if encoded_sentence.ndim == 3:
            encoded_sentence = encoded_sentence[0]

        # Map from combining Unicode tone marks to bracket notation
        tone_map = {
            '\u0300': '<`>',
            '\u0301': '</>',
            '\u0303': '<~>',
            '\u0309': '<?>',
            '\u0323': '<.>',
        }

        words = []
        current_word = []
        i = 0
        while i < len(encoded_sentence):
            onset_idx, tone_idx, rhyme_idx = encoded_sentence[i].tolist()

            onset = self.itos_onset.get(onset_idx, "")
            tone  = self.itos_tone.get(tone_idx, "")
            rhyme = self.itos_rhyme.get(rhyme_idx, "")

            # BOS/EOS check (if needed)
            if (onset, tone, rhyme) == (self.bos_token, self.bos_token, self.bos_token):
                i += 1
                continue
            if (onset, tone, rhyme) == (self.eos_token, self.eos_token, self.eos_token):
                i += 1
                break

            # SPACE token indicates a pivot into "non-Vietnamese" sequence
            if (onset, tone, rhyme) == (self.space_token, self.space_token, self.space_token):
                # We've hit the "space" delimiter. 
                # Next chunk is the non-VN word until the next space token or EOS.
                i += 1
                current_word = []
                while i < len(encoded_sentence):
                    on_idx, to_idx, rh_idx = encoded_sentence[i].tolist()
                    on_str = self.itos_onset.get(on_idx, "")
                    to_str = self.itos_tone.get(to_idx, "")
                    rh_str = self.itos_rhyme.get(rh_idx, "")
                    if (on_str, to_str, rh_str) == (self.space_token, self.space_token, self.space_token):
                        # End of non-VN word
                        i += 1
                        break
                    # We treat the onset or rhyme as a single char for non-VN text
                    char_to_add = on_str if on_str else rh_str
                    current_word.append(char_to_add)
                    i += 1

                if current_word:
                    words.append("".join(current_word))
                continue

            # Otherwise, it's a Vietnamese word piece
            # Convert bracket tone to actual Unicode combining mark
            tone_char = [k for k, v in tone_map.items() if v == tone]
            tone_char = tone_char[0] if tone_char else ""

            # Recompose a single Vietnamese word from (onset + rhyme + tone)
            word = self._recompose_word(onset, tone_char, rhyme)
            words.append(word)
            i += 1

        return ' '.join(words)


    def _recompose_word(self, onset: str, tone_mark: str, rhyme: str) -> str:
        """
        Rebuild a Vietnamese word from onset + rhyme, inserting the tone mark 
        in the correct place if needed.
        """
        word = onset + rhyme
        # Insert the tone mark (if any) onto the first main vowel in the rhyme
        if tone_mark:
            word = self._insert_tone_mark(word, tone_mark)
        return word

    def _insert_tone_mark(self, word: str, tone: str) -> str:
        """
        Insert a tone combining mark (like '\u0301') into the first main vowel.
        """
        # Normalize to NFC to keep the diacritics combined properly
        word = unicodedata.normalize('NFC', word)

        # The main vowels in Vietnamese
        vowels_priority = [
            'a','â','ă','e','ê','i','o','ô','ơ','u','ư','y'
        ]
        for vowel in vowels_priority:
            idx = word.find(vowel)
            if idx != -1:
                # Insert the combining mark right after that vowel
                combined_char = vowel + tone
                return word[:idx] + combined_char + word[idx+1:]
        return word  # if no recognized vowel found, just return the original

    def encode_label(self, label: str) -> torch.Tensor:
        """
        Convert a string label to an integer label ID.
        """
        if self.config.get("task_type", None) == "seq_labeling":
            
            labels = [self.l2i[l] for l in label]
 
            return torch.Tensor(labels).long()
        else:
            return torch.tensor([self.l2i[label]], dtype=torch.long)

    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        """
        Convert integer label IDs back into strings.

        Args:
            label_vecs: 1D or 2D tensor of label IDs (e.g. shape [batch_size]).
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
            labels_out = []
            for vec in label_vecs:
                label_id = vec.item()
                labels_out.append(self.i2l[label_id])
            return labels_out

    @property
    def total_tokens_dict(self) -> dict:
        """
        Return a dict summarizing how many onset/tone/rhyme tokens exist,
        plus their sum.
        """
        return {
            'onset': len(self.itos_onset),
            'tone': len(self.itos_tone),
            'rhyme': len(self.itos_rhyme),
            'all': (len(self.itos_onset) 
                  + len(self.itos_tone) 
                  + len(self.itos_rhyme))
        }

    @property
    def total_tokens(self) -> int:
        """
        Combined count of onset, tone, and rhyme tokens.
        """
        return (len(self.itos_onset)
              + len(self.itos_tone)
              + len(self.itos_rhyme))

    @property
    def total_labels(self) -> int:
        """
        Number of distinct labels encountered in the dataset.
        """
        return len(self.l2i)
    
    @property
    def get_pad_idx(self) -> int:
        """Get the ID of the padding token."""
        return self.pad_idx[0]