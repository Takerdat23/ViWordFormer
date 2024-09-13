from .vietnamese_mapping import VietnameseMapping
import unicodedata

vietnamese_mapping = VietnameseMapping()

def split_tone(word: str):
    tone_map = {
        '\u0300': '<grave>',
        '\u0301': '<acute>',
        '\u0303': '<tilde>',
        '\u0309': '<hook_above>',
        '\u0323': '<dot_below>',
    }
    decomposed_word = unicodedata.normalize('NFD', word)
    tone = ''
    remaining_word = ''
    for char in decomposed_word:
        if char in tone_map:
            tone = tone_map[char]
        else:
            remaining_word += char
    remaining_word = unicodedata.normalize('NFC', remaining_word)
    
    kwargs = {
        'tone': tone,
        'remaining_word': remaining_word,
    }
    return kwargs


def split_word(word: str):
    onset_medial_map = VietnameseMapping().graph_onset_medial
    medial_nucleus_map = VietnameseMapping().graph_medial_nucleus
    nucleus_coda_map = VietnameseMapping().graph_nucleus_coda
    
    parts = split_tone(word)
    tone = parts['tone']
    base_word = parts['remaining_word']
    
    for onset, medial_list in onset_medial_map.items():
        if base_word.startswith(onset):
            remaining_word = base_word[len(onset):]
            _temp1 = remaining_word
            for medial in medial_list:
                remaining_word = _temp1
                if remaining_word.startswith(medial):
                    remaining_word = remaining_word[len(medial):]
                    _temp2 = remaining_word
                    for nucleus in medial_nucleus_map[medial]:
                        remaining_word = _temp2 
                        if remaining_word.startswith(nucleus):
                            remaining_word = remaining_word[len(nucleus):]
                            _temp3 = remaining_word
                            for coda in nucleus_coda_map[nucleus]:
                                remaining_word = _temp3
                                if remaining_word == coda:
                                    kwargs = {
                                        'is_vietnamese': True,
                                        'onset': onset,
                                        'medial': medial,
                                        'nucleus': nucleus,
                                        'coda': coda,
                                        'tone': tone,
                                    }
                                    return kwargs
    # If non satisfied return False
    return {'is_vietnamese': False, 'word': word}
    
    
special_case = set() 
# # Run python word_splitter.py if you want to test the function
# sentences = "Tôi yêu Việt Nam. Tôi yêu người Việt Nam.".lower()
# sentences = "test luon ap cậ dóng goi min that thien hct thuyền khuyên chuyến khuyến thê thêm này giá"
# words = sentences.split()
# for word in words:
#     dict_word = split_word(word)
#     if dict_word['is_vietnamese']:
#         print(f"vi: {dict_word}")
#     else:
#         print(f"non-vi: {dict_word}")
