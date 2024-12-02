import re
from unicodedata import normalize
import unicodedata


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
    final_consonants = ["m", "n", "p", "t", "ng", "nh", "c", "ch", "u", "o", "i", "y"]

    for consonant in sorted(final_consonants, key=len, reverse=True):
        if van.endswith(consonant):
            am_cuoi = consonant
            van = van[:-len(consonant)]  # Remove final consonant from van
            break

    # 3. The rest is âm chính (main vowel)
    am_chinh = van

    return am_dem, am_chinh, am_cuoi


def split_vietnamese_word(self, word):
    """
    Split a Vietnamese word into Âm đầu (initial consonants), Thanh Điệu (tone marks),
    and Vần (rhyme, including accents but without tone).
    """

    if word.isnumeric():
        return word

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

    # 3. Remove tone marks from the word
    van = self.remove_tone_marks(decomposed_word)
    van = unicodedata.normalize('NFC', van)  # Recompose the word after removing tones

    # 4. Further split van into âm đệm, âm chính, âm cuối
    am_dem, am_chinh, am_cuoi = self.split_van(van)

    return am_dau, tone, am_dem, am_chinh, am_cuoi





def syllable_split(syllable: str):
    special_tokens = ["!", "?", ":", ";", ",", "\"", "'", "%", "^", "`", "~",
                      "(", ")", "[", "]", "/", ".", "-", "$", "&", "*"]
    if syllable in special_tokens:
        return syllable

    if syllable.isnumeric():
        return syllable

    onset = ""
    rhyme = ""
    tone = 0

    vowels = ['a', 'e', 'i', 'o', 'u', 'y']

    normalized_word = normalize('NFD', syllable)
    # huyền, sắc, ngã, hỏi, nặng
    tones = [0x0300, 0x0301, 0x0303, 0x0309, 0x0323]
    tones_tokens = ["<grave>", "<acute>",
                    "<tile>", "<hook_above>", "<dot_below>"]

    for i in range(len(normalized_word)):
        if normalized_word[i] in vowels:
            onset = normalized_word[:i]
            rhyme = normalized_word[i:]
            break

    # Handle special case
    if onset == "g" and rhyme[0] == "i":
        onset = "gi"
        rhyme = rhyme[1:]
    has_tone = False
    for i in range(len(rhyme)):
        tone = ord(rhyme[i])
        for j in range(len(tones)):
            if tone == tones[j]:
                rhyme = rhyme[:i] + rhyme[i + 1:]
                tone = tones_tokens[j]
                has_tone = True
        if has_tone:
            return str(onset) + " " + str(rhyme) + " " + str(tone)
    return str(onset) + " " + str(rhyme)


def preprocess_sentence(sentence: str):
    sentence = sentence.lower()
    # sentence = normalize("NFD", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    sentence = re.sub(r"%", " % ", sentence)

    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()

    return tokens


if __name__ == "__main__":
    setences = "xin chào"
    preprocess = preprocess_sentence(setences)
    print(preprocess)
