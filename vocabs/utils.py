import re
from unicodedata import normalize


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
    sentence = normalize("NFD", sentence)
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

    words = sentence.strip().split()

    tokens = [syllable_split(word) for word in words]

    return tokens


if __name__ == "__main__":
    setences = "đc"
    preprocess = preprocess_sentence(setences)
    print(preprocess)
