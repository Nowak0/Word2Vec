import re


RARE_TRESHOLD = 1
N_COMMON_WORD = 10


def get_data():
    with open("dataset/frankenstein.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text = re.sub(r'[^\w\s\.]', '', text)
    raw_sentences = text.split(".")
    sentences = [s.lower().split() for s in raw_sentences]

    return sentences


def assign_word_to_index(data):
    word_as_index = {}
    index_as_word = {}
    index_frequency = {}

    index = 0
    for sentence in data:
        for word in sentence:
            if word not in word_as_index:
                word_as_index[word] = index
                index_as_word[index] = word
                index_frequency[index] = 1
                index += 1
            else:
                index_frequency[word_as_index[word]] += 1

    return word_as_index, index_as_word, index_frequency