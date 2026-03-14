import re


TRAIN_PERCENTAGE = 0.8


def get_data():
    with open("dataset/frankenstein.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text = re.sub(r'[^\w\s\.]', '', text)
    raw_sentences = text.split(".")
    sentences = [s.lower().split() for s in raw_sentences]

    train_data_length = int(len(sentences) * TRAIN_PERCENTAGE)
    train_data = sentences[:train_data_length]
    test_data = sentences[train_data_length:]

    return train_data, test_data


def assign_word_to_index(test_data):
    word_as_index = {}
    index_as_word = {}

    index = 0
    for sentence in test_data:
        for word in sentence:
            if word not in word_as_index:
                word_as_index[word] = index
                index_as_word[index] = word
                index += 1

    return word_as_index, index_as_word