from handle_data import *
from model import Model


def main():
    data = get_data()
    word_as_index, index_as_word, index_frequency = assign_word_to_index(data)

    model = Model(data, word_as_index, index_frequency)
    model.train()

    print("sister", model.evaluation_most_similar("sister", index_as_word))
    print("home", model.evaluation_most_similar("home", index_as_word))
    print("man", model.evaluation_most_similar("man", index_as_word))


if __name__ == "__main__":
    main()