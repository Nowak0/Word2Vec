from handle_data import *
from model import Model


def main():
    train_data, test_data = get_data()
    word_as_index, index_as_word = assign_word_to_index(train_data)
    model = Model(train_data, test_data, word_as_index)
    model.train()


if __name__ == "__main__":
    main()