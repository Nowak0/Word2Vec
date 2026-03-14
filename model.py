import numpy as np


class Model():
    WINDOW = 2
    EMBEDDING_SIZE = 50

    def __init__(self, train_data, test_data, word_as_index):
        self.skip_gram_pairs = []
        self.train_data = train_data
        self.test_data = test_data
        self.word_as_index = word_as_index
        self.vocabulary_size = len(self.word_as_index)
        self.embedding_center = np.random.uniform(-0.01, 0.01, (self.vocabulary_size, self.EMBEDDING_SIZE))
        self.embedding_context = np.random.uniform(-0.01, 0.01, (self.vocabulary_size, self.EMBEDDING_SIZE))

    def create_skip_gram_association(self):
        for sentence in self.train_data:
            for i in range(len(sentence)):
                numerical_value_center = self.word_as_index[sentence[i]]

                for j in range(max(0, i-self.WINDOW), i):
                    numerical_value_context = self.word_as_index[sentence[j]]
                    self.skip_gram_pairs.append((numerical_value_center, numerical_value_context))

                for k in range(i+1, min(len(sentence), i+self.WINDOW+1)):
                    numerical_value_context = self.word_as_index[sentence[k]]
                    self.skip_gram_pairs.append((numerical_value_center, numerical_value_context))

    def train(self):
        self.create_skip_gram_association()
