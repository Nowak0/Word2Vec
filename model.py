import numpy as np
from utils import sigmoid


class Model:
    GENERATIONS = 10
    WINDOW = 2
    EMBEDDING_SIZE = 50
    N_NEGATIVE_SAMPLES = 5
    EPSILON = 1e-10
    LEARNING_RATE = 1e-2

    def __init__(self, train_data, test_data, word_as_index):
        self.skip_gram_pairs = []
        self.train_data = train_data
        self.test_data = test_data
        self.word_as_index = word_as_index
        self.vocabulary_size = len(self.word_as_index)
        self.embedding_center = np.random.uniform(-0.01, 0.01, (self.vocabulary_size, self.EMBEDDING_SIZE))
        self.embedding_context = np.random.uniform(-0.01, 0.01, (self.vocabulary_size, self.EMBEDDING_SIZE))

    def train(self):
        self.create_skip_gram_association()

        for i in range(self.GENERATIONS):
            total_loss = 0
            for (center, context) in self.skip_gram_pairs:
                loss = self.calculate_loss(center, context)
                total_loss += loss
            print(f"Iteration: {i+1}, loss: {total_loss}")


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

        np.random.shuffle(self.skip_gram_pairs)

    def calculate_loss(self, index_center, index_context):
        value_center = self.embedding_center[index_center]
        value_context = self.embedding_context[index_context]

        score_positive = np.dot(value_center, value_context)
        probability_positive = sigmoid(score_positive)
        loss = -np.log(probability_positive + self.EPSILON)

        gradient_center = (probability_positive - 1) * value_context
        gradient_context = (probability_positive - 1) * value_center

        negative_samples = self.get_random_negative_samples(index_context)
        for n in negative_samples:
            value_negative = self.embedding_context[n]
            score_negative = np.dot(value_center, value_negative)
            probability_negative = sigmoid(-score_negative)
            loss += (-np.log(probability_negative + self.EPSILON))

            gradient_center += probability_negative * value_negative
            gradient_negative = probability_negative * value_center

            self.embedding_context[n] -= self.LEARNING_RATE * gradient_negative

        self.embedding_center[index_center] -= self.LEARNING_RATE * gradient_center
        self.embedding_context[index_context] -= self.LEARNING_RATE * gradient_context

        return loss


    def get_random_negative_samples(self, blocked_index):
        negative_samples = []

        while len(negative_samples) < self.N_NEGATIVE_SAMPLES:
            sample = np.random.randint(0, len(self.word_as_index))
            if sample != blocked_index and sample not in negative_samples:
                negative_samples.append(sample)

        return negative_samples