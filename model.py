import numpy as np
from utils import sigmoid, unigram_distribution


class Model:
    EPOCHS = 10
    WINDOW = 5
    EMBEDDING_SIZE = 50
    N_NEGATIVE_SAMPLES = 10
    EPSILON = 1e-10
    LEARNING_RATE = 0.01
    SUBSAMPLING_NUMERATOR = 1e-3

    def __init__(self, data, word_as_index, index_frequency=None):
        self.skip_gram_pairs = []
        self.index_frequency_probabilities = []
        self.data = data
        self.word_as_index = word_as_index
        self.index_frequency = index_frequency
        self.total_word_count = np.sum(list(self.index_frequency.values()))
        self.vocabulary_size = len(self.word_as_index)
        self.embedding_center = np.random.uniform(-0.01, 0.01, (self.vocabulary_size, self.EMBEDDING_SIZE))
        self.embedding_context = np.random.uniform(-0.01, 0.01, (self.vocabulary_size, self.EMBEDDING_SIZE))

    def train(self):
        if self.index_frequency is not None:
            self.index_frequency_probabilities = unigram_distribution(self.index_frequency)
        self._create_skip_gram_pairs()

        for i in range(self.EPOCHS):
            np.random.shuffle(self.skip_gram_pairs)
            total_loss = 0

            for (center, context) in self.skip_gram_pairs:
                loss = self._calculate_loss(center, context)
                total_loss += loss

            print(f"Iteration: {i+1}, loss: {total_loss}")

    def evaluation_most_similar(self, word, index_as_word, n_elements=5):
        index = self.word_as_index[word]
        vector = self.embedding_center[index]

        similarities = np.dot(self.embedding_center, vector) / (
            np.linalg.norm(self.embedding_center, axis=1) *
            np.linalg.norm(vector) + self.EPSILON
        )

        nearest = similarities.argsort()[::-1][1:n_elements+1]

        return [(index_as_word[i], similarities[i]) for i in nearest]

    def _create_skip_gram_pairs(self):
        self.data = self._subsampling(self.data)

        for sentence in self.data:
            for i in range(len(sentence)):
                numerical_value_center = self.word_as_index[sentence[i]]

                for j in range(max(0, i-self.WINDOW), i):
                    numerical_value_context = self.word_as_index[sentence[j]]
                    self.skip_gram_pairs.append((numerical_value_center, numerical_value_context))

                for k in range(i+1, min(len(sentence), i+self.WINDOW+1)):
                    numerical_value_context = self.word_as_index[sentence[k]]
                    self.skip_gram_pairs.append((numerical_value_center, numerical_value_context))

    def _subsampling(self, data):
        new_data = []

        for sentence in data:
            new_sentence = []
            for word in sentence:
                freq = self.index_frequency[self.word_as_index[word]] / self.total_word_count
                discard_prob = max(0.0, 1 - np.sqrt(self.SUBSAMPLING_NUMERATOR / freq))

                if np.random.rand() >= discard_prob:
                    new_sentence.append(word)

            if new_sentence:
                new_data.append(new_sentence)

        return new_data

    def _calculate_loss(self, index_center, index_context):
        vector_center = self.embedding_center[index_center]
        vector_context = self.embedding_context[index_context]

        gradient_center, gradient_context, loss_positive = self._handle_positive_sample(vector_center, vector_context)
        gradient_center, loss_negative = self._handle_negative_samples(gradient_center, index_context, vector_center)

        self.embedding_center[index_center] -= self.LEARNING_RATE * gradient_center
        self.embedding_context[index_context] -= self.LEARNING_RATE * gradient_context

        return loss_positive + loss_negative

    def _handle_positive_sample(self, vector_center, vector_context):
        score_positive = np.dot(vector_center, vector_context)
        probability_positive = sigmoid(score_positive)
        loss_positive = -np.log(probability_positive + self.EPSILON)

        gradient_center = (probability_positive - 1) * vector_context
        gradient_context = (probability_positive - 1) * vector_center

        return gradient_center, gradient_context, loss_positive

    def _handle_negative_samples(self, gradient_center, index_context, vector_center):
        negative_samples = self._get_random_negative_samples(index_context)
        loss = 0
        for n in negative_samples:
            vector_negative = self.embedding_context[n]
            score_negative = np.dot(vector_center, vector_negative)
            probability_negative = sigmoid(-score_negative)
            loss += (-np.log(probability_negative + self.EPSILON))

            gradient_center += (1 - probability_negative) * vector_negative
            gradient_negative = (1 - probability_negative) * vector_center

            self.embedding_context[n] -= self.LEARNING_RATE * gradient_negative

        return gradient_center, loss

    def _get_random_negative_samples(self, blocked_index):
        samples = []

        if self.index_frequency is not None:
            samples = np.random.choice(
                len(self.word_as_index),
                size=self.N_NEGATIVE_SAMPLES,
                p=self.index_frequency_probabilities
            )
            samples = [s for s in samples if s != blocked_index]

        else:
            while len(samples) < self.N_NEGATIVE_SAMPLES:
                sample = np.random.randint(0, len(self.word_as_index))
                if sample != blocked_index and sample not in samples:
                    samples.append(sample)

        return samples