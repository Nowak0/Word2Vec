import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def unigram_distribution(index_frequency):
    index_frequency_array = np.array(list(index_frequency.values()))

    index_frequency_probabilities = np.power(index_frequency_array, 0.75)
    index_frequency_probabilities /= np.sum(index_frequency_probabilities)

    return index_frequency_probabilities