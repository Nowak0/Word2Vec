# Skip-gram Word2Vec Implementation

This project is a foundational implementation of Word2Vec, specifically the **Skip-gram** architecture with **Negative Sampling**. Developed from scratch using Python and NumPy, it aims to demonstrate the mathematical foundations behind learning dense vector representations of words from a text dataset.

## Project Objective

The primary goal is to map words from a dataset (here: Mary Shelley's *Frankenstein*) into a continuous vector space where words that appear in similar contexts obtain similar vector representations. This project focuses on the optimization of word embeddings using gradient descent.

## Dataset

The dataset was preprocessed through tokenization, lowercasing and removal of punctuation before building the vocabulary. It includes:
- ~75000 total words
- ~7200 unique words
- ~540000 Skip-gram pairs without subsampling
- ~330000 Skip-gram pairs after subsampling

## Core Implementation

The Skip-gram model learns word embeddings by predicting surrounding context words given a center word. The implementation features several critical components:

1.  **Forward Pass**:
    - The model maintains two embedding matrices: one for center words and one for context words. For each center–context pair, the dot product between their vectors produces a score representing their similarity. This score is then passed through a sigmoid function to estimate the probability that the context word appears near the center word.

2.  **Negative Sampling & Loss Function**:
    - The loss encourages embeddings of true center–context pairs to have high similarity, while embeddings of randomly sampled word pairs are pushed apart through negative sampling. The objective maximizes the log-sigmoid of the positive pair while minimizing the log-sigmoid of negatively sampled pairs.

3.  **Subsampling of Frequent Words**:
    - Frequent words are probabilistically discarded using a frequency-based subsampling function, which reduces the dominance of very common words such as “the” or “and”.

4.  **Stochastic Gradient Descent**:
    - The model updates weights incrementally after each training pair. Gradients are derived from the loss function and applied to both the center and context vectors using a fixed learning rate, allowing the embeddings to progressively converge toward useful representations.

## Training Parameters

The following hyperparameters were utilized to produce the analyzed results:

- **Epochs**: 10
- **Embedding Size**: 50
- **Learning Rate**: 0.01
- **Negative Samples**: 10
- **Subsampling Numerator**: 0.001
- **Windows Tested**: 3, 4 and 5

---

## Analysis of Results

The performance was evaluated using cosine similarity between embedding vectors across different window sizes to compare the results of standard training versus subsampling method.

### 1. Performance without Subsampling
| Window | Test Word | Top 5 Similar Words (Similarity Score) |
| :--- | :--- | :--- |
| **3** | sister | son (0.93), home (0.92), tenderly (0.92), wife (0.92), brother (0.91) |
| | home | return (0.94), henry (0.94), family (0.93), condemned (0.93), sufferings (0.93) |
| | man | old (0.88), young (0.88), woman (0.86), felix (0.85), creature (0.84) |
| **4** | sister | wife (0.94), brother (0.94), constant (0.92), darling (0.92), friend (0.92) |
| | home | return (0.93), necessary (0.92), cast (0.92), afterwards (0.91), clerval (0.91) |
| | man | old (0.84), felix (0.84), woman (0.84), young (0.83), pursuit (0.82) |
| **5** | sister | beloved (0.93), darling (0.93), brothers (0.92), wife (0.92), smile (0.92) |
| | home | bitterly (0.92), return (0.92), condemned (0.92), dwelling (0.92), boy (0.92) |
| | man | old (0.89), young (0.83), felix (0.81), blind (0.80), woman (0.80) |

**Observations**:
Without subsampling, the model captures meaningful semantic associations across all window sizes. It effectively captures familial relationships ("sister" -> "brother", "wife") and gender-based pairings ("man" -> "woman"). The number of center–context examples is significantly larger, resulting in longer training time.

### 2. Performance with Subsampling
| Window | Test Word | Top 5 Similar Words (Similarity Score) |
| :--- | :--- | :--- |
| **3** | sister | console (0.96), relate (0.96), imagine (0.95), possibly (0.94), credit (0.94) |
| | home | confess (0.95), leave (0.95), proved (0.95), absence (0.94), restore (0.94) |
| | man | companion (0.92), young (0.92), woman (0.92), period (0.92), child (0.91) |
| **4** | sister | gratitude (0.96), duty (0.96), evil (0.96), guilt (0.95), enough (0.95) |
| | home | agitates (0.96), shake (0.96), express (0.96), inhabit (0.96), preserve (0.95) |
| | man | old (0.90), child (0.88), girl (0.87), woman (0.87), spoke (0.86) |
| **5** | sister | beloved (0.97), lady (0.96), loved (0.96), constant (0.96), brother (0.95) |
| | home| lady (0.98), situation (0.97), blessed (0.97), benefactor (0.97), tenderly (0.97) |
| | man| old (0.86), felix (0.86), blind (0.85), name (0.85), woman (0.85) |

**Observations**:
While subsampling offers a substantial increase in training speed, it introduces a trade-off in result quality that is highly dependent on the window size. At smaller windows (3, 4), the results are less intuitive. This is likely because subsampling reduces the number of training pairs involving frequent words, which can decrease the amount of useful training information available to the model. The window size of 5 produced semantically meaningful similar words. A larger dataset could further improve embedding quality by providing more contextual information.

## Summary
The experiments highlight how context window size and subsampling affect embedding quality. Subsampling was intentionally applied to reduce training time and focus on meaningful words and while it can reduce the number of effective training examples with smaller windows, a window size of 5 provided the most stable and semantically meaningful results in this implementation.