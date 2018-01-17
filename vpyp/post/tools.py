import numpy as np
from corpus import ngrams


def sample_multinomial(n, dist):
    samples = np.random.multinomial(n=n, pvals=dist)
    # print(samples)
    samples = [[i for j in range(samples[i])] for i in range(len(dist)) if samples[i] > 0]
    samples = [i for sublist in samples for i in sublist]
    return samples


def increment_sentence(model, sentence, order):
    for seq in ngrams(sentence, order):
        model.increment(seq[:-1], seq[-1])


def decrement_sentence(model, sentence, order):
    for seq in ngrams(sentence, order):
        model.decrement(seq[:-1], seq[-1])