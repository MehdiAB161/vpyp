import numpy as np


def sample_multinomial(P, dist) :
    dist = np.arange(5) / np.sum(np.arange(5))
    samples = np.random.multinomial(n=5, pvals=dist)
    print(samples)
    samples = [[i for j in range(samples[i])] for i in range(P) if samples[i] > 0]
    samples = [i for sublist in samples for i in sublist]
    return samples