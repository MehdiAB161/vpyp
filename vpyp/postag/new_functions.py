import numpy as np


def sample_multinomial(P, dist) :
    samples = np.random.multinomial(n=P, pvals=dist)
    # print(samples)
    samples = [[i for j in range(samples[i])] for i in range(len(dist)) if samples[i] > 0]
    samples = [i for sublist in samples for i in sublist]
    return samples