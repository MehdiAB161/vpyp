import numpy as np
from corpus import ngrams
import os
import errno


def sample_multinomial(n, dist):
    samples = np.random.multinomial(n=n, pvals=dist)
    samples = [[i for j in range(samples[i])] for i in range(len(dist)) if samples[i] > 0]
    samples = [i for sublist in samples for i in sublist]
    return samples


def increment_tag_sentence(model, sentence, order):
    for seq in ngrams(sentence, order):
        model.increment(ctx=seq[:-1], w=seq[-1])


def increment_word_sentence(model, sentence):
    for seq in sentence:
        model.increment(ctx=seq[0], w=seq[1])


def decrement_tag_sentence(model, sentence, order):
    for seq in ngrams(sentence, order):
        model.decrement(ctx=seq[:-1], w=seq[-1])


def decrement_word_sentence(model, sentence):
    for seq in sentence:
        model.decrement(ctx=seq[0], w=seq[1])


def random_tags(corpus, n_tags):
    return [[(np.random.randint(1, n_tags + 1), word) for word in sentence] for sentence in corpus]


def write_to_file(filename, data) :
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(filename, "w") as f:
        f.write(str(data))