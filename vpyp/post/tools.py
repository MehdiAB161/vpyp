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


def tags_words(sentence):
    tags = [sentence[i][0] for i in range(len(sentence))]
    words = [sentence[i][1] for i in range(len(sentence))]
    return tags, words


def corpus_ll(word_model, tag_model, corpus):
    return sum(sentence_ll(word_model, tag_model, sentence) for sentence in corpus)


# TODO : Check the likelihood computation functions
def word_tag_ll(word_model, tag_model, tag_ctx, tag, word) :
    return np.log(word_model.prob(tag, word)) + np.log(tag_model.prob(tag_ctx, tag))


def sentence_ll(word_model, tag_model, sentence):
    tags, words = tags_words(sentence)
    tag_sequence = list(ngrams(tags, order=3))
    return sum(word_tag_ll(word_model, tag_model, tag_ctx=tag_sequence[i][:-1], tag=tag_sequence[i][-1], word=words[i])
               for i in range(len(words)))


# TODO : Check that this function works
def decrement_word_tag_corpus(corpus, word_model, tag_model, w):
    for sentence in corpus :
        i = 0
        for seq in sentence:
            i += 1
            if seq[1] == w:
                tag_sequence = ngrams(sentence, order=5)
                # TODO : Check this indexes
                tag_ctx = tag_sequence[i][0:2]
                w_tag = tag_sequence[i][2]

                # Decrement the word from the word model
                word_model.decrement(ctx=w_tag, w=w)
                # Decrement the associated tag from the tag model
                tag_model.decrement(ctx=tag_ctx, w=w_tag)


# TODO : Check that this function works
def unique_words(corpus):
    words_sets = [set(sentence) for sentence in corpus]
    word_types = set.union(*words_sets)
    return list(word_types)


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