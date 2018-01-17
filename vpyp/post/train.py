import logging
import math

from particle import Particles
from tools import increment_sentence, decrement_sentence


def run_sentence_sampler(word_model, tag_model, corpus, n_iter, n_particles, n_tags) :
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)

    # Assign uniform randomly a tag to each word in the corpus
    corpus.random_tags(n_tags)

    # Assign words and random tags to the table counts
    # TODO : Check if it is necessary to fill word and tag restaurants
    for sentence in corpus:
        words = [sentence[i][1] for i in range(len(sentence))]
        tags = [sentence[i][0] for i in range(len(sentence))]
        increment_sentence(word_model, words, word_model.order)
        increment_sentence(tag_model, tags, tag_model.order)

    for it in range(n_iter):
        logging.info('Iteration %d/%d', it + 1, n_iter)

        for sentence in corpus:
            words = [sentence[i][1] for i in range(len(sentence))]
            tags = [sentence[i][0] for i in range(len(sentence))]

            # TODO : Check if necessary to decrement tag seats
            # Remove the words and tags from the table counts
            decrement_sentence(word_model, words, word_model.order)
            decrement_sentence(tag_model, tags, tag_model.order)

            # Particle filter
            particles = Particles(n_particles, tag_model, word_model, sentence, n_tags)
            new_tags = particles.particle_filter()

            # Assign the new tags and words to the table counts
            increment_sentence(word_model, words, word_model.order)
            increment_sentence(tag_model, new_tags, tag_model.order)

        # Compute the model likelihood
        if it % 10 == 0:
            logging.info('Model: %s', word_model)
            ll = word_model.log_likelihood()
            ppl = math.exp(-ll / (n_words + n_sentences))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)