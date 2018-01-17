import logging
import math

from particle import Particles
from tools import random_tags, decrement_tag_sentence, decrement_word_sentence, increment_tag_sentence,\
    increment_word_sentence


def run_sentence_sampler(corpus, word_model, tag_model, n_tags, n_iter, n_particles):
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)

    # Assign words and random tags to the table counts
    # TODO : Check if it is necessary to fill word and tag restaurants
    corpus = random_tags(corpus, n_tags=n_tags)

    for sentence in corpus:
        tags = [sentence[i][0] for i in range(len(sentence))]
        increment_word_sentence(model=word_model, sentence=sentence)
        increment_tag_sentence(model=tag_model, sentence=tags, order=tag_model.order)

    for it in range(n_iter):
        new_corpus = []

        logging.info('Iteration %d/%d', it + 1, n_iter)

        s = 0

        for sentence in corpus:
            if s < 10 :
                print(s)
                s += 1

                len_sentence = len(sentence)

                words = [sentence[i][1] for i in range(len_sentence)]
                tags = [sentence[i][0] for i in range(len_sentence)]

                # TODO : Check if necessary to decrement tag seats
                # Remove the words and tags from the table counts
                decrement_word_sentence(model=word_model, sentence=sentence)
                decrement_tag_sentence(model=tag_model, sentence=tags, order=tag_model.order)

                # Particle filter
                particles = Particles(n_particles, tag_model, word_model, sentence, n_tags)
                new_tags = particles.particle_filter()
                new_sentence = [(new_tags[i], words[i]) for i in range(len_sentence)]

                # Assign the new tags and words to the table counts
                increment_word_sentence(word_model, new_sentence)
                increment_tag_sentence(tag_model, new_tags, tag_model.order)

                # Add the new word tag pairs to the new corpus
                new_corpus.append(new_sentence)

                # Compute the model likelihood
                if i % 20 == 0:
                    logging.info('Model: %s', word_model)
                    ll = word_model.log_likelihood()
                    ppl = math.exp(-ll / (n_words + n_sentences))
                    logging.info('LL=%.0f ppl=%.3f', ll, ppl)

        # Update the old corpus
        corpus = new_corpus

        # Compute the model likelihood
        if it % 5 == 0:
            logging.info('Model: %s', word_model)
            ll = word_model.log_likelihood()
            ppl = math.exp(-ll / (n_words + n_sentences))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)