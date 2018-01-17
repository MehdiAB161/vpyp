import logging
import math

from tools import write_to_file
from particle import Particles
from tools import random_tags, decrement_tag_sentence, decrement_word_sentence, increment_tag_sentence,\
    increment_word_sentence


def run_sentence_sampler(corpus, word_models, tag_models, n_tags, n_iter, n_particles, log_dir):
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)

    # Log data
    ll_list =[]
    ppl_list = []

    # Assign words and random tags to the table counts
    # TODO : Check if it is necessary to fill word and tag restaurants at the initialization
    corpus = random_tags(corpus, n_tags=n_tags)

    # for sentence in corpus:
    #     tags = [sentence[i][0] for i in range(len(sentence))]
    #     for word_model in word_models :
    #         increment_word_sentence(model=word_model, sentence=sentence)
    #     for tag_model in tag_models :
    #         increment_tag_sentence(model=tag_model, sentence=tags, order=tag_model.order)

    for it in range(n_iter):
        new_corpus = []

        logging.info('Iteration %d/%d', it + 1, n_iter)

        for sentence in corpus:

            len_sentence = len(sentence)

            words = [sentence[i][1] for i in range(len_sentence)]
            tags = [sentence[i][0] for i in range(len_sentence)]

            # TODO : Check if necessary to decrement tag seats on first iteration
            if it > 0 :
                # Remove the words and tags from the table counts
                for word_model in word_models:
                    decrement_word_sentence(model=word_model, sentence=sentence)
                for tag_model in tag_models:
                    decrement_tag_sentence(model=tag_model, sentence=tags, order=tag_model.order)

            # Particle filter
            particles = Particles(n_particles, tag_models, word_models, sentence, n_tags)
            new_tags = particles.particle_filter()
            new_sentence = [(new_tags[i], words[i]) for i in range(len_sentence)]

            # Assign the new tags and words to the table counts
            for word_model in word_models:
                increment_word_sentence(word_model, new_sentence)
            for tag_model in tag_models:
                increment_tag_sentence(tag_model, new_tags, tag_model.order)

            # Add the new word tag pairs to the new corpus
            new_corpus.append(new_sentence)

        # Update the old corpus
        corpus = new_corpus

        # Compute the model likelihood
        logging.info('Model: %s', word_model[0])
        ll = word_model[0].log_likelihood()
        ppl = math.exp(-ll / (n_words + n_sentences))
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)

        # Log the temporary data (just in case :))
        ll_list.append(ll)
        ppl_list.append(ppl)
        write_to_file(filename=log_dir + '/ll_temp', data=ll_list)
        write_to_file(filename=log_dir + '/ppl_temp', data=ppl_list)
    return ll_list, ppl_list
