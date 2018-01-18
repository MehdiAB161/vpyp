import logging
import math
from copy import deepcopy
from tqdm import tqdm

from particle import Particles
from tools import *


def run_sentence_sampler(corpus, word_models, tag_models, n_tags, n_iter, n_particles, log_dir):
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)
    model_type = "sentence"

    word_model = word_models[0]
    tag_model = tag_models[0]

    # Log data
    ll_list = []
    ppl_list = []

    # TODO : Check if it is necessary to fill word and tag restaurants at the initialization
    corpus = random_tags(corpus, n_tags=n_tags)

    for it in tqdm(range(n_iter)):
        new_corpus = []

        logging.info('Iteration %d/%d', it + 1, n_iter)

        for sentence in corpus:

            tags, words = tags_words(sentence)

            if it > 0 :
                # Remove the words and tags from the table counts
                decrement_word_sentence(model=word_model, sentence=sentence)
                decrement_tag_sentence(model=tag_model, sentence=tags, order=tag_model.order)

            # Particle filter
            particles = Particles(n_particles, tag_model, word_model, sentence, n_tags, model_type)
            filtered_particle = particles.particle_filter()
            new_tags = filtered_particle.tag_assignments
            new_sentence = [(new_tags[i], words[i]) for i in range(len(sentence))]

            # Assign the new tags and words to the table counts
            increment_word_sentence(word_model, new_sentence)
            increment_tag_sentence(tag_model, new_tags, tag_model.order)

            # Add the new word tag pairs to the new corpus
            new_corpus.append(new_sentence)

        # Update the old corpus
        corpus = new_corpus

        # Compute the model likelihood
        logging.info('Model: %s', word_model)
        # ll = corpus_ll(word_model, tag_model, corpus)
        ll = word_model.log_likelihood() + tag_model.log_likelihood()
        ppl = math.exp(-ll / (n_words + n_sentences))
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)

        # Log the temporary data (just in case :))
        ll_list.append(ll)
        ppl_list.append(ppl)
        write_to_file(filename=log_dir + '/ll_temp', data=ll_list)
        write_to_file(filename=log_dir + '/ppl_temp', data=ppl_list)

    return ll_list, ppl_list


def run_word_sampler(corpus, word_models, tag_models, n_tags, n_iter, n_particles, log_dir):
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)
    model_type = "word"

    # Log data
    ll_list = []
    ppl_list = []

    # Assign words and random tags to the table counts
    # TODO : Check if it is necessary to fill word and tag restaurants at the initialization
    corpus = random_tags(corpus, n_tags=n_tags)
    word_types = unique_words(corpus)

    for it in tqdm(range(n_iter)):

        logging.info('Iteration %d/%d', it + 1, n_iter)

        for word in tqdm(word_types):
            logging.info('Word-type iteration')
            new_corpus = []

            # Remove a word from the table counts
            if it > 0 :
                for i in range(n_particles):
                    decrement_word_tag_corpus(corpus, word_models[i], tag_models[i], word)

            for sentence in corpus:

                len_sentence = len(sentence)
                tags = [sentence[i][0] for i in range(len_sentence)]
                words = [sentence[i][1] for i in range(len_sentence)]

                # Particle filter
                particles = Particles(n_particles, tag_models, word_models, sentence, n_tags, model_type)
                filtered_particle = particles.particle_filter()
                new_tags = filtered_particle.tag_assignments
                new_sentence = [(new_tags[i], words[i]) for i in range(len_sentence)]

                # Update the seating arrangements of the sentence with the new sentence
                if it > 0:
                    # Remove the words and tag assignments from the old sentence
                    for word_model in word_models:
                        decrement_word_sentence(model=word_model, sentence=sentence)
                    for tag_model in tag_models:
                        decrement_tag_sentence(model=tag_model, sentence=tags, order=tag_model.order)
                # Assign the new tags and the word tags couples from the new sentence to the table counts
                for word_model in word_models:
                    increment_word_sentence(word_model, new_sentence)
                for tag_model in tag_models:
                    increment_tag_sentence(tag_model, new_tags, tag_model.order)

                # Add the new word tag pairs to the new corpus
                new_corpus.append(new_sentence)

            # TODO : the models are incremented in the particle filter and after here

            corpus = new_corpus

            # Copy the sampled particle model to all the particles
            word_models = [deepcopy(filtered_particle.word_model) for _ in range(n_particles)]
            tag_models = [deepcopy(filtered_particle.tag_model) for _ in range(n_particles)]

            # Compute the model likelihood
            logging.info('Model: %s', word_models[0])
            ll = word_models[0].log_likelihood()
            ppl = math.exp(-ll / (n_words + n_sentences))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)

            # Log the temporary data (just in case :))
            ll_list.append(ll)
            ppl_list.append(ppl)
            write_to_file(filename=log_dir + '/ll_temp', data=ll_list)
            write_to_file(filename=log_dir + '/ppl_temp', data=ppl_list)
