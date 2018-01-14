import argparse
import logging
import math
import cPickle
from corpus import Vocabulary, read_corpus, ngrams
from prob import Uniform
from pyp import PYP
from prior import PYPPrior
from model import PYPLM
import numpy as np

from new_functions import sample_multinomial

# mh_iter = 100  # number of Metropolis-Hastings sampling iterations


def run_sampler(model, corpus, n_iter, n_particles, n_tags):
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)
    weights = [float(1)/n_particles for i in range(n_particles)]
    all_tags = np.arange(n_tags)


    for it in range(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for sentence in corpus:
            # TODO : decrement emissions
            for seq in ngrams(sentence, 5):
                if it > 0:
                    model.decrement(seq[0:2], seq[2])
                    model.decrement(seq[1:3], seq[3])
                    model.decrement(seq[2:4], seq[4])

                # We compute all the possible sequences by replacing the middle tag
                possible_sequences = [seq[0:2] + (w,) + seq[3:5] for w in all_tags]

                # We compute the proposal distribution
                proposal_distribution = [model.prob(ctx=seq[0:2], w=seq[2]) * model.prob(ctx=seq[1:3], w=seq[3])
                                         * model.prob(ctx=seq[2:4], w=seq[4]) for seq in possible_sequences]

                # Pick P particles from the proposal distribution
                temp_proposal_distribution = [proposal_distribution[i]/sum(proposal_distribution) for i in range(len(proposal_distribution))]

                particles = sample_multinomial(n_particles, temp_proposal_distribution)

                # Compute the weights of the sampled particles with the new sampling arrangements (Not sure about this one)
                alphas = [model.prob(ctx=seq[0:2], w=all_tags[particle]) / proposal_distribution[particle]
                          for particle in particles]
                weights = [weights[i] * alphas[i] for i in range(len(weights))]
                weights = [weights[i]/sum(weights) for i in range(len(weights))]

                # Resample the particles to obtain P equally weighted particles
                particles_indexes = sample_multinomial(n_particles, weights)
                particles = [particles[particles_indexes[i]] for i in range(len(particles))]

                # Update the seating arrangements for all the particles
                for particle in particles:
                    new_seq = seq[0:2] + (all_tags[particle],) + seq[3:5]

                    model.increment(new_seq[0:2], new_seq[2])
                    model.increment(new_seq[1:3], new_seq[3])
                    model.increment(new_seq[2:4], new_seq[4])

        if it % 10 == 0:
            logging.info('Model: %s', model)
            ll = model.log_likelihood()
            ppl = math.exp(-ll / (n_words + n_sentences))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        # if it % 30 == 29:
        #     logging.info('Resampling hyperparameters...')
        #     acceptance, rejection = model.resample_hyperparemeters(mh_iter)
        #     arate = acceptance / float(acceptance + rejection)
        #     logging.info('Metropolis-Hastings acceptance rate: %.4f', arate)
        #     logging.info('Model: %s', model)
        #     ll = model.log_likelihood()
        #     ppl = math.exp(-ll / (n_words + n_sentences))
        #     logging.info('LL=%.0f ppl=%.3f', ll, ppl)


def main():
    train="../data/Verne.80jours-short.en"
    order = 3
    P=20
    T=17
    n_iter=100
    vocabulary = Vocabulary()
    logging.info('Reading training corpus')
    training_corpus = read_corpus(train, vocabulary)
    base = Uniform(len(vocabulary))
    model = PYPLM(order, base)
    logging.info('Training model of order %d', order)
    run_sampler(model, training_corpus, n_iter, n_particles=P, n_tags=T)



if __name__ == '__main__':
    main()
