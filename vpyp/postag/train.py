import argparse
import logging
import math
import cPickle
from corpus import Vocabulary, read_corpus, ngrams
from prob import Uniform
from pyp import PYP
from prior import PYPPrior
from model import PYPLM

from new_functions import sample_multinomial

# mh_iter = 100  # number of Metropolis-Hastings sampling iterations


def run_sampler(model, corpus, n_iter, P):
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)
    for it in range(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for sentence in corpus:
            n = 0

            # We decrement the transitions and emissions of the sentence from the table arrangements
            for seq in ngrams(sentence, 3):

                if it > 0: model.decrement(seq[0:2], seq[2])
                model.increment(seq[:-1], seq[-1])

                # Remove the current sentence
                model.decrement(seq[:-1], seq[-1])

                # TODO : decrement emissions

            for seq in ngrams(sentence, 5):

                # We compute all the possible sequences by replacing the middle tag
                possible_sequences = [seq[0:2] + [w] + seq[4:6] for w in all_tags]

                # We compute the proposal distribution
                proposal_distribution = [model.prob(ctx=seq[0:2], w=seq[2]) * model.prob(ctx=seq[1:3], w=seq[3])
                                         * model.prob(ctx=seq[2:4], w=seq[5]) for seq in possible_sequences]


                # Pick P particles from the proposal distribution

                particles = sample_multinomial(P, proposal_distribution)

                # Update the seating arrangements for all the particles
                for particle in particles :
                    new_seq = seq[0:2] + all_tags[particle] + seq[3:5]

                    model.increment(new_seq[0:2], new_seq[2])
                    model.increment(new_seq[1:3], new_seq[3])
                    model.increment(new_seq[2:4], new_seq[4])

                # Compute the weights of the sampled particles with the new sampling arrangements (Not sure about this one)
                alphas = [model.prob(ctx=seq[0:2], w=all_tags[particle]) / proposal_distribution[particle]
                          for particle in particles]
                weights = [weights[i] * alphas[i] for i in range(len(weights))]

                # Resample the particles to obtain P equally weighted particles
                particles = sample_multinomial(P, weights)

        if it % 10 == 0:
            logging.info('Model: %s', model)
            ll = model.log_likelihood()
            ppl = math.exp(-ll / (n_words + n_sentences))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)
        if it % 30 == 29:
            logging.info('Resampling hyperparameters...')
            acceptance, rejection = model.resample_hyperparemeters(mh_iter)
            arate = acceptance / float(acceptance + rejection)
            logging.info('Metropolis-Hastings acceptance rate: %.4f', arate)
            logging.info('Model: %s', model)
            ll = model.log_likelihood()
            ppl = math.exp(-ll / (n_words + n_sentences))
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train n-gram model')
    parser.add_argument('--train', help='training corpus', required=True)
    parser.add_argument('--order', help='order of the model', type=int, required=True)
    parser.add_argument('--iter', help='number of iterations', type=int, required=True)
    parser.add_argument('--output', help='model output path')
    parser.add_argument('--P', help='Number of particles')

    args = parser.parse_args()

    vocabulary = Vocabulary()

    logging.info('Reading training corpus')
    with open(args.train) as train:
        training_corpus = read_corpus(train, vocabulary)

    if args.charlm:
        from ..charlm import CharLM
        char_lm = CharLM(args.charlm, vocabulary)
        if args.pyp:
            base = PYP(char_lm, PYPPrior(1.0, 1.0, 1.0, 1.0, 0.8, 1.0))
        else:
            base = char_lm
    else:
        base = Uniform(len(vocabulary))
    model = PYPLM(args.order, base)

    logging.info('Training model of order %d', args.order)
    run_sampler(model, training_corpus, args.iter)

    if args.output:
        model.vocabulary = vocabulary
        with open(args.output, 'w') as f:
            cPickle.dump(model, f, protocol=-1)


if __name__ == '__main__':
    main()
