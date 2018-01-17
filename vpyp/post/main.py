import logging

from train import run_sentence_sampler
from corpus import read_corpus, Vocabulary
from prob import Uniform
from model import PYPLM

from parameters import args

logging.basicConfig(level=logging.INFO, format='%(message)s')

logging.info('Reading training corpus')
vocabulary = Vocabulary()
with open(args["train_file"]) as train:
    corpus = read_corpus(train, vocabulary)

tag_model = PYPLM(args["tag_order"], initial_base=Uniform(args["n_tags"]))
word_model = PYPLM(args["word_order"], initial_base=Uniform(len(vocabulary)))

logging.info('Training model of order %d', args["tag_order"])

run_sentence_sampler(corpus, word_model, tag_model,
                     n_tags=args["n_tags"],
                     n_iter=args["n_iter"],
                     n_particles=args["n_particles"])