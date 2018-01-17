import logging
from corpus import Vocabulary, read_corpus
from prob import Uniform
from model import PYPLM
from train import run_mh_sampler, run_sentence_sampler

mh_iter = 100 # number of Metropolis-Hastings sampling iterations


logging.basicConfig(level=logging.INFO, format='%(message)s')

args = {"train": "../data/Verne.80jours.en",
        "tag_order": 3,
        "word_order": 1,
        "iter": 100}

vocabulary = Vocabulary()

logging.info('Reading training corpus')
with open(args["train"]) as train:
    training_corpus = read_corpus(train, vocabulary)

base = Uniform(len(vocabulary))
tag_model = PYPLM(args["tag_order"], base)
word_model = PYPLM(args["word_order"], base)


logging.info('Training model of order %d', args["tag_order"])
run_mh_sampler(tag_model, training_corpus, args["iter"], mh_iter)
# run_sentence_sampler(word_model, tag_model, training_corpus, args["iter"], mh_iter)

