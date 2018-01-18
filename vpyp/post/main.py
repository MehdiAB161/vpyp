import logging

from train import run_sentence_sampler
from corpus import read_corpus, Vocabulary
from prob import Uniform
from model import PYPLM

from time import time
from time import strftime

from tools import write_to_file
from parameters import args

now = strftime("%c")
log_dir = "logs/" + now.format(time())
write_to_file(filename=log_dir + '/args', data=args)

logging.basicConfig(level=logging.INFO, format='%(message)s')

logging.info('Reading training corpus')
vocabulary = Vocabulary()
with open(args["train_file"]) as train:
    corpus = read_corpus(train, vocabulary)

# tag_models = [PYPLM(args["tag_order"], initial_base=Uniform(args["n_tags"])) for _ in range(args["n_particles"])]
# word_models = [PYPLM(args["word_order"], initial_base=Uniform(len(vocabulary))) for _ in range(args["n_particles"])]

tag_models = [PYPLM(args["tag_order"], initial_base=Uniform(args["n_tags"]))]
word_models = [PYPLM(args["word_order"], initial_base=Uniform(len(vocabulary)))]

logging.info('Training model of order %d', args["tag_order"])

ll_list, ppl_list = run_sentence_sampler(corpus, word_models, tag_models,
                                         n_tags=args["n_tags"],
                                         n_iter=args["n_iter"],
                                         n_particles=args["n_particles"],
                                         log_dir=log_dir)


write_to_file(filename=log_dir + '/ll', data=ll_list)
write_to_file(filename=log_dir + '/ppl', data=ppl_list)