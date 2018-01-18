import logging
import math


from vpyp.ngram.model import PYPLM
from vpyp.corpus import Vocabulary, read_corpus, ngrams
from vpyp.prob import Uniform


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/ngram.log',
                    filemode='w',
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

mh_iter = 100 # number of Metropolis-Hastings sampling iterations


def run_sampler(model, corpus, n_iter):
    n_sentences = len(corpus)
    n_words = sum(len(sentence) for sentence in corpus)
    for it in range(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for sentence in corpus:
            for seq in ngrams(sentence, model.order):
                if it > 0: model.decrement(seq[:-1], seq[-1])
                model.increment(seq[:-1], seq[-1])
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
    train="../data/Verne.80jours.en"
    # train = "../data/simplewiki-20140903-pages-articles.200000first.100000last.txt"
    # train = "../data/wsj.words"

    order = 3
    n_iter= 100
    vocabulary = Vocabulary()
    logging.info('Reading training corpus')
    with open(train) as train:
        training_corpus = read_corpus(train, vocabulary)
        base = Uniform(len(vocabulary))
        model = PYPLM(order, base)
        logging.info('Training model of order %d', order)
        run_sampler(model, training_corpus, n_iter)


if __name__ == '__main__':
    main()
