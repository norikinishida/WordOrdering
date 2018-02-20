# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pyprind

import utils
from main import load_corpus

def count_sentence_length(corpus, count):
    for s in pyprind.prog_bar(corpus):
        length = len(s)
        if length >= len(count):
            continue
        count[length] += 1
    return count

def plot_histogram(count):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    keys = np.arange(len(count))
    ax.hist(keys, weights=count, bins=len(count))
    ax.set_title("Sentence Length Distribution")
    ax.set_xlabel("Length")
    ax.set_ylabel("Frequency")
    fig.show()

def main():
    config = utils.Config()
    
    path_corpus_train = config.getpath("prep_corpus") + ".train"
    path_corpus_val = config.getpath("prep_corpus") + ".val"
    corpus_train = load_corpus(
                    path_corpus_train,
                    vocab=path_corpus_train + ".vocab",
                    max_length=1000000000)
    corpus_val = load_corpus(path_corpus_val,
                    vocab=corpus_train.vocab,
                    max_length=1000000000)
    
    count = np.zeros((101,))
    count = count_sentence_length(corpus_train, count=count)
    count = count_sentence_length(corpus_val, count=count)
    
    diff = len(corpus_train) + len(corpus_val) - count.sum()
    utils.logger.debug("[info] Excluded %d sentences of length longer than %d" % (diff, len(count)-1))
    
    path_out = config.getpath("prep_corpus") + ".histogram.npy"
    np.save(path_out, count)

    plot_histogram(count)

if __name__ == "__main__":
    main()
