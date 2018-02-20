# -*- coding: utf-8 -*-

import logging
from logging import getLogger, Formatter, StreamHandler, DEBUG
import os
import sys

import numpy as np
from chainer import cuda, Variable

##############################
# logging

logger = getLogger("logger")
logger.setLevel(DEBUG)

handler = StreamHandler()
handler.setLevel(DEBUG)
handler.setFormatter(Formatter(fmt="%(message)s"))
logger.addHandler(handler)

def set_logger(filename):
    if os.path.exists(filename):
        logger.debug("[info] A file %s already exists." % filename)
        do_remove = raw_input("[info] Delete the existing log file? [y/n]: ")
        if (not do_remove.lower().startswith("y")) and (not len(do_remove) == 0):
            logger.debug("[info] Done.")
            sys.exit(0)
    logging.basicConfig(level=DEBUG, format="%(message)s", filename=filename, filemode="w")

##############################
# pre-trained word embeddings

def load_word2vec_weight_matrix(path, dim, vocab, scale):
    word2vec = load_word2vec(path, dim=dim)
    W = convert_word2vec_to_weight_matrix(vocab, word2vec, dim=dim, scale=scale)
    return W

def load_word2vec(path, dim):
    word2vec = {}
    with open(path) as f:
        for line_i, line in enumerate(f):
            l = line.strip().split()
            if len(l[1:]) != dim:
                logger.debug("[info] dim %d(actual) != %d(expected), skipped line %d" % \
                        (len(l[1:]), dim, line_i+1))
                continue
            word2vec[l[0].decode("utf-8")] = np.asarray([float(x) for x in l[1:]])
    return word2vec

def convert_word2vec_to_weight_matrix(vocab, word2vec, dim, scale):
    task_vocab = vocab.keys()
    logger.debug("[info] Vocabulary size (corpus): %d" % len(task_vocab))
    word2vec_vocab = word2vec.keys()
    logger.debug("[info] Vocabulary size (pre-trained): %d" % len(word2vec_vocab))
    common_vocab = set(task_vocab) & set(word2vec_vocab)
    logger.debug("[info] Pre-trained words in the corpus: %d (%d/%d = %.2f%%)" \
            % (len(common_vocab), len(common_vocab), len(task_vocab),
                float(len(common_vocab))/len(task_vocab)*100))
    W = np.random.RandomState(1234).uniform(-scale, scale, (len(task_vocab), dim)).astype(np.float32)
    for w in common_vocab:
        W[vocab[w], :] = word2vec[w]
    return W

##############################
# NN

def padding(xs, head, with_mask):
    N = len(xs)
    max_length = max([len(x) for x in xs])
    ys = np.zeros((N, max_length), dtype=np.int32)
    if head:
        for i in xrange(N):
            l = len(xs[i])
            ys[i, 0:l] = xs[i]
            ys[i, l:] = -1
    else:
        for i in xrange(N):
            l = len(xs[i])
            ys[i, 0:max_length-l] = -1
            ys[i, max_length-l:] = xs[i]
    if with_mask:
        ms = np.greater(ys, -1).astype(np.float32)
        return ys, ms
    else:
        return ys

def convert_ndarray_to_variable(xs, seq, train):
    """
    xs.shape = (N, T)
    """
    if seq:
        return [Variable(cuda.cupy.asarray(xs[:,j]), volatile=not train)
                    for j in xrange(xs.shape[1])]
    else:
        return Variable(cuda.cupy.asarray(xs), volatile=not train)

##############################
# others

def mkdir(path, newdir=None):
    if newdir is None:
        target = path
    else:
        target = os.path.join(path, newdir)
    if not os.path.exists(target):
        os.makedirs(target)
        logger.debug("[info] Created a directory: %s" % target)

def reorder(xs, order):
    assert len(xs) == len(order)
    return [xs[i] for i in order]


