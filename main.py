# -*- coding: utf-8 -*-

import argparse
import os
import time

import chainer
from chainer import cuda, optimizers, serializers, Variable
import chainer.functions as F
import numpy as np
import pyprind

import corpus_wrapper

import models
import utils

def load_corpus(path_corpus, vocab, max_length):
    start_time = time.time()

    corpus = corpus_wrapper.CorpusWrapper(path_corpus, vocab=vocab, max_length=max_length)
    utils.logger.debug("[info] Vocabulary size: %d" % len(corpus.vocab))

    utils.logger.debug("[info] Completed. %d [sec.]" % (time.time() - start_time))
    return corpus

def make_labels(batch_sents):
    return [[i for i in xrange(len(s))]
                for s in batch_sents]

def forward(
        model, batch_sents, batch_labels,
        lmd, identity_penalty,
        train):
    ys, _ = model.forward(batch_sents, train=train) # T x (N,T)    
    ys = F.concat(ys, axis=0) # => (T*N, T)

    ts, M = utils.padding(batch_labels, head=True, with_mask=True) # => (N, T), (N, T)
    ts = ts.T # => (T, N)
    ts = ts.reshape(-1,) # => (T*N,)
    M = M[:,None,:] * M[:,:,None] # => (N, T, T)
    ts = utils.convert_ndarray_to_variable(ts, seq=False, train=train) # => (T*N,)
    M = utils.convert_ndarray_to_variable(M, seq=False, train=train) # => (N, T, T)

    loss = F.softmax_cross_entropy(ys, ts)
    acc = F.accuracy(ys, ts, ignore_label=-1)

    if identity_penalty:
        loss_id = loss_identity_penalty(ys, M, train=train)
        loss = loss + lmd * loss_id
    return loss, acc

def loss_identity_penalty(ys, M, train):
    """
    ys.data.shape = (T*N, T)
    M.data.shape = (N, T, T)
    """
    N, T, _ = M.data.shape
    
    # TODO: masking before softmax?
    ys = F.softmax(ys) # => (T*N, T)
    Y = F.reshape(ys, (T, N, T))
    Y = F.swapaxes(Y, 0, 1) # => (N, T, T) : N permutation matrices (of shape T x T)
    YYt = F.batch_matmul(Y, Y, transb=True) # => (N, T, T)

    I = cuda.cupy.eye(T, dtype=np.float32).reshape(1, T, T) # => (1, T, T)
    I = Variable(I, volatile=not train) # => (1, T, T)
    I = F.broadcast_to(I, (N, T, T)) # => (N, T, T)

    diff = YYt - I # => (N, T, T)
    diff = M * diff
    loss = F.sum(diff * diff) / F.sum(M) # => ()
    return loss

def frobenius_squared_error(W_1, W_2):
    vocab_size = float(W_1.data.shape[0])
    W_dif = W_1 - W_2
    v_dif = F.reshape(W_dif, (1,-1))
    return F.reshape(F.matmul(v_dif, v_dif, transb=True), ()) / vocab_size

def evaluate(model, corpus, lmd, identity_penalty):
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for s in pyprind.prog_bar(corpus):
        # data preparation
        batch_sents = [s]
        batch_labels = make_labels(batch_sents)
        # forward 
        loss, acc = forward(model, batch_sents, batch_labels,
                                    lmd, identity_penalty, train=False)
        total_loss += loss * len(batch_sents[0])
        total_acc += acc * len(batch_sents[0])
        count += len(batch_sents[0])
    total_loss = float(cuda.to_cpu(total_loss.data)) / count
    total_acc = float(cuda.to_cpu(total_acc.data)) / count   
    return total_loss, total_acc

def extract_word2vec(model, vocab):
    word2vec = {}
    for w in vocab.keys():
        word2vec[w] = cuda.to_cpu(model.embed.W.data[vocab[w]])
    return word2vec

def save_word2vec(path, word2vec):
    eps = 1e-6
    def normalize(v):
        return v / (np.linalg.norm(v) + eps)
    with open(path, "w") as f:
        for w, v in word2vec.items():
            line = " ".join([w] + [str(v_i) for v_i in v])
            f.write("%s\n" % line.encode("utf-8"))
    with open(path + ".normalized", "w") as f:
        for w, v in word2vec.items():
            v_normalized = normalize(v)
            line = " ".join([w] + [str(v_i) for v_i in v_normalized])
            f.write("%s\n" % line.encode("utf-8"))

def main(args):
    gpu = args.gpu
    path_config = args.config
    mode = args.mode
    path_word2vec = args.word2vec
    curriculum = False if args.curriculum == 0 else True
    
    # Hyper parameters (const)
    MAX_EPOCH = 10000000000
    MAX_PATIENCE = 20
    EVAL = 10000
    if curriculum:
        LENGTH_LIMITS = [10, 20, 30, 40, 50] # NOTE: experimental
    else:
        LENGTH_LIMITS = [50]

    config = utils.Config(path_config)
    
    # Preparaton
    path_corpus_train = config.getpath("prep_corpus") + ".train"
    path_corpus_val = config.getpath("prep_corpus") + ".val"
    basename = "won.%s.%s" % (
                    os.path.basename(path_corpus_train),
                    os.path.splitext(os.path.basename(path_config))[0])
    path_snapshot = os.path.join(config.getpath("snapshot"), basename + ".model")
    path_snapshot_vectors = os.path.join(config.getpath("snapshot"), basename + ".vectors.txt")
    if mode == "train":
        path_log = os.path.join(config.getpath("log"), basename + ".log")
        utils.set_logger(path_log)
    elif mode == "evaluation":
        path_evaluation = os.path.join(config.getpath("evaluation"), basename + ".txt")
        utils.set_logger(path_evaluation)
    elif mode == "analysis":
        path_analysis = os.path.join(config.getpath("analysis"), basename)

    utils.logger.debug("[info] TRAINING CORPUS: %s" % path_corpus_train)
    utils.logger.debug("[info] VALIDATION CORPUS: %s" % path_corpus_val)
    utils.logger.debug("[info] CONFIG: %s" % path_config)
    utils.logger.debug("[info] PRE-TRAINED WORD EMBEDDINGS: %s" % path_word2vec)
    utils.logger.debug("[info] SNAPSHOT (MODEL): %s " % path_snapshot)
    utils.logger.debug("[info] SNAPSHOT (WORD EMBEDDINGS): %s " % path_snapshot_vectors)
    if mode == "train":
        utils.logger.debug("[info] LOG: %s" % path_log)
    elif mode == "evaluation":
        utils.logger.debug("[info] EVALUATION: %s" % path_evaluation)
    elif mode == "analysis":
        utils.logger.debug("[info] ANALYSIS: %s" % path_analysis)

    # Hyper parameters
    word_dim = config.getint("word_dim")
    state_dim = config.getint("state_dim")
    aggregation = config.getstr("aggregation")
    attention = config.getstr("attention")
    retrofitting = config.getbool("retrofitting")
    alpha = config.getfloat("alpha")
    scale = config.getfloat("scale")
    identity_penalty = config.getbool("identity_penalty")
    lmd = config.getfloat("lambda")
    grad_clip = config.getfloat("grad_clip")
    weight_decay = config.getfloat("weight_decay")
    batch_size = config.getint("batch_size")

    utils.logger.debug("[info] WORD DIM: %d" % word_dim)
    utils.logger.debug("[info] STATE DIM: %d" % state_dim)
    utils.logger.debug("[info] AGGREGATION METHOD: %s" % aggregation)
    utils.logger.debug("[info] ATTENTION METHOD: %s" % attention)
    utils.logger.debug("[info] RETROFITTING: %s" % retrofitting)
    utils.logger.debug("[info] ALPHA = %f" % alpha) 
    utils.logger.debug("[info] SCALE: %f" % scale)
    utils.logger.debug("[info] IDENTITY PENALTY: %s" % identity_penalty)
    utils.logger.debug("[info] LAMBDA: %f" % lmd)
    utils.logger.debug("[info] GRADIENT CLIPPING: %f" % grad_clip)
    utils.logger.debug("[info] WEIGHT DECAY: %f" % weight_decay)
    utils.logger.debug("[info] BATCH SIZE: %d" % batch_size)

    if retrofitting:
        assert path_word2vec is not None

    # Data preparation
    corpus_train_list = [
        load_corpus(
                path_corpus_train,
                vocab=path_corpus_train + ".vocab",
                max_length=length_limit)
        for length_limit in LENGTH_LIMITS]
    corpus_val = load_corpus(
                path_corpus_val,
                vocab=corpus_train_list[0].vocab,
                max_length=LENGTH_LIMITS[-1])

    # Model preparation 
    if (mode == "train") and (path_word2vec is not None):
        initialW_data = utils.load_word2vec_weight_matrix(
                                    path_word2vec,
                                    word_dim,
                                    corpus_train_list[0].vocab,
                                    scale)
    else:
        initialW_data = None
    cuda.get_device(gpu).use()
    model = models.WON(
                vocab_size=len(corpus_train_list[0].vocab),
                word_dim=word_dim,
                state_dim=state_dim,
                aggregation=aggregation,
                attention=attention,
                initialW=initialW_data,
                EOS_ID=corpus_train_list[0].vocab["<EOS>"])
    if mode != "train":
        serializers.load_npz(path_snapshot, model)
    model.to_gpu(gpu)
    
    # Training/Evaluation/Analysis
    if mode == "train":
        length_index = 0
        utils.logger.debug("[info] Evaluating on the validation set ...")
        loss, acc = evaluate(model, corpus_val,
                                lmd, identity_penalty)
        utils.logger.debug("[validation] iter=0, epoch=0, max_length=%d, loss=%.03f, accuracy=%.2f%%" % \
                                (LENGTH_LIMITS[length_index], loss, acc*100))
        for _ in np.random.randint(0, len(corpus_val), 10):
            s = corpus_val.random_sample()
            batch_sents = [s]
            batch_labels = make_labels(batch_sents)
            _, order_pred = model.forward(batch_sents, train=False)
            order_pred = [a[0] for a in order_pred]
            order_gold = batch_labels[0]
            s = [corpus_val.ivocab[w] for w in s]
            s_pred = utils.reorder(s, order_pred)
            s_gold = utils.reorder(s, order_gold)
            s_pred = " ".join(s_pred).encode("utf-8")
            s_gold = " ".join(s_gold).encode("utf-8")
            utils.logger.debug("[check] <Gold> %s" % s_gold)
            utils.logger.debug("[check] <Pred> %s" % s_pred)
            utils.logger.debug("[check] <Gold:order> %s" % order_gold)
            utils.logger.debug("[check] <Pred:order> %s" % order_pred)
        # training & validation
        opt = optimizers.SMORMS3()
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(grad_clip))
        opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
        # best_acc = -1.0
        best_acc = acc
        patience = 0
        it = 0
        n_train = len(corpus_train_list[0]) # TODO
        finish_training = False
        for epoch in xrange(1, MAX_EPOCH+1): 
            if finish_training:
                break
            for data_i in xrange(0, n_train, batch_size):
                if data_i + batch_size > n_train:
                    break
                # data preparation
                batch_sents = corpus_train_list[length_index].next_batch(size=batch_size)
                batch_labels = make_labels(batch_sents)
                # forward
                loss, acc = forward(model, batch_sents, batch_labels,
                                    lmd, identity_penalty,
                                    train=True)
                # TODO: BEGIN
                if retrofitting:
                    part_indices_data = np.asarray(list(
                        set([w for s_ in batch_sents for w in s_])
                        ))
                    part_initialW_data = initialW_data[part_indices_data]
                
                    part_indices = Variable(cuda.cupy.asarray(part_indices_data, dtype=np.int32),
                                            volatile=False)
                    part_initialW = Variable(cuda.cupy.asarray(part_initialW_data, dtype=np.float32),
                                            volatile=False)
                    loss_ret = frobenius_squared_error(model.embed(part_indices), part_initialW)
                else:
                    loss_ret = 0.0
                loss = loss + alpha * loss_ret
                # TODO: END
                # backward & update
                model.zerograds()
                loss.backward()
                loss.unchain_backward()
                opt.update()
                it += 1
                # log
                loss = float(cuda.to_cpu(loss.data))
                acc = float(cuda.to_cpu(acc.data))
                utils.logger.debug("[training] iter=%d, epoch=%d (%d/%d=%.03f%%), max_length=%d, loss=%.03f, accuracy=%.2f%%" % \
                                    (it, epoch, 
                                    data_i+batch_size,
                                    n_train,
                                    float(data_i+batch_size)/n_train * 100,
                                    LENGTH_LIMITS[length_index],
                                    loss,
                                    acc*100))
                if it % EVAL == 0: 
                    # validation
                    utils.logger.debug("[info] Evaluating on the validation set ...")
                    loss, acc = evaluate(model, corpus_val,
                                            lmd, identity_penalty)
                    utils.logger.debug("[validation] iter=%d, epoch=%d, max_length=%d, loss=%.03f, accuracy=%.2f%%" % \
                                            (it, epoch, LENGTH_LIMITS[length_index], loss, acc*100))
                    for _ in np.random.randint(0, len(corpus_val), 10):
                        s = corpus_val.random_sample()
                        batch_sents = [s]
                        batch_labels = make_labels(batch_sents)
                        _, order_pred = model.forward(batch_sents, train=False)
                        order_pred = [a[0] for a in order_pred]
                        order_gold = batch_labels[0]
                        s = [corpus_val.ivocab[w] for w in s]
                        s_pred = utils.reorder(s, order_pred)
                        s_gold = utils.reorder(s, order_gold)
                        s_pred = " ".join(s_pred).encode("utf-8")
                        s_gold = " ".join(s_gold).encode("utf-8")
                        utils.logger.debug("[check] <Gold> %s" % s_gold)
                        utils.logger.debug("[check] <Pred> %s" % s_pred)
                        utils.logger.debug("[check] <Gold:order> %s" % order_gold)
                        utils.logger.debug("[check] <Pred:order> %s" % order_pred)

                    if best_acc < acc:
                        # save
                        utils.logger.debug("[info] Best accuracy is updated: %.2f%% => %.2f%%" % (best_acc*100.0, acc*100.0))
                        best_acc = acc
                        patience = 0
                        serializers.save_npz(path_snapshot, model)
                        serializers.save_npz(path_snapshot + ".opt", opt)
                        save_word2vec(path_snapshot_vectors, extract_word2vec(model, corpus_train_list[length_index].vocab))
                        utils.logger.debug("[info] Saved.")
                    else:
                        patience += 1
                        utils.logger.debug("[info] Patience: %d (best accuracy: %.2f%%)" % (patience, best_acc*100.0))
                        if patience >= MAX_PATIENCE:
                            if curriculum and (length_index != len(LENGTH_LIMITS)-1):
                                length_index += 1
                                break
                            else:
                                utils.logger.debug("[info] Patience %d is over. Training finished." \
                                        % patience)
                                finish_training = True
                                break
    elif mode == "evaluation":
        pass
    elif mode == "analysis":
        utils.mkdir(path_analysis)
        f = open(os.path.join(path_analysis, "dump.txt"), "w")
        data_i = 0
        for s in pyprind.prog_bar(corpus_val):
            # NOTE: analysisの場合は, 文長を気にせずすべて解かせる
            batch_sents = [s]
            batch_labels = make_labels(batch_sents)
            _, order_pred = model.forward(batch_sents, train=False)
            order_pred = [a[0] for a in order_pred]
            order_gold = batch_labels[0]
            s = [corpus_val.ivocab[w] for w in s]
            s_pred = utils.reorder(s, order_pred)
            s_gold = utils.reorder(s, order_gold)
            s_pred = " ".join(s_pred).encode("utf-8")
            s_gold = " ".join(s_gold).encode("utf-8")
            f.write("[%d] <Gold> %s\n" % (data_i+1, s_gold))
            f.write("[%d] <Pred> %s\n" % (data_i+1, s_pred))
            f.write("[%d] <Gold:order> %s\n" % (data_i+1, order_gold))
            f.write("[%d] <Pred:order> %s\n" % (data_i+1, order_pred))
            data_i += 1
        f.flush()
        f.close()

    utils.logger.debug("[info] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--word2vec", type=str, default=None)
    parser.add_argument("--curriculum", type=int, default=0)
    args = parser.parse_args()
    main(args)

