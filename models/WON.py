# -*- coding: utf-8 -*-

import sys

import numpy as np
import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L

import utils


class WON(chainer.Chain):

    def __init__(self,
            vocab_size,
            word_dim,
            state_dim,
            aggregation,
            attention,
            initialW,
            EOS_ID):
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.state_dim = state_dim
        self.aggregation = aggregation
        self.attention = attention
        self.initialW = initialW
        self.EOS_ID = EOS_ID
        
        if initialW is not None:
            assert initialW.shape[0] == vocab_size
            assert initialW.shape[1] == word_dim
            tmp = np.random.RandomState(1234).uniform(-0.01, 0.01, (vocab_size+1, word_dim)) 
            tmp[0:-1, :] = initialW
            initialW = tmp
        else:
            initialW = None
        self.vocab_size_in = self.vocab_size + 1
        self.BOS_ID = self.vocab_size_in - 1

        super(WON, self).__init__(
                embed = L.EmbedID(self.vocab_size_in, self.word_dim,
                                    ignore_label=-1, initialW=initialW),
                
                W_dinit=L.Linear(self.word_dim, self.state_dim),
                
                # LSTM ver.
                W_upd=L.Linear(self.word_dim, 4 * self.state_dim),
                U_upd=L.Linear(self.state_dim, 4 * self.state_dim, nobias=True),
                # GRU ver.
                # Wz_upd=L.Linear(self.word_dim, self.state_dim),
                # Uz_upd=L.Linear(self.state_dim, self.state_dim, nobias=True),
                # Wr_upd=L.Linear(self.word_dim, self.state_dim),
                # Ur_upd=L.Linear(self.state_dim, self.state_dim, nobias=True),
                # W_upd=L.Linear(self.word_dim, self.state_dim),
                # U_upd=L.Linear(self.state_dim, self.state_dim, nobias=True),

                W_att=L.Linear(self.word_dim, self.state_dim),
                U_att=L.Linear(self.state_dim, self.state_dim, nobias=True),
                u_att=L.Linear(self.state_dim, 1, nobias=True),
                )
        # LSTM ver.
        self.U_upd.W.data[self.state_dim*0:self.state_dim*1, :] = self.init_ortho(self.state_dim)
        self.U_upd.W.data[self.state_dim*1:self.state_dim*2, :] = self.init_ortho(self.state_dim)
        self.U_upd.W.data[self.state_dim*2:self.state_dim*3, :] = self.init_ortho(self.state_dim)
        self.U_upd.W.data[self.state_dim*3:self.state_dim*4, :] = self.init_ortho(self.state_dim)
        # GRU ver.
        # self.Uz_upd.W.data = self.init_ortho(self.state_dim)
        # self.Ur_upd.W.data = self.init_ortho(self.state_dim)
        # self.U_upd.W.data = self.init_ortho(self.state_dim)

    def init_ortho(self, dim):
        A = np.random.randn(dim, dim)
        U, S, V = np.linalg.svd(A)
        return U.astype(np.float32)
        
    def forward(self, xs, train):
        """
        xs: [[int=ID]]
        ys: T x (N,T)
        """
        xs, ms = utils.padding(xs, head=True, with_mask=True) # (N, T), (N, T)
        xs = utils.convert_ndarray_to_variable(xs, seq=True, train=train) # T x (N,)
        ms = utils.convert_ndarray_to_variable(ms, seq=True, train=train) # T x (N,)

        es = self.embed_words(xs, train=train)
        g = self.aggregate(es, ms)
        # Note: Here, we assume that "es" follows the original order
        ys, outputs = self.reorder(es, g, ms, train=train) # bottleneck
        return ys, outputs

    def embed_words(self, xs, train):
        es = [self.embed(x) for x in xs]
        return es
    
    def aggregate(self, es, ms):
        """
        es: T x (N,D)
        ms: T x (N,)
        """
        T = len(es)
        N = es[0].data.shape[0]

        g = F.concat(es, axis=0) #  (T*N, D)
        ms = F.concat(ms, axis=0) #  (T*N,)

        g = F.reshape(g, (T, N, self.word_dim)) #  (T, N, D)
        ms = F.reshape(ms, (T, N)) # (T, N)

        g = F.swapaxes(g, 0, 1) #  (N, T, D)
        ms = F.swapaxes(ms, 0, 1) # (N, T)

        ms_brd = F.broadcast_to(ms[:,:,None], (N, T, self.word_dim)) # (N, T, D)

        if self.aggregation == "sum":
            g = g * ms_brd # (N, T, D)
            g = F.sum(g, axis=1) #  (N, D)
        elif self.aggregation == "avg":
            g = g * ms_brd # (N, T, D)
            g = F.sum(g, axis=1) #  (N, D)
            g = g / F.broadcast_to(
                    F.sum(ms, axis=1)[:, None], (N, self.word_dim))
        elif self.aggregation == "max":
            g = g - 1000.0 * (1.0 - ms_brd) # (N, T, D)
            g = F.max(g, axis=1) #  (N, D)
        else:
            utils.logger.debug("[error] Unknown aggregation method: %s" % self.aggregation)
        return g

    def reorder(self, es, g, ms, train):
        """
        es: T x [(N,D)]
        g: (N, D)
        ms: T x [(N,1)]
        """
        ys = []
        outputs = []

        T = len(es)
        N = es[0].data.shape[0]

        es_ = F.concat(es, axis=0) # (T*N, D)
        W_es = self.W_att(es_) # (T*N, D)
        ms_ = F.concat(ms, axis=0) #  (T*N,1)
        ms_ = F.reshape(ms_, (T, N)) # (T, N)
        
        # LSTM ver.
        state = {
            "d": F.tanh(self.W_dinit(g)), # (N, D)
            "c": Variable(cuda.cupy.zeros((N, self.state_dim), dtype=np.float32), volatile=not train)}
        # GRU ver.
        # d = F.tanh(self.W_dinit(g)) # (N, D)

        bos = Variable(cuda.full((N,1), self.BOS_ID, dtype=np.int32), volatile=not train)
        bos = self.embed(bos) # because we do NOT embed es in update_state()
        xs = [bos] + es[:-1] # input (embedding) sequence for the decoder RNN
        for x in xs:
            # LSTM ver.
            state = self.update_state(x, state, train=train) # about 30%
            y = self.attend(es_, W_es, state["d"], ms_) # about 70% (bottleneck)
            # GRU ver.
            # d = self.update_state(x, d, train=train) # about 30%
            # y = self.attend(es_, W_es, d, ms_) # about 70% (bottleneck)

            output = np.argmax(cuda.to_cpu(y.data), axis=1)
            outputs.append(output)
            ys.append(y)
        
        return ys, outputs
    
    # LSTM ver.
    def update_state(self, e, state, train):
        d_in = self.W_upd(e) + self.U_upd(state["d"])
        c, d = F.lstm(state["c"], d_in)
        state = {"d": d, "c": c}
        return state
    # GRU ver.
    # def update_state(self, e, d, train):
    #     z = F.sigmoid(self.Wz_upd(e) + self.Uz_upd(d))
    #     r = F.sigmoid(self.Wr_upd(e) + self.Ur_upd(d))
    #     _d = F.tanh(self.W_upd(e) + self.U_upd(r * d))
    #     return (1.0 - z) * d + z * _d

    def attend(self, es_, W_es, d, ms_):
        """
        es_.data.shape = (T*N, D)
        W_es.data.shape: (T*N, D)
        d.data.shape: (N, D)
        ms_.data.shape = (T, N)
        """
        T, N = ms_.data.shape
        
        d_ = d[None, :, :] #  (1, N, D)
        d_ = F.broadcast_to(d_, (T, N, self.state_dim)) #  (T, N, D)
        d_ = F.reshape(d_, (T*N, self.state_dim)) #  (T*N, D)
        
        # attention
        if self.attention == "innerproduct":
            # inner product
            y = F.batch_matmul(es_, d_, transa=True) # (T*N, 1, 1)
        elif self.attention == "bilinear":
            # bilinear (テスト済み)
            y = F.batch_matmul(W_es, d_, transa=True) #  (T*N, 1, 1)
        elif self.attention == "nonlinear":
            # nonlinear
            y = self.u_att(F.tanh(W_es + self.U_att(d_))) #  (T*N, 1)
        else:
            utils.logger.debug("[error] Unknown attention method: %s" % self.attention)
            sys.exit(-1)
        y = F.reshape(y, (T, N)) #  (T, N)

        # masking
        y = y - 100.0 * (1.0 - ms_) # (T, N)
            
        # transpose
        # with cuda.get_device(y.data) as device:
        #     y = F.transpose(y) #  (N, T)
        #     y = F.copy(y, dst=int(device))
        y = F.swapaxes(y, 0, 1) #  (N, T)

        return y

