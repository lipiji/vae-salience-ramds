#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class DotAttentionLayer(object):
    def __init__(self, layer_id, shape, sent_encs, cmt_encs, sent_decs, pz):
        prefix = "AttentionLayer_"
        layer_id = "_" + layer_id
        self.num_summs, self.num_sents, self.num_cmts, self.out_size = shape
        
        self.W_a1 = init_weights((self.out_size, self.out_size), prefix + "W_a1" + layer_id)
        self.W_a2 = init_weights((self.num_summs, 1), prefix + "W_a2" + layer_id)
        self.W_a3 = init_weights((self.num_summs, 1), prefix + "W_a3" + layer_id)
        self.W_a4 = init_weights((self.num_summs, 1), prefix + "W_a4" + layer_id)

        a = T.nnet.softmax(T.dot(sent_decs, sent_encs.T))
        c = T.dot(a, sent_encs)

        a2 = T.nnet.softmax(T.dot(sent_decs, cmt_encs.T))
        pz = T.repeat(pz, self.num_summs, axis=0)
        c2 = T.dot(a2 * pz, cmt_encs)



        # new sentence codes
        self.activation = T.tanh(T.dot(sent_decs, self.W_a1)) * T.repeat(self.W_a2, self.out_size, axis=1) \
        + c * T.repeat(self.W_a3, self.out_size, axis=1) \
        + c2 * T.repeat(self.W_a4, self.out_size, axis=1) 
        self.A = a
        self.A2 = a2
        self.params = [self.W_a1, self.W_a2, self.W_a3,  self.W_a4]
