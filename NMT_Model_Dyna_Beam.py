from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import keras as k
import tensorflow as tf
import tensorflow.contrib.eager as tfe

#tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution()

import re
import os
import time
import nltk
import shlex
import shutil
import codecs
import random
import numpy as np
import configparser
import _pickle as pickle
import matplotlib.pyplot as plt

from tabulate import tabulate
from keras import backend as K
from keras.layers import Dropout
from nltk.corpus import stopwords
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from sklearn.model_selection import train_test_split


from .lang_util import *

class NMT_Model(tf.keras.Model):
#{    
    def __init__(self, encoder_inst, decoder_inst, optimizer_inst, debug_mode=False):
    #{	
        super(NMT_Model, self).__init__()
        
        self.encoder = encoder_inst
        self.decoder = decoder_inst
        self.optimizer_inst = optimizer_inst
        self.debug_mode = debug_mode
    #}

    def loss_function(self, real, pred, optimizer_inst):
    #{	
        if optimizer_inst.loss_function_api.upper() == "TENSORFLOW-API-LOSS":
        #{
            mask  = 1 - np.equal(real, 0)
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
            return tf.reduce_mean(loss_)
        #}

        elif optimizer_inst.loss_function_api.upper() == "KERAS-API-LOSS":
        #{   
            # When using tensorflow==2.0.0-alpha0

            loss_object = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)
            mask  = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(labels=real, logits=pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)                
        #}

        elif optimizer_inst.loss_function_api.upper() == "SEQ2SEQ_LOSS":
        #{
            loss = tf.contrib.seq2seq.sequence_loss(logits= pred,
                                                    targets= real,
                                                    weights= self.mask)
        #}
    #}		


    def call(self, inputs, hidden, targets, Ty, batch_size, loss, targ_lang_train, training_type, teacher_forcing_ratio, training):    
    #{
        printb("-------------------------------------(NMT_MODEL STR)-------------------------------------\n", printable=self.debug_mode)
        #Encoder inputs.shape       =(BS, Tx)                   ex: inputs       : (16, 55)
        #Encoder hidden.shape       =(BS, units)                ex: hidden       : (16, 1024)
        #Encoder enc_output.shape   =(BS, Tx, units)            ex: enc_output   : (16, 55, 1024)
        #Encoder enc_hidden_H.shape =(BS, units)                ex: enc_hidden_H : (16, 1024)
        #Encoder enc_hidden_C.shape =(BS, units)                ex: enc_hidden_C : (16, 1024)
        #dec_input.shape            =(BS, 1)                    ex: enc_hidden_C : (16, 1)
        #start.shape                =(BS, Target_Vocab_Size)    ex: enc_hidden_C : (16, 1237)

        outs        = [] 
        
        #enc_output, enc_hidden_H, enc_hidden_C = self.encoder(inputs, hidden, training)      
        #dec_hidden  = [enc_hidden_H, enc_hidden_C]
        
        enc_output, enc_state = self.encoder(inputs, hidden, training)        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        dec_state  = enc_state

        printb("[NMT_MODEL] (inputs)       -> Encoder : {}".format(inputs.shape), printable=self.debug_mode)
        #printb("[NMT_MODEL] (hidden)       -> Encoder : {}".format(hidden.shape), printable=self.debug_mode)
        printb("[NMT_MODEL] (training)     -> Encoder : {}".format(training), printable=self.debug_mode)
        printb("[NMT_MODEL] (enc_output)   <- Encoder : {}".format(enc_output.shape), printable=self.debug_mode)
        #printb("[NMT_MODEL] (dec_state)    <- Encoder : {}".format(dec_state.shape), printable=self.debug_mode)

        #for t in range(1, Ty):
        #{
        #print(">>>>>>>>>>>>>>>>.t:", t)
        # passing enc_output to the decoder

        # dec_input.shape     =(BS, 1)                   ex: dec_input       : (16, 1)                        
        # dec_hidden[0].shape =(BS, units)               ex: dec_hidden      : (16, 1024)
        # dec_hidden[1].shape =(BS, units)               ex: dec_hidden      : (16, 1024)
        # len(dec_hidden)     =List of size 2            ex: len(dec_hidden) : 2 
        # enc_output.shape    =(BS, Tx, units)           ex: enc_output      : (16, 55, 1024)
        # predictions.shape   =(BS, Target_Vocab_Size)   ex: predictions     : (16, 1237)
        # dec_hidden_H.shape  =(BS, units)               ex: dec_hidden_H    : (16, 1024)
        # dec_hidden_C.shape  =(BS, units)               ex: dec_hidden_C    : (16, 1024)

        decoder_logits, decoder_predict = self.decoder(targets, dec_state, enc_output, training)  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        printb("[NMT_MODEL] (decoder_logits)  <- Decoder : {}".format(decoder_logits.shape), printable=self.debug_mode)
        printb("[NMT_MODEL] (decoder_predict) -> Decoder : {}".format(decoder_predict.shape), printable=self.debug_mode)
 
        loss = self.loss_function(targets, decoder_logits, self.optimizer_inst)
        
        printb("-------------------------------------(NMT_MODEL END)-------------------------------------\n", printable=self.debug_mode)

        return outs, loss		
    #}
#}