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

from NMT_lib.text_processing_util import *
from NMT_lib.lang_util import *
from NMT_lib.graph_plotting_util import *
from NMT_lib.RNN import *
from NMT_lib.BahdanauAttention import *
from NMT_lib.LuongsAttention import *
from NMT_lib.Encoder import *
from NMT_lib.Decoder import *
from NMT_lib.Optimizer import *
from NMT_lib.NMT_Model import *



class NMT_Model(tf.keras.Model):
#{    
    def __init__(self):
        super(NMT_Model, self).__init__()
        #self.predictions = []
        
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
    #}		

	
    def call(self, inputs, hidden, targets, Ty, batch_size, encoder, decoder, loss, optimizer_inst, targ_lang_train, training):    
	#{
        #Encoder inputs.shape       =(BS, Tx)                   ex: inputs       : (16, 55)
        #Encoder hidden.shape       =(BS, units)                ex: hidden       : (16, 1024)
        #Encoder enc_output.shape   =(BS, Tx, units)            ex: enc_output   : (16, 55, 1024)
        #Encoder enc_hidden_H.shape =(BS, units)                ex: enc_hidden_H : (16, 1024)
        #Encoder enc_hidden_C.shape =(BS, units)                ex: enc_hidden_C : (16, 1024)
        #dec_input.shape            =(BS, 1)                    ex: enc_hidden_C : (16, 1)
        #start.shape                =(BS, Target_Vocab_Size)    ex: enc_hidden_C : (16, 1237)

        outs        = [] 
        enc_output, enc_hidden_H, enc_hidden_C = encoder(inputs, hidden, training)        
        dec_hidden  = [enc_hidden_H, enc_hidden_C] 
        
        del inputs
        del hidden
        del enc_hidden_H
        del enc_hidden_C		
		
        dec_input   = tf.expand_dims([targ_lang_train.word2idx['<start>']] * batch_size, 1)            
        
        start = tf.one_hot( np.full((batch_size),targ_lang_train.word2idx['<start>']), len(targ_lang_train.word2idx))
        outs.append( start )

        # Teacher forcing - feeding the target as the next input
        for t in range(1, Ty):
		#{
          
            # passing enc_output to the decoder
            
            # dec_input.shape     =(BS, 1)                   ex: dec_input       : (16, 1)                        
            # dec_hidden[0].shape =(BS, units)               ex: dec_hidden      : (16, 1024)
            # dec_hidden[1].shape =(BS, units)               ex: dec_hidden      : (16, 1024)
            # len(dec_hidden)     =List of size 2            ex: len(dec_hidden) : 2 
            # enc_output.shape    =(BS, Tx, units)           ex: enc_output      : (16, 55, 1024)
            # predictions.shape   =(BS, Target_Vocab_Size)   ex: predictions     : (16, 1237)
            # dec_hidden_H.shape  =(BS, units)               ex: dec_hidden_H    : (16, 1024)
            # dec_hidden_C.shape  =(BS, units)               ex: dec_hidden_C    : (16, 1024)

            predictions, dec_hidden_H, dec_hidden_C, _ = decoder(dec_input, dec_hidden, enc_output, training)
            dec_hidden  = [dec_hidden_H, dec_hidden_C]
			
            del dec_hidden_H
            del dec_hidden_C			

            loss += self.loss_function(targets[:, t], predictions, optimizer_inst)

            outs.append(predictions)
            
            if training==True:
                # using teacher forcing
                dec_input = tf.expand_dims(targets[:, t], 1) 
           
            else:        
                predicted_id = tf.argmax(predictions, axis=-1)          
                predicted_id = np.reshape(predicted_id, (len(predicted_id),1))
                dec_input    = tf.convert_to_tensor( predicted_id)
         #}       

        return outs, loss		
	#}
#}