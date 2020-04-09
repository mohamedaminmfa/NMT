import keras as k
import tensorflow as tf

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

from .text_processing_util import *
from .lang_util import *
from .graph_plotting_util import *
from .RNN import *
from .BahdanauAttention import *
from .LuongsAttention import *




#UNIDIR_LAYER , BIDIR_LAYER: 
#[GRU, LSTM]
#[  1,    0]

#[GRU, LSTM, GRU, LSTM, GRU, LSTM, GRU, LSTM]
#[  0,    2,   0,    2,   1,    0,   1,    1]


"""
__init__

decoder = Decoder(  vocab_size= vocab_tar_size_train, 
                    embedding_dim= embedding_dim, 
                    dec_units= units, 
                    batch_size= BATCH_SIZE, 
                    decoder_layer= MODEL_DECODER_LAYER,
                    decoder_dropout= DROPOUT,
                    decoder_recurrent_dropout= RECURRENT_DROPOUT,
                    decoder_embedding_TimestepDropout=EMBEDDING_TIMESTEPDROPOUT,
                    decoder_embedding_SpatialDropout1D=EMBEDDING_SPATIALDROPOUT1D,
                    rnn_cell_mode= RNN_CELL_MODE,	
                    attention_obj= attention,
                    residual_connections_enabled= RESIDUAL_CONNECTIONS_ENABLED,
                    residual_start_from= RESIDUAL_START_FROM,
                    debug_mode= DEBUG_MODE,
                    embedding_enable=True,
                    output_fc_enable=True
                )

"""

class Decoder(tf.keras.Model):

    def __init__(self, 
                vocab_size, 
                embedding_dim, 
                dec_units, 
                batch_size,
                decoder_layer, 
                return_sequences=True, 
                return_state=True, 
                decoder_dropout=0.0, 
                decoder_recurrent_dropout=0.0,
                decoder_embedding_TimestepDropout=0.0,
                decoder_embedding_SpatialDropout1D=0.0,
                rnn_cell_mode="NATIVE",
                attention_obj= None,
                residual_connections_enabled=False, 
                residual_start_from=-1,
                debug_mode=False,
                embedding_enable=True,
                output_fc_enable=True):    

    #{    
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units  = dec_units		

        self.fc         = tf.keras.layers.Dense(vocab_size)
        
        self.output_fc_enable =  output_fc_enable
        
        if embedding_enable:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)    

        self.decoder_embedding_TimestepDropout  = decoder_embedding_TimestepDropout
        self.decoder_embedding_SpatialDropout1D = decoder_embedding_SpatialDropout1D

        self.rnn_cell_mode = rnn_cell_mode
        self.attention_obj = attention_obj
        self.residual_connections_enabled = residual_connections_enabled
        self.residual_start_from = residual_start_from
        self.debug_mode = debug_mode
        self.embedding_enable = embedding_enable

                                    
        rnn_UNI1 = RNN().rnn_keras( self,
                                    units= self.dec_units, 
                                    cell_type= cell_type,
                                    mode= rnn_cell_mode, 
                                    return_sequences= return_sequences, 
                                    return_state= return_state,
                                    input_dropout= decoder_dropout, 
                                    rec_dropout= decoder_recurrent_dropout
                                  )	
                                
        rnn_UNI2 = RNN().rnn_keras( self,
                                    units= self.dec_units, 
                                    cell_type= cell_type,
                                    mode= rnn_cell_mode, 
                                    return_sequences= return_sequences, 
                                    return_state= return_state,
                                    input_dropout= decoder_dropout, 
                                    rec_dropout= decoder_recurrent_dropout
                                  )	
    #}    

    def call(self, x, hidden, enc_output, TRAINING, t_step):  #decoder(dec_input, dec_hidden, enc_output)
    #{    
        printb("------------------------------------(DECODER {} STR)-------------------------------------\n".format(t_step), printable=self.debug_mode)
        
        printb("[DECODER] (x): {}".format(x.shape), printable=self.debug_mode)     
        
        # enc_output shape == (batch_size, max_length, hidden_size)

        attention_weights = None
        
        if self.embedding_enable:   
        #{
            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)
            printb("[DECODER] (x) [embedding(x)]: {}".format(x.shape), printable=self.debug_mode)     

            input_shape=K.shape(x)	   

            if TRAINING == True:
            #{
                x = tf.nn.dropout(x, keep_prob= 1-self.decoder_embedding_TimestepDropout,  noise_shape=(input_shape[0], input_shape[1], 1)) # (TimestepDropout) then zero-out word embeddings
                x = tf.nn.dropout(x, keep_prob= 1-self.decoder_embedding_SpatialDropout1D, noise_shape=(input_shape[0], 1, input_shape[2])) # (SpatialDropout1D) and possibly drop some dimensions on every single embedding (timestep)
                printb("[DECODER] (x) [after dropout_Embedding]: {}".format(x.shape), printable=self.debug_mode)
            #}
        #}
        
        if( self.attention_obj != None ): 
        #{
            context_vector, attention_weights = self.attention_obj(hidden[0], enc_output)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            context_dec_input = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            printb("[DECODER] (context_dec_input) [tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)]: {}".format(context_dec_input.shape), printable=self.debug_mode)
            del x
        #}
        
        else:
        #{
            context_dec_input = x
            printb("[DECODER] (context_dec_input) [context_dec_input = x]: {}".format(context_dec_input.shape), printable=self.debug_mode)
            del x
        #}
        
        printb("[DECODER] (hidden) : {}".format( hidden[0].shape), printable=self.debug_mode)

        # passing the concatenated vector to the RNN cell         
        a      = context_dec_input
        del context_dec_input

        a, stateH       = rnn_UNI1(a , initial_state= hidden, training= TRAINING)		
        output, stateH  = rnn_UNI2(a , initial_state= hidden, training= TRAINING)
        
        stateC     = stateH				

        if self.output_fc_enable == True:
        #{
            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))
            printb("[DECODER] (output) [tf.reshape(output, (-1, output.shape[2]))]: {}".format(output.shape), printable=self.debug_mode)
            
            # output shape == (batch_size * 1, vocab)
            output = self.fc(output)			
            printb("[DECODER] (output) [fc(output)]: {}".format(output.shape), printable=self.debug_mode)
        #}
        
        printb("[DECODER] (stateH) :{}".format(stateH.shape), printable=self.debug_mode)
        printb("[DECODER] (stateC) :{}".format(stateC.shape), printable=self.debug_mode)

        printb("--------------------------------------(DECODER END)--------------------------------------\n", printable=self.debug_mode)

        return output, stateH, stateC, attention_weights
    #}

    def initialize_hidden_state(self):
    #{
        return tf.zeros((self.batch_size, self.dec_units))
    #}
#}