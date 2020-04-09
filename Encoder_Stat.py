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

#UNIDIR_LAYER , BIDIR_LAYER: 
#[GRU, LSTM]
#[  1,    0]

#[GRU, LSTM, GRU, LSTM, GRU, LSTM, GRU, LSTM]
#[  0,    2,   0,    2,   1,    0,   1,    1]

#encoder_dropout                      = [[0.2], [0.2, 0.3]]
#encoder_recurrent_dropout            = [[0.2], [0.2, 0.3]]

"""
__init__

encoder = Encoder(  vocab_size= vocab_inp_size_train, 
                    embedding_dim= embedding_dim, 
                    enc_units= units, 
                    batch_size= BATCH_SIZE, 
                    encoder_layer= MODEL_ENCODER_LAYER,
                    encoder_dropout= DROPOUT,
                    encoder_recurrent_dropout= RECURRENT_DROPOUT,
                    encoder_embedding_TimestepDropout=EMBEDDING_TIMESTEPDROPOUT,
                    encoder_embedding_SpatialDropout1D=EMBEDDING_SPATIALDROPOUT1D,
                    rnn_cell_mode= RNN_CELL_MODE,
                    reverse_input_tensor= REVERSE_SOURCE_INPUT,
                    residual_connections_enabled= RESIDUAL_CONNECTIONS_ENABLED,
                    residual_start_from= RESIDUAL_START_FROM,
                    debug_mode= DEBUG_MODE,
                    embedding_enable=True
                )

"""

class Encoder(tf.keras.Model):
#{

    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 enc_units, 
                 batch_size, 
                 num_encoder_bi_layers,
				 num_encoder_uni_layers,
                 return_sequences=True, 
                 return_state=True,
                 encoder_dropout= 0.0,
                 encoder_recurrent_dropout= 0.0,
                 encoder_embedding_TimestepDropout= 0.0,
                 encoder_embedding_SpatialDropout1D= 0.0,
                 input_dropout= False,  
                 output_dropout= False, 
                 state_dropout= False, 
                 variational_rec= False,
                 cell_type= "GRU",
                 rnn_cell_mode="NATIVE",
                 reverse_input_tensor=False,
                 residual_connections_enabled=False, 
                 residual_start_from=-1,
                 debug_mode=False,
                 embedding_enable=True
                 ):
    #{    
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.enc_units  = enc_units
        
        if embedding_enable:
            self.embedding  = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.encoder_embedding_TimestepDropout  = encoder_embedding_TimestepDropout
            self.encoder_embedding_SpatialDropout1D = encoder_embedding_SpatialDropout1D


        self.rnn_cell_mode = rnn_cell_mode
        self.cell_type= cell_type
        self.reverse_input_tensor = reverse_input_tensor
        self.residual_connections_enabled = residual_connections_enabled
        self.residual_start_from = residual_start_from
        self.debug_mode = debug_mode
        self.embedding_enable = embedding_enable        
        
        rnn_BI = RNN().bidirectional_rnn_keras(	self,
                                                units= self.enc_units, 
                                                cell_type= cell_type,
                                                mode= rnn_cell_mode,
                                                return_sequences= return_sequences,													   
                                                dropout= encoder_dropout, 
                                                rec_dropout= encoder_recurrent_dropout
                                              )   
                                
        rnn_UNI1 = RNN().rnn_keras( self,
                                    units= self.enc_units,  
                                    cell_type= cell_type,
                                    mode= rnn_cell_mode, 
                                    return_sequences= return_sequences, 
                                    return_state= return_state,
                                    dropout= encoder_dropout, 
                                    rec_dropout= encoder_recurrent_dropout
                                  )						
    #}
    
    def call(self, x, hidden, TRAINING):  #encoder(inp, hidden)             called from NMT_Model.py
    #{
        printb("--------------------------------------(ENCODER STR)--------------------------------------\n", printable=self.debug_mode)
        
        printb("[ENCODER] (x):{}".format(x.shape) , printable=self.debug_mode)
        printb("[ENCODER] (hidden) : {}".format(hidden.shape), printable=self.debug_mode)
        
        if self.reverse_input_tensor:
            x = reverse_tensor(x)
        
        if self.embedding_enable:
        #{
            x = self.embedding(x)
            printb("[ENCODER] (x) [embedding(x)]: {}".format(x.shape), printable=self.debug_mode)		

            input_shape=K.shape(x)	   

            if TRAINING == True:
            #{
                x = tf.nn.dropout(x, keep_prob= 1-self.encoder_embedding_TimestepDropout,  noise_shape=(input_shape[0], input_shape[1], 1)) # (TimestepDropout) then zero-out word embeddings
                x = tf.nn.dropout(x, keep_prob= 1-self.encoder_embedding_SpatialDropout1D, noise_shape=(input_shape[0], 1, input_shape[2])) # (SpatialDropout1D) and possibly drop some dimensions on every single embedding (timestep)
                printb("[ENCODER] (x) [after dropout_Embedding]: {}".format(x.shape), printable=self.debug_mode)
            #}
        #}

        a          = x
        residual_x = a
        stateH     = hidden
        stateC     = hidden
        
        del x
                
        a          = rnn_BI(a, training=TRAINING)		
        a, stateH  = rnn_UNI1(a , initial_state= hidden, training= TRAINING)
        stateC     = stateH	
        
        printb("[ENCODER] (output) :{}".format(output.shape), printable=self.debug_mode)
        printb("[ENCODER] (stateH) :{}".format(stateH.shape), printable=self.debug_mode)
        printb("[ENCODER] (stateC) :{}".format(stateC.shape), printable=self.debug_mode)
        
        printb("--------------------------------------(ENCODER END)--------------------------------------\n", printable=self.debug_mode)

        return output, stateH, stateC    
    #}
    
    def initialize_hidden_state(self):
    #{
        return tf.zeros((self.batch_size, self.enc_units))
    
        """
        https://www.kaggle.com/jaikishore7/tensorflow-eager-language-model?scriptVersionId=5784188
        
        if state == "train":
            x,_,_ = self.LSTM_1(train_values, initial_state = [self.hiddenH,self.hiddenC] )
            x,_,_ = self.LSTM_2(x, initial_state = [self.lstm2_ht,self.lstm2_ct] )
            x = self.out(x)
            return x

        else:
            x,lstm_1_ht,lstm_1_ct = self.LSTM_1(train_values,initial_state = [hiddenH,hiddenC])
            x,lstm_2_ht,lstm_2_ct = self.LSTM_2(x,initial_state = [lstm_2_ht,lstm_2_ct] )
            x = self.out(x)
            return x
        """
    #}
#}