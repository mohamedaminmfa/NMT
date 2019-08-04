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

from NMT_lib.text_processing_util import *
from NMT_lib.lang_util import *
from NMT_lib.graph_plotting_util import *
from NMT_lib.RNN import *






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
                 encoder_layer, 
                 encoder_dropout=[0.0], 
                 encoder_recurrent_dropout=[0.0],
				 encoder_embedding_TimestepDropout=0.0,
				 encoder_embedding_SpatialDropout1D=0.0,
                 rnn_cell_mode="NATIVE",
                 reverse_input_tensor=False,
                 residual_connections_enabled=False, 
                 residual_start_from=-1,
				 debug_mode=False,
                 embedding_enable=True):
    #{    
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.enc_units  = enc_units
        
        if embedding_enable:
            self.embedding  = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        self.encoder_embedding_TimestepDropout  = encoder_embedding_TimestepDropout
        self.encoder_embedding_SpatialDropout1D = encoder_embedding_SpatialDropout1D

        self.model_encoder_list   = []

        self.rnn_cell_mode = rnn_cell_mode
        self.reverse_input_tensor = reverse_input_tensor
        self.residual_connections_enabled = residual_connections_enabled
        self.residual_start_from = residual_start_from
        self.debug_mode = debug_mode
        self.embedding_enable = embedding_enable
        
        
        #BI-DIRECTIONAL
        numLayer_counter = 0
        
        for index, numLayer in enumerate(encoder_layer[0]):
        #{    
            if ( (index+1) % 2 == 0 ): #BI-LSTM 
            #{    
                for n in range(numLayer):                    
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(encoder_dropout) ==1:
                        Ni = 0
                    
                    if len(encoder_recurrent_dropout) ==1:
                        Nr = 0

                    BI_lstm = RNN().Bidirectional_lstm(self.enc_units, mode=rnn_cell_mode, input_dropout=encoder_dropout[Ni], rec_dropout=encoder_recurrent_dropout[Nr])                    
                    self.model_encoder_list.append( BI_lstm )
                    print("[Encoder] BI_LSTM")
                #}        
            #}
            
            else: #BI-GRU
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(encoder_dropout) ==1:
                        Ni = 0
                    
                    if len(encoder_recurrent_dropout) ==1:
                        Nr = 0
                    
                    BI_gru = RNN().Bidirectional_gru(self.enc_units, mode=rnn_cell_mode, input_dropout=encoder_dropout[Ni], rec_dropout=encoder_recurrent_dropout[Nr])
                    self.model_encoder_list.append( BI_gru )
                    print("[Encoder] BI_GRU")
                #}  
            #}
            
            numLayer_counter += numLayer
        #}           

        
        #UNI-DIRECTIONAL
        
        for index, numLayer in enumerate(encoder_layer[1]):
        #{
            if ( (index+1) % 2 == 0 ): #UNI-LSTM              
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(encoder_dropout) ==1:
                        Ni = 0
                    
                    if len(encoder_recurrent_dropout) ==1:
                        Nr = 0                    
                    
                    lstm = RNN().lstm(self.enc_units, mode=rnn_cell_mode, input_dropout=encoder_dropout[Ni], rec_dropout=encoder_recurrent_dropout[Nr])
                    self.model_encoder_list.append( lstm )
                    print("[Encoder] UNI_LSTM")
                #}
            #}
            
            else: #UNI-GRU
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(encoder_dropout) ==1:
                        Ni = 0
                    
                    if len(encoder_recurrent_dropout) ==1:
                        Nr = 0                    
                    
                    gru = RNN().gru(self.enc_units, mode=rnn_cell_mode, input_dropout=encoder_dropout[Ni], rec_dropout=encoder_recurrent_dropout[Nr])
                    self.model_encoder_list.append( gru )
                    print("[Encoder] UNI_GRU")
                #}
            #}
            
            numLayer_counter += numLayer           
        #}                    
    #}
    
    def call(self, x, hidden, TRAINING):  #encoder(inp, hidden)
    #{
        printb("--------------------------------------(ENCODER STR)--------------------------------------\n", printable=self.debug_mode)
        
        printb("(x):{}".format(x.shape) , printable=self.debug_mode)
        printb("(hidden) : {}".format(hidden.shape), printable=self.debug_mode)
        
        if self.reverse_input_tensor:
            x = reverse_tensor(x)
        
        if self.embedding_enable:
            x = self.embedding(x)
            printb("(x) [embedding(x)]: {}".format(x.shape), printable=self.debug_mode)		

        input_shape=K.shape(x)	   
        noise_shape =(input_shape[0], 1, input_shape[2])
        x = tf.layers.dropout(x, self.encoder_embedding_TimestepDropout, noise_shape=noise_shape, training=TRAINING) # (TimestepDropout) then zero-out word embeddings
        x = tf.layers.dropout(x, self.encoder_embedding_SpatialDropout1D, noise_shape=noise_shape, training=TRAINING)# (SpatialDropout1D) and possibly drop some dimensions on every single embedding (timestep)
        printb("(x) [after dropout_Embedding]: {}".format(x.shape), printable=self.debug_mode)

        a          = x
        residual_x = a
        stateH     = hidden
        stateC     = hidden
        
        del x
				
        dropout_step = 0        
        for index, LayerObject in enumerate(self.model_encoder_list): 
        #{    
            if(self.residual_connections_enabled and index >= self.residual_start_from):  
                residual_x = a
                
            if( isinstance( LayerObject, tf.keras.layers.Bidirectional ) ):
                #Encoder BI
                a = LayerObject(a, training=TRAINING)
                printb("(a)      Encoder BI-DIRECTIONAL: {}".format(a.shape), printable=self.debug_mode)
                
            elif( isinstance( LayerObject, tf.keras.layers.CuDNNGRU ) or isinstance( LayerObject, tf.keras.layers.GRU )):   
                #Encoder GRU
                a, stateH  = LayerObject(a , initial_state = hidden, training=TRAINING)
                stateC     = stateH
                printb("(a)      Encoder GRU: {}".format(a.shape), printable=self.debug_mode)
                printb("(stateH) Encoder GRU: {}".format(stateH.shape), printable=self.debug_mode)
                printb("(stateC) Encoder GRU: {}".format(stateC.shape), printable=self.debug_mode)
				
            elif( isinstance( LayerObject, tf.keras.layers.CuDNNLSTM ) or isinstance( LayerObject, tf.keras.layers.LSTM )):
                #Encoder LSTM
                a, stateH , stateC = LayerObject(a , initial_state = [hidden, hidden] , training=TRAINING)
                printb("(a)      Encoder LSTM: {}".format(a.shape), printable=self.debug_mode)
                printb("(stateH) Encoder LSTM: {}".format(stateH.shape), printable=self.debug_mode)
                printb("(stateC) Encoder LSTM: {}".format(stateC.shape), printable=self.debug_mode)

            
            if(self.residual_connections_enabled and index >= self.residual_start_from):
                a = tf.math.add(a, residual_x) 
                printb("(a)      Encoder add(a, residual_x): {}".format(a.shape), printable=self.debug_mode)					
                del residual_x
            
            #Apply Dropout
            if(self.rnn_cell_mode.upper == "CuDNN".upper()):
            #{
                if(index % len(encoder_dropout) == 0):
                    dropout_step =0
                else:
                    dropout_step +=1

				
                a = tf.layers.dropout(a, encoder_dropout[dropout_step], training=TRAINING)
                printb("(a)     Encoder Dropout Layer: {}".format(a.shape), printable=self.debug_mode)

            #}          
        #}
        
        output = a
        del a

        printb("(output) [ENCODER]:{}".format(output.shape), printable=self.debug_mode)
        printb("(stateH) [ENCODER]:{}".format(stateH.shape), printable=self.debug_mode)
        printb("(stateC) [ENCODER]:{}".format(stateC.shape), printable=self.debug_mode)
        
        printb("--------------------------------------(ENCODER END)--------------------------------------\n", printable=self.debug_mode)

        return output, stateH, stateC    
    #}
    
    def initialize_hidden_state(self):
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