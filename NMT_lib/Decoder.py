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
from NMT_lib.BahdanauAttention import *
from NMT_lib.LuongsAttention import *




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
					embedding_enable=True
				 )

"""

class Decoder(tf.keras.Model):
    
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 dec_units, 
                 batch_size,
                 decoder_layer, 
                 decoder_dropout=[0.0], 
                 decoder_recurrent_dropout=[0.0],
				 decoder_embedding_TimestepDropout=0.0,
				 decoder_embedding_SpatialDropout1D=0.0,
				 rnn_cell_mode="NATIVE",
				 attention_obj= None,
                 residual_connections_enabled=False, 
                 residual_start_from=-1,
				 debug_mode=False,
                 embedding_enable=True):    

    #{    
        super(Decoder, self).__init__()
        self.batch_size  = batch_size
        self.dec_units = dec_units		
        
        self.fc        = tf.keras.layers.Dense(vocab_size)
        
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
		
        self.model_decoder_list   = []
        
        printb("(vocab_size)   : {}".format(vocab_size), printable=self.debug_mode)
        printb("(batch_size)   : {}".format(batch_size), printable=self.debug_mode)
        printb("(embedding_dim): {}".format(embedding_dim), printable=self.debug_mode)       
        printb("(dec_units)    : {}".format(self.dec_units), printable=self.debug_mode)  
        
        
        #BI-DIRECTIONAL
        numLayer_counter = 0
        
        for index, numLayer in enumerate(decoder_layer[0]):
        #{
            if ( (index+1) % 2 == 0 ): #BI-LSTM 
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(decoder_dropout) ==1:
                        Ni = 0
                    
                    if len(decoder_recurrent_dropout) ==1:
                        Nr = 0
                    
                    BI_lstm = RNN().Bidirectional_lstm(self.dec_units, mode=rnn_cell_mode, input_dropout= decoder_dropout[Ni], rec_dropout= decoder_recurrent_dropout[Nr])
                    self.model_decoder_list.append( BI_lstm )
                    print("[Decoder] BI_LSTM")
                #}                    
            #}
            
            else: #BI-GRU
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(decoder_dropout) == 1:
                        Ni = 0
                    
                    if len(decoder_recurrent_dropout) == 1:
                        Nr = 0
                        
                    BI_gru = RNN().Bidirectional_gru(self.dec_units, mode=rnn_cell_mode, input_dropout= decoder_dropout[Ni], rec_dropout= decoder_recurrent_dropout[Nr])
                    self.model_decoder_list.append( BI_gru )
                    print("[Decoder] BI_GRU")
                #}    
            #}
            
            numLayer_counter += numLayer   
        #}
        
        
        #UNI-DIRECTIONAL
        
        for index, numLayer in enumerate(decoder_layer[1]):
        #{
            if ( (index+1) % 2 == 0 ): #UNI-LSTM              
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(decoder_dropout) == 1:
                        Ni = 0
                    
                    if len(decoder_recurrent_dropout) == 1:
                        Nr = 0
                    
                    lstm = RNN().lstm(self.dec_units, mode=rnn_cell_mode, input_dropout= decoder_dropout[Ni], rec_dropout= decoder_recurrent_dropout[Nr])
                    self.model_decoder_list.append( lstm )
                    print("[Decoder] UNI_LSTM")
                #}    
            #} 
            
            else: #UNI-GRU
            #{    
                for n in range(numLayer):
                #{   
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(decoder_dropout) == 1:
                        Ni = 0
                    
                    if len(decoder_recurrent_dropout) == 1:
                        Nr = 0
                    
                    gru = RNN().gru(self.dec_units, mode=rnn_cell_mode, input_dropout= decoder_dropout[Ni], rec_dropout= decoder_recurrent_dropout[Nr])
                    self.model_decoder_list.append( gru ) 
                    print("[Decoder] UNI_GRU")
                #}
            #}
            
            numLayer_counter += numLayer
        #}
    #}    
    
    def call(self, x, hidden, enc_output, TRAINING):  #decoder(dec_input, dec_hidden, enc_output)
    #{    
        printb("--------------------------------------(DECODER STR)--------------------------------------\n", printable=self.debug_mode)

        # enc_output shape == (batch_size, max_length, hidden_size)

        attention_weights = None
          
        if self.embedding_enable:            
            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)
            printb("(x) [embedding(x)]: {}".format(x.shape), printable=self.debug_mode)     

        input_shape=K.shape(x)	   
        noise_shape =(input_shape[0], 1, input_shape[2])
        x = tf.layers.dropout(x, self.decoder_embedding_TimestepDropout, noise_shape=noise_shape, training=TRAINING) # (TimestepDropout) then zero-out word embeddings
        x = tf.layers.dropout(x, self.decoder_embedding_SpatialDropout1D, noise_shape=noise_shape, training=TRAINING) # (SpatialDropout1D) and possibly drop some dimensions on every single embedding (timestep)
        printb("(x) [after dropout_Embedding]: {}".format(x.shape), printable=self.debug_mode)

        if( self.attention_obj != None ):            
            context_vector, attention_weights = self.attention_obj(hidden[0], enc_output)
                
            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            context_dec_input = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            printb("(context_dec_input) [tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)]: {}".format(context_dec_input.shape), printable=self.debug_mode)
            del x
			
        else:
            context_dec_input = x
            printb("(context_dec_input) [context_dec_input = x]: {}".format(context_dec_input.shape), printable=self.debug_mode)
            del x

        printb("(hidden) : {}".format( hidden[0].shape), printable=self.debug_mode)
        
        # passing the concatenated vector to the RNN cell         
        a      = context_dec_input
        del context_dec_input

        #print("model_decoder_list = %s" % len(self.model_decoder_list))
        
        dropout_step = 0
        
        for index, LayerObject in enumerate(self.model_decoder_list): 
        #{                
            if( self.residual_connections_enabled and index >= self.residual_start_from):
                residual_x = a
                
            if( isinstance( LayerObject, tf.keras.layers.Bidirectional ) ):
                #Decoder BI
                a   = LayerObject(a, training=TRAINING)
                printb("(a) Decoder BI: {}".format(a.shape), printable=self.debug_mode)
                
            elif( isinstance( LayerObject, tf.keras.layers.CuDNNGRU ) or isinstance( LayerObject, tf.keras.layers.GRU )):   
                #Decoder GRU
                if( self.attention_obj != None ):
                    a, stateH  = LayerObject(a, training=TRAINING)
                    stateC     = stateH
                    printb("(a)      Decoder GRU with Attention: {}".format(a.shape), printable=self.debug_mode)
                    printb("(stateH) Decoder GRU with Attention: {}".format(stateH.shape), printable=self.debug_mode)
                    printb("(stateC) Decoder GRU with Attention: {}".format(stateC.shape), printable=self.debug_mode)
					
                else:
                    a, stateH  = LayerObject(a, initial_state=hidden[0], training=TRAINING)
                    stateC     = stateH
                    printb("(a)      Decoder GRU without Attention: {}".format(a.shape), printable=self.debug_mode)
                    printb("(stateH) Decoder GRU without Attention: {}".format(stateH.shape), printable=self.debug_mode)
                    printb("(stateC) Decoder GRU without Attention: {}".format(stateC.shape), printable=self.debug_mode)
                    
            elif( isinstance( LayerObject, tf.keras.layers.CuDNNLSTM ) or isinstance( LayerObject, tf.keras.layers.LSTM )):
                #Decoder LSTM
                if( self.attention_obj != None  ):
                    a, stateH , stateC = LayerObject(a, training=TRAINING)
                    printb("(a)      Decoder LSTM with Attention: {}".format(a.shape), printable=self.debug_mode)
                    printb("(stateH) Decoder LSTM with Attention: {}".format(stateH.shape), printable=self.debug_mode)
                    printb("(stateC) Decoder LSTM with Attention: {}".format(stateC.shape), printable=self.debug_mode)
                else:
                    a, stateH, stateC  = LayerObject(a, initial_state=hidden, training=TRAINING)
                    printb("(a)      Decoder LSTM without Attention: {}".format(a.shape), printable=self.debug_mode)
                    printb("(stateH) Decoder LSTM without Attention: {}".format(stateH.shape), printable=self.debug_mode)
                    printb("(stateC) Decoder LSTM without Attention: {}".format(stateC.shape), printable=self.debug_mode)
                
            if(self.residual_connections_enabled and index >= self.residual_start_from):
                a = tf.math.add(a, residual_x) 
                printb("(a) Decoder add(a, residual_x): {}".format(a.shape), printable=self.debug_mode)				
                del residual_x
         
            #Apply Dropout
            if(self.rnn_cell_mode.upper == "CuDNN".upper()):
            #{
                if(index % len(decoder_dropout) == 0):
                    dropout_step =0
                else:
                    dropout_step +=1

                a = tf.layers.dropout(a, decoder_dropout[dropout_step], training=TRAINING)
                printb("(a) Decoder Dropout Layer: {}".format(a.shape), printable=self.debug_mode)
            #}           
        #}
        
        output = a
        del a
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        printb("(output) [tf.reshape(output, (-1, output.shape[2]))]: {}".format(output.shape), printable=self.debug_mode)
        
        # output shape == (batch_size * 1, vocab)
        out_fc = self.fc(output)
        printb("(out_fc) [DECODER][fc(output)]: {}".format(out_fc.shape), printable=self.debug_mode)
        printb("(stateH) [DECODER]:{}".format(stateH.shape), printable=self.debug_mode)
        printb("(stateC) [DECODER]:{}".format(stateC.shape), printable=self.debug_mode)
        
        del output
		
        printb("--------------------------------------(DECODER END)--------------------------------------\n", printable=self.debug_mode)

        return out_fc, stateH, stateC, attention_weights
    #}
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))