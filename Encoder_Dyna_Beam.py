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

from .RNN_Dyna import *
from .lang_util import *

class Encoder(tf.keras.Model):
#{
    def __init__(self,
                vocab_size, 
                embedding_dim,
                batch_size, 
                enc_units,             
                num_encoder_bi_layers,
                num_encoder_uni_layers,  
                num_residual_bi_layers,               
                num_residual_uni_layers,
                learn_mode= None,
                rnn_cell_type= "gru",
                rnn_cell_mode= "NATIVE",  
                reverse_input_tensor=False,
                debug_mode=True,
                embedding_enable=True,
                activation=None, 
                reuse=None, 
                kernel_initializer=None, 
                bias_initializer=None, 
                name=None, 
                dtype=None,
                input_dropout= 0.0,  
                output_dropout= 0.0, 
                state_dropout= 0.0, 
                encoder_embedding_TimestepDropout= 0.0,
                encoder_embedding_SpatialDropout1D= 0.0,
                variational_recurrent= False,
                use_peepholes= False,
                cell_clip= None, 
                initializer= None, 
                num_proj= None,
                proj_clip= None, 
                num_unit_shards= None,
                num_proj_shards= None,
                forget_bias= 1.0, 
                state_is_tuple= True,
                sequence_length=None,
                parallel_iterations=None, 
                swap_memory=False, 
                time_major=False, 
                scope=None                                             
                ):
    #{    
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.enc_units  = enc_units
        self.num_encoder_bi_layers = num_encoder_bi_layers
        self.num_encoder_uni_layers = num_encoder_uni_layers
        self.reverse_input_tensor = reverse_input_tensor        
        self.rnn_cell_mode = rnn_cell_mode
        self.rnn_cell_type = rnn_cell_type
        self.debug_mode = debug_mode
        self.embedding_enable = embedding_enable

        if self.reverse_input_tensor:
            x = reverse_tensor(x)


        if embedding_enable:
        #{
            self.embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
            #self.embedding= tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.encoder_embedding_TimestepDropout  = encoder_embedding_TimestepDropout
            self.encoder_embedding_SpatialDropout1D = encoder_embedding_SpatialDropout1D
        #}

        
        if self.num_encoder_bi_layers > 0:
            #BI-DIRECTIONAL
            self.rnn_train_fw, self.rnn_eval_fw,  self.rnn_train_bw, self.rnn_eval_bw= RNN().bidirectional_rnn_nn_dyna_layers(  units= enc_units, 
                                                                                                                                rnn_cell_type= rnn_cell_type,
                                                                                                                                learn_mode= learn_mode, 
                                                                                                                                num_layers= num_encoder_bi_layers,
                                                                                                                                num_residual_layers= num_residual_bi_layers,
                                                                                                                                activation= activation, 
                                                                                                                                reuse= reuse, 
                                                                                                                                kernel_initializer= kernel_initializer, 
                                                                                                                                bias_initializer= bias_initializer, 
                                                                                                                                name= name, 
                                                                                                                                dtype= dtype,
                                                                                                                                input_dropout= input_dropout,  
                                                                                                                                output_dropout= output_dropout, 
                                                                                                                                state_dropout= state_dropout, 
                                                                                                                                variational_recurrent= variational_recurrent,
                                                                                                                                use_peepholes= use_peepholes,
                                                                                                                                cell_clip= cell_clip, 
                                                                                                                                initializer= initializer, 
                                                                                                                                num_proj= num_proj,
                                                                                                                                proj_clip= proj_clip, 
                                                                                                                                num_unit_shards= num_unit_shards,
                                                                                                                                num_proj_shards= num_proj_shards,
                                                                                                                                forget_bias= forget_bias, 
                                                                                                                                state_is_tuple= state_is_tuple)
        if self.num_encoder_uni_layers > 0:
            #UNI-DIRECTIONAL        
            self.rnn_train, self.rnn_eval= RNN().rnn_nn_dyna_layers(units= enc_units, 
                                                                    rnn_cell_type= rnn_cell_type,
                                                                    learn_mode= learn_mode, 
                                                                    num_layers= num_encoder_uni_layers,
                                                                    num_residual_layers= num_residual_uni_layers,
                                                                    activation= activation, 
                                                                    reuse= reuse, 
                                                                    kernel_initializer= kernel_initializer, 
                                                                    bias_initializer= bias_initializer, 
                                                                    name= name, 
                                                                    dtype= dtype,
                                                                    input_dropout= input_dropout,  
                                                                    output_dropout= output_dropout, 
                                                                    state_dropout= state_dropout, 
                                                                    variational_recurrent= variational_recurrent,
                                                                    use_peepholes= use_peepholes,
                                                                    cell_clip= cell_clip, 
                                                                    initializer= initializer, 
                                                                    num_proj= num_proj,
                                                                    proj_clip= proj_clip, 
                                                                    num_unit_shards= num_unit_shards,
                                                                    num_proj_shards= num_proj_shards,
                                                                    forget_bias= forget_bias, 
                                                                    state_is_tuple= state_is_tuple)           
    #}
    
    def call(self, x, hidden, TRAINING):  #encoder(inp, hidden)             called from NMT_Model.py
    #{
        printb("--------------------------------------(ENCODER STR)--------------------------------------\n", printable=self.debug_mode)
        
        printb("[ENCODER] (x):{}".format(x.shape) , printable=self.debug_mode)
        #printb("[ENCODER] (hidden) : {}".format(hidden.shape), printable=self.debug_mode)
            
        if self.reverse_input_tensor:
            #x = tensor_reversed = np.fliplr(x)
            x = tensor_reversed = tf.reverse(x, [-1])
        
        if self.embedding_enable:
        #{
            
            x = tf.nn.embedding_lookup(self.embedding, x)
            #x = self.embedding(x)

            printb("[ENCODER] (x) [tf.nn.embedding_lookup(x)]: {}".format(x.shape), printable=self.debug_mode)		

            input_shape=K.shape(x)	   

            if TRAINING == True:
            #{
                x = tf.nn.dropout(x, keep_prob= 1-self.encoder_embedding_TimestepDropout,  noise_shape=(input_shape[0], input_shape[1], 1)) # (TimestepDropout) then zero-out word embeddings
                x = tf.nn.dropout(x, keep_prob= 1-self.encoder_embedding_SpatialDropout1D, noise_shape=(input_shape[0], 1, input_shape[2])) # (SpatialDropout1D) and possibly drop some dimensions on every single embedding (timestep)
                printb("[ENCODER] (x) [after dropout_Embedding]: {}".format(x.shape), printable=self.debug_mode)
            #}
        #}

        a     = x
        state = None
        
        del x

        if TRAINING:       
        #{
            #-----------------------------------------------------------------------------------------------------------
            #BI-Directional Trianable

            if self.num_encoder_bi_layers > 0: 
                bi_a, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_fw= self.rnn_train_fw, 
                                                                cell_bw= self.rnn_train_bw,
                                                                inputs= a,
                                                                initial_state_fw=None, 
                                                                initial_state_bw=None, 
                                                                dtype=tf.float32, 
                                                                sequence_length=None,
                                                                parallel_iterations=None, 
                                                                time_major=False, 
                                                                scope=None, 
                                                                swap_memory=False)                
                a = tf.concat(bi_a, -1)
                
                #print(">>>>>>>>>>>>>> bi_a before concat:\n", bi_a)                
                #print(">>>>>>>>>>>>>> a after concat:\n", a)

            #-----------------------------------------------------------------------------------------------------------
            #UNI-Directional Trianable
            
            if self.num_encoder_uni_layers > 0:
                a, state = tf.nn.dynamic_rnn(cell= self.rnn_train, 
                                            inputs= a, 
                                            sequence_length=None, 
                                            initial_state=None, 
                                            dtype=tf.float32, 
                                            parallel_iterations=None, 
                                            swap_memory=False, 
                                            time_major=False, 
                                            scope=None)             
        #}

        else:
        #{
            #-----------------------------------------------------------------------------------------------------------
            #BI-Directional Evaluation 

            if self.num_encoder_bi_layers > 0:         
                bi_a, bi_state = tf.nn.bidirectional_dynamic_rnn(cell_fw= self.rnn_eval_fw, 
                                                                cell_bw= self.rnn_eval_bw,
                                                                inputs= a,
                                                                initial_state_fw=None, 
                                                                initial_state_bw=None, 
                                                                dtype=tf.float32, 
                                                                sequence_length=None,
                                                                parallel_iterations=None, 
                                                                time_major=False, 
                                                                scope=None, 
                                                                swap_memory=False)
                
                a = tf.concat(bi_a, -1)
            #-----------------------------------------------------------------------------------------------------------
            #UNI-Directional Evaluation

            if self.num_encoder_uni_layers > 0:          
                a, state = tf.nn.dynamic_rnn(cell= self.rnn_eval, 
                                            inputs= a, 
                                            sequence_length=None, 
                                            initial_state=None, 
                                            dtype=tf.float32, 
                                            parallel_iterations=None, 
                                            swap_memory=False, 
                                            time_major=False, 
                                            scope=None)
        #}

        output = a
        #printb("[ENCODER] (output) :{}".format(output.shape), printable=self.debug_mode)
        #printb("[ENCODER] (state)  :{}".format( hidden[0].shape if isinstance(hidden, tuple) else hidden.shape), printable=self.debug_mode)
        
        printb("--------------------------------------(ENCODER END)--------------------------------------\n", printable=self.debug_mode)

        return output, state   
    #}
    
    # Define the initial state
    def initialize_hidden_state(self):        
        init_hidden_state =  self.num_encoder_uni_layers * (tf.zeros((self.batch_size, self.enc_units)),)
        return init_hidden_state
#}