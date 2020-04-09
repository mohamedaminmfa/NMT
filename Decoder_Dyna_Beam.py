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


class Decoder(tf.keras.Model):
#{
    def __init__(self,
                vocab_size, 
                embedding_dim,
                batch_size, 
                dec_units,             
                num_decoder_bi_layers,
                num_decoder_uni_layers, 
                num_residual_bi_layers,          
                num_residual_uni_layers,
                learn_mode= None,
                rnn_cell_type= "lstm",
                rnn_cell_mode= "NATIVE", 
                beam_size=0, 
                targ_lang_train= None,
                attention_obj= None, 
                reverse_input_tensor=False,
                debug_mode=True,
                embedding_enable=True,
                output_fc_enable=True, 
                activation=None, 
                reuse=None, 
                kernel_initializer=None, 
                bias_initializer=None, 
                name=None, 
                dtype=None,
                input_dropout= 0.0,  
                output_dropout= 0.0, 
                state_dropout= 0.0, 
                decoder_embedding_TimestepDropout= 0.0,
                decoder_embedding_SpatialDropout1D= 0.0,
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
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units  = dec_units
        self.num_decoder_bi_layers  = num_decoder_bi_layers
        self.num_decoder_uni_layers = num_decoder_uni_layers	
        self.rnn_cell_type  = rnn_cell_type
        self.fc = tf.keras.layers.Dense(vocab_size)
       
        self.targ_lang_train  = targ_lang_train
        self.output_fc_enable =  output_fc_enable
        
        self.vocab_size = vocab_size
        self.beam_size = beam_size

        if embedding_enable:
        #{
            self.embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
            
            #self.embedding= tf.keras.layers.Embedding(vocab_size, embedding_dim)
            
            self.decoder_embedding_TimestepDropout  = decoder_embedding_TimestepDropout
            self.decoder_embedding_SpatialDropout1D = decoder_embedding_SpatialDropout1D
        #}

        self.rnn_cell_mode = rnn_cell_mode
        self.reverse_input_tensor = reverse_input_tensor
        self.attention_obj = attention_obj
        self.debug_mode = debug_mode
        self.embedding_enable = embedding_enable
        
        if self.num_decoder_bi_layers > 0:

            self.rnn_train_fw, self.rnn_eval_fw, self.rnn_train_bw, self.rnn_eval_bw= RNN().bidirectional_rnn_nn_dyna_layers(units= dec_units, 
                                                                                                                            rnn_cell_type= rnn_cell_type,
                                                                                                                            learn_mode= learn_mode, 
                                                                                                                            num_layers= num_decoder_bi_layers,
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
        if self.num_decoder_uni_layers > 0:

            self.rnn_train, self.rnn_eval= RNN().rnn_nn_dyna_layers(units= dec_units, 
                                                                    rnn_cell_type= rnn_cell_type,
                                                                    learn_mode= learn_mode, 
                                                                    num_layers= num_decoder_uni_layers,
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

    def length(self, sequence):
    #{
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length
    #}

    def call(self, y, hidden, enc_output, TRAINING):  #decoder(dec_input, dec_hidden, enc_output)
    #{ 
       

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units= self.dec_units,
                                                                   memory= enc_output,
                                                                   memory_sequence_length= enc_output.shape[1])


        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell= self.rnn_train, 
                                                           attention_mechanism= attention_mechanism,
                                                           attention_layer_size= self.dec_units, 
                                                           name= "Attention_Wrapper")
        
        """
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units= self.dec_units, 
                                                                memory= encoder_outputs,
                                                                memory_sequence_length= encoder_inputs_length)
        """

        decoder_initial_state = decoder_cell.zero_state(batch_size= self.batch_size, dtype=tf.float32).clone(cell_state= hidden)
        output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # mode == 'training'
        if TRAINING == True:  
        #{
            ending = tf.strided_slice(y, [0, 0], [self.batch_size, -1], [1, 1])
            
            decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.targ_lang_train.word2idx['<start>']), ending], 1)
            
            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_input)

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs= decoder_inputs_embedded,
                                                                sequence_length= self.length(decoder_inputs_embedded),
                                                                time_major= False, 
                                                                name= 'training_helper')
            
            
            training_decoder = tf.contrib.seq2seq.BasicDecoder( cell= decoder_cell, 
                                                                helper= training_helper,
                                                                initial_state= decoder_initial_state, 
                                                                output_layer= output_layer)

            print("self.targ_lang_train.maxLength:", self.targ_lang_train.maxLength)
            print("self.targ_lang_train.maxLength:", tf.reduce_max(self.targ_lang_train.maxLength))

            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(  decoder= training_decoder,
                                                                        impute_finished= True,
                                                                        maximum_iterations= tf.reduce_max(self.targ_lang_train.maxLength, name='max_target_len'))


            self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
            
            self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis= -1, name= 'decoder_pred_train')
            
            """
            self.loss = tf.contrib.seq2seq.sequence_loss(logits= self.decoder_logits_train,
                                                        targets= y,
                                                        weights= self.mask)
            
            # Training summary for the current batch_loss
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer(self.learing_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
            """
            return decoder_logits_train, decoder_predict_train
        #}

        # mode == 'infer'
        elif TRAINING == False:  
        #{
            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.targ_lang_train.word2idx['<start>']
            end_token    = self.targ_lang_train.word2idx['<end>']

            if self.beam_search:
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell= decoder_cell, 
                                                                        embedding= self.embedding,
                                                                        start_tokens= start_tokens, 
                                                                        end_token= end_token,
                                                                        initial_state= decoder_initial_state,
                                                                        beam_width= self.beam_size,
                                                                        output_layer= output_layer)
            else:
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding= self.embedding,
                                                                        start_tokens= start_tokens, 
                                                                        end_token= end_token)
                
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell= decoder_cell,
                                                                    helper= decoding_helper,
                                                                    initial_state= decoder_initial_state,
                                                                    output_layer= output_layer)
                
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder= inference_decoder,
                                                                    maximum_iterations= 10)

            if self.beam_search:
                self.decoder_predict_decode = decoder_outputs.predicted_ids
            else:
                self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)

        #}        
    #}
#}