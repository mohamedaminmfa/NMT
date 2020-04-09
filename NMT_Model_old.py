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
    #}		


    def call(self, inputs, hidden, targets, loss, training_type, teacher_forcing_ratio, training):    
    #{
        printb("-------------------------------------(NMT_MODEL STR)-------------------------------------\n", printable=self.debug_mode)
        #Encoder inputs.shape       =(BS, Tx)                   ex: inputs       : (16, 55)
        #Encoder hidden.shape       =(BS, units)                ex: hidden       : (16, 1024)
        #Encoder enc_output.shape   =(BS, Tx, units)            ex: enc_output   : (16, 55, 1024)
        #Encoder enc_hidden_H.shape =(BS, units)                ex: enc_hidden_H : (16, 1024)
        #Encoder enc_hidden_C.shape =(BS, units)                ex: enc_hidden_C : (16, 1024)
        #dec_input.shape            =(BS, 1)                    ex: enc_hidden_C : (16, 1)
        #start.shape                =(BS, Target_Vocab_Size)    ex: enc_hidden_C : (16, 1237)

        batch_size = inputs.shape[0]
        outs        = [] 
        enc_output, enc_hidden_H, enc_hidden_C = self.encoder(inputs, hidden, training)        
        dec_hidden  = [enc_hidden_H, enc_hidden_C] 

        printb("[NMT_MODEL] (inputs)       -> Encoder : {}".format(inputs.shape), printable=self.debug_mode)
        printb("[NMT_MODEL] (hidden)       -> Encoder : {}".format(hidden.shape), printable=self.debug_mode)
        printb("[NMT_MODEL] (training)     -> Encoder : {}".format(training), printable=self.debug_mode)
        printb("[NMT_MODEL] (enc_output)   <- Encoder : {}".format(enc_output.shape), printable=self.debug_mode)
        printb("[NMT_MODEL] (enc_hidden_H) <- Encoder : {}".format(enc_hidden_H.shape), printable=self.debug_mode)
        printb("[NMT_MODEL] (enc_hidden_C) <- Encoder : {}".format(enc_hidden_C.shape), printable=self.debug_mode)
        
        del inputs
        del hidden
        del enc_hidden_H
        del enc_hidden_C		
		
        targ_lang_train = self.decoder.targ_lang_train

        if targ_lang_train != None:
            dec_input = tf.expand_dims([targ_lang_train.word2idx['<start>']] * batch_size, 1)            
            printb("[NMT_MODEL] (dec_input) [tf.expand_dims([targ_lang_train.word2idx['<start>']] * batch_size, 1)] : {}".format(dec_input.shape), printable=self.debug_mode)

            start = tf.one_hot( np.full((batch_size),targ_lang_train.word2idx['<start>']), len(targ_lang_train.word2idx))
            printb("[NMT_MODEL] (start)[tf.one_hot( np.full((batch_size),targ_lang_train.word2idx['<start>']), len(targ_lang_train.word2idx))] : {}".format(start.shape), printable=self.debug_mode)

            outs.append( start )
        else:
            dec_input = tf.expand_dims([0] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targets.shape[1]):
        #{

            # passing enc_output to the decoder

            # dec_input.shape     =(BS, 1)                   ex: dec_input       : (16, 1)                        
            # dec_hidden[0].shape =(BS, units)               ex: dec_hidden      : (16, 1024)
            # dec_hidden[1].shape =(BS, units)               ex: dec_hidden      : (16, 1024)
            # len(dec_hidden)     =List of size 2            ex: len(dec_hidden) :  2 
            # enc_output.shape    =(BS, Tx, units)           ex: enc_output      : (16, 55, 1024)
            # predictions.shape   =(BS, Target_Vocab_Size)   ex: predictions     : (16, 1237)
            # dec_hidden_H.shape  =(BS, units)               ex: dec_hidden_H    : (16, 1024)
            # dec_hidden_C.shape  =(BS, units)               ex: dec_hidden_C    : (16, 1024)

            predictions, dec_hidden_H, dec_hidden_C, _ = self.decoder(dec_input, dec_hidden, enc_output, training, t)
            dec_hidden  = [dec_hidden_H, dec_hidden_C]

            printb("[NMT_MODEL] (dec_input     -> Decoder : {}".format(dec_input.shape), printable=self.debug_mode)
            printb("[NMT_MODEL] (enc_output)   -> Decoder : {}".format(enc_output.shape), printable=self.debug_mode)
            printb("[NMT_MODEL] (training)     -> Decoder : {}".format(training), printable=self.debug_mode)
            printb("[NMT_MODEL] (predictions)  <- Decoder : {}".format(predictions.shape), printable=self.debug_mode)
            printb("[NMT_MODEL] (dec_hidden_H) <- Decoder : {}".format(dec_hidden_H.shape), printable=self.debug_mode)
            printb("[NMT_MODEL] (dec_hidden_C) <- Decoder : {}".format(dec_hidden_C.shape), printable=self.debug_mode)

        
            del dec_hidden_H
            del dec_hidden_C			

            loss += self.loss_function(targets[:, t], predictions, self.optimizer_inst)

            outs.append(predictions)

            if training==True:
            #{
                #https://www.nextremer.com/blog/5556/

                if training_type.upper() == "FREE_RUNNING".upper():
                #{                    
                    # using free running
                    predicted_id = tf.argmax(predictions, axis=-1)          
                    dec_input    = tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))
                    printb("[NMT_MODEL] (dec_input) = [dec_input = tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))]".format(dec_input.shape), printable=self.debug_mode)					
                #}

                elif training_type.upper() == "TEACHER_FORCING".upper():
                #{    
                    # using teacher forcing
                    dec_input = tf.expand_dims(targets[:, t], 1)		
                    printb("[NMT_MODEL] (dec_input) = [tf.expand_dims(targets[:, t], 1)]".format(dec_input.shape), printable=self.debug_mode)					
                #}

                elif training_type.upper() == "SEMI_TEACHER_FORCING".upper():
                #{    
                    # using teacher forcing
                    Y_t_1 = tf.expand_dims(targets[:, t], 1)
                    printb("[NMT_MODEL] (Y_t_1) = [tf.expand_dims(targets[:, t], 1)]".format(dec_input.shape), printable=self.debug_mode)		

                    # using free running
                    predicted_id = tf.argmax(predictions, axis=-1)  
                    printb("[NMT_MODEL] (predicted_id) = [tf.argmax(predictions, axis=-1)]".format(dec_input.shape), printable=self.debug_mode)
                    
                    Yh_t_1       = tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))
                    printb("[NMT_MODEL] (Yh_t_1) = [tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))]".format(dec_input.shape), printable=self.debug_mode)		

                    # using semi-teacher forcing Dt = a*Y_t_1 + (1-a)Yh_t_1
                    a = teacher_forcing_ratio					
                    Y_t_1  = a * Y_t_1
                    Yh_t_1 = (1-a) * Yh_t_1
                    
                    printb("[NMT_MODEL] (Y_t_1)  = [a * Y_t_1]".format(dec_input.shape), printable=self.debug_mode)	
                    printb("[NMT_MODEL] (Yh_t_1) = [(1-a) * Yh_t_1]".format(dec_input.shape), printable=self.debug_mode)	

                    dec_input = tf.concat(Y_t_1, Yh_t_1)
                    printb("[NMT_MODEL] (dec_input) = [tf.concat(Y_t_1, Yh_t_1)]".format(dec_input.shape), printable=self.debug_mode)
                #}

                elif training_type.upper() == "SCHEDULED_SAMPLING".upper():
                #{                    
                    use_teacher_forcing = random.random() < teacher_forcing_ratio

                    if use_teacher_forcing:
                    #{
                        # using teacher forcing
                        dec_input = tf.expand_dims(targets[:, t], 1)
                        printb("[NMT_MODEL] (dec_input) = [tf.expand_dims(targets[:, t], 1)]".format(dec_input.shape), printable=self.debug_mode)
                    #}
                    
                    else:
                    #{
                        # using free running
                        predicted_id = tf.argmax(predictions, axis=-1)     
                        printb("[NMT_MODEL] (predicted_id) [tf.argmax(predictions, axis=-1) ]".format(predicted_id.shape), printable=self.debug_mode)
                        
                        dec_input    = tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))	
                        printb("[NMT_MODEL] (dec_input) [tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))]".format(dec_input.shape), printable=self.debug_mode)
                    #}
                #}

                else:
                #{
                    raise ValueError("Unknown training type %s!" % training_type)
                #}
            #}

            else:
            #{   			
                # using free running
                predicted_id = tf.argmax(predictions, axis=-1)  
                printb("[NMT_MODEL] (predicted_id) [tf.argmax(predictions, axis=-1) ]".format(dec_input.shape), printable=self.debug_mode)
                
                dec_input    = tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))
                printb("[NMT_MODEL] (dec_input) [tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))]".format(dec_input.shape), printable=self.debug_mode)
            #}
        #}       
        
        printb("-------------------------------------(NMT_MODEL END)-------------------------------------\n", printable=self.debug_mode)

        return outs, loss		
    #}
#}