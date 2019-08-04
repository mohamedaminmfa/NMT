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


class RNN():
#{    
    """
    def __init__(self, units):
        self.units  = units
    """
#...........................................................................................................................................................
    
    def gru(self, units, mode, input_dropout= None, rec_dropout= None):
        
        # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
        # the code automatically does that.
        if( tf.test.is_gpu_available() and  mode.upper() == "CuDNN".upper() ):
            print(">>>>> CuDNNGRU")
            return tf.keras.layers.CuDNNGRU(units, 
                                            return_sequences=True, 
                                            return_state=True, 
                                            recurrent_initializer='glorot_uniform')        
        else:  
            return tf.keras.layers.GRU(units, 
                                       return_sequences=True, 
                                       return_state=True, 
                                       recurrent_activation='sigmoid', 
                                       recurrent_initializer='glorot_uniform', dropout=input_dropout, recurrent_dropout=rec_dropout)

#...........................................................................................................................................................
    
    def lstm_simple(self, units, mode, input_dropout= None, rec_dropout= None):
                
        # If you have a GPU, we recommend using CuDNNLSTM(provides a 3x speedup than LSTM)
        # the code automatically does that.
        if( tf.test.is_gpu_available() and  mode.upper() == "CuDNN".upper() ):
            return tf.keras.layers.CuDNNLSTM(units, 
                                             return_sequences=True, 
                                             return_state=True, 
                                             recurrent_initializer='glorot_uniform')
        
        else:
            return tf.keras.layers.lstm(units, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_activation='sigmoid', 
                                        recurrent_initializer='glorot_uniform', dropout=input_dropout, recurrent_dropout=rec_dropout)
    
#...........................................................................................................................................................
    
    def lstm(self, units, mode, input_dropout= None, rec_dropout= None):
        
                        
        # If you have a GPU, we recommend using CuDNNLSTM(provides a 3x speedup than LSTM)
        # the code automatically does that.
        if( tf.test.is_gpu_available() and  mode.upper() == "CuDNN".upper() ):
            return tf.keras.layers.CuDNNLSTM(units,
                                            return_sequences = True,
                                            return_state = True,
                                            recurrent_initializer= tf.keras.initializers.truncated_normal(stddev=0.1),
                                            recurrent_regularizer = tf.keras.regularizers.l2(0.01),
                                            kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.1),
                                            bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                            bias_regularizer = tf.keras.regularizers.l2(0.01))
        
        else:
            return tf.keras.layers.LSTM(units,
                                        return_sequences = True,
                                        return_state = True,
                                        recurrent_initializer= tf.keras.initializers.truncated_normal(stddev=0.1),
                                        recurrent_regularizer = tf.keras.regularizers.l2(0.01),
                                        kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.1),
                                        bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                        bias_regularizer = tf.keras.regularizers.l2(0.01), dropout=input_dropout, recurrent_dropout=rec_dropout)
        
#...........................................................................................................................................................    
    
    def Bidirectional_lstm(self, units, mode, input_dropout= None, rec_dropout= None):

                        
        # If you have a GPU, we recommend using CuDNNLSTM(provides a 3x speedup than LSTM)
        # the code automatically does that.
        if( tf.test.is_gpu_available() and  mode.upper() == "CuDNN".upper() ):
            return  tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units,
                                                                        return_sequences = True,
                                                                        recurrent_initializer = tf.keras.initializers.truncated_normal(stddev=0.1),
                                                                        recurrent_regularizer = tf.keras.regularizers.l2(0.01),
                                                                        kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.1),
                                                                        bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                                                        bias_regularizer = tf.keras.regularizers.l2(0.01))) 
        
        else:
            return  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units,
                                                                       return_sequences = True,
                                                                       recurrent_initializer = tf.keras.initializers.truncated_normal(stddev=0.1),
                                                                       recurrent_regularizer = tf.keras.regularizers.l2(0.01),
                                                                       kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.1),
                                                                       bias_initializer='zeros',kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                                                       bias_regularizer = tf.keras.regularizers.l2(0.01), dropout=input_dropout, recurrent_dropout=rec_dropout))

#...........................................................................................................................................................

    def Bidirectional_gru(self, units, mode, input_dropout= None, rec_dropout= None):

                        
        # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
        # the code automatically does that.
        if( tf.test.is_gpu_available() and  mode.upper() == "CuDNN".upper() ):
            return  tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(units, 
                                                                           return_sequences=True, 
                                                                           recurrent_initializer='glorot_uniform'))
        
        else:
            return tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units, 
                                                                     return_sequences=True, 
                                                                     recurrent_activation='sigmoid', 
                                                                     recurrent_initializer='glorot_uniform', dropout=input_dropout, recurrent_dropout=rec_dropout))
#}
