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


#https://www.cnblogs.com/lovychen/p/10470564.html

class BahdanauAttention(tf.keras.Model):
  
  def __init__(self, units, debug_mode=False):
    super(BahdanauAttention, self).__init__()
    
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V  = tf.keras.layers.Dense(1)

    self.debug_mode = debug_mode
  
 #def call(self, query, values):
  def call(self, H, EO):  # H: Hidden State   , EO: Encoder Output
    
    printb(".....................................(ATTENTION STR).....................................\n", printable=self.debug_mode)
    
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    
    H = tf.expand_dims(H, 1)
    printb("(H) [hidden after expand]: {}".format(H.shape), printable=self.debug_mode)

    
    # score shape == (batch_size, max_length, hidden_size)
    # applying tanh(FC(EO) + FC(H)) to self.V
    score = self.V(tf.nn.tanh(self.W1(EO) + self.W2(H)))
    printb("(score) [V(tanh( W1*EO + W2*H ))]: {}".format(score.shape), printable=self.debug_mode)
    
    
    # attention_weights shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    attention_weights = tf.nn.softmax(score, axis=1)
    printb("(attention_weights) [softmax(self.V(score), axis=1)]: {}".format(attention_weights.shape), printable=self.debug_mode)


    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * EO
    printb("(context_vector) [attention_weights * enc_output]: {}".format(context_vector.shape), printable=self.debug_mode)
    
    
    context_vector = tf.reduce_sum(context_vector, axis=1)
    printb("(context_vector) [reduce_sum(context_vector)]: {}".format(context_vector.shape), printable=self.debug_mode)
    
    
    printb(".....................................(ATTENTION END).....................................\n", printable=self.debug_mode)

    
    return context_vector, attention_weights
