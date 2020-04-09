import keras as k
import tensorflow as tf

from SL.lib.NMT_lib.lang_util import *

#https://www.cnblogs.com/lovychen/p/10470564.html
#https://github.com/MayerMax/deeplavrov/blob/c1c3e87143da2f4f21d34c34bc5f6759c2db0e87/deeplavrov/models/attention/layers.py
	
class LuongsAttention(tf.keras.Model):
  
  def __init__(self, units, debug_mode=False):
    super(LuongsAttention, self).__init__()
    
    self.W = tf.keras.layers.Dense(units)
	
    self.debug_mode = debug_mode

  #def call(self, query, values):
  def  call(self, H, EO):  # H: Hidden State   , EO: Encoder Output  
    
    printb(".....................................(ATTENTION STR).....................................\n", printable=self.debug_mode)
    

    # H.shape (hidden shape) == (batch_size, unit) 
	# H.shape (after expand) == (batch_size, 1, unit)
    # EO.shape = batch_size, Tx, unit)     
    # we are doing this to perform addition to calculate the score
	
	# query = H
	# values = EO
	
    H  = tf.expand_dims(H, 1)
    printb("(H) [hidden after expand]: {}".format(H.shape), printable=self.debug_mode)


    # score shape == (batch_size, max_length, hidden_size)	
	# Matrix Before transposition: [batch_size, max_length, hidden_size] After transposition: [batch_size, hidden_size, max_length]
	

    score = tf.matmul(H, self.W(EO), transpose_b=True)
    printb("(score) [tf.matmul(H, self.W(EO)]: {}".format(score.shape), printable=self.debug_mode)

    
    # attention_weights shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    attention_weights = tf.nn.softmax(score, axis=2)	
    printb("(attention_weights) [softmax(score, axis=2)]: {}".format(attention_weights.shape), printable=self.debug_mode)

    
    #context_vector shape after sum == (batch_size, hidden_size)
    #context_vector = attention_weights * EO
    context_vector = tf.matmul(attention_weights, EO)
    printb("(context_vector) [tf.matmul(attention_weights, EO)]: {}".format(context_vector.shape), printable=self.debug_mode)
    
    context_vector = tf.reduce_sum(context_vector, axis=1)	
    printb("(context_vector) [reduce_sum(context_vector, axis=1)]: {}".format(context_vector.shape), printable=self.debug_mode)
    
    printb(".....................................(ATTENTION END).....................................\n", printable=self.debug_mode)
    
    return context_vector, attention_weights