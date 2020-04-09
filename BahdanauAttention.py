import keras as k
import tensorflow as tf


from SL.lib.NMT_lib.lang_util import *

#https://www.cnblogs.com/lovychen/p/10470564.html
#https://www.tensorflow.org/tutorials/text/nmt_with_attention

class BahdanauAttention(tf.keras.Model):
  
  def __init__(self, hparams, num_layers, debug_mode=False):
    super(BahdanauAttention, self).__init__()
    
    self.W1 = tf.keras.layers.Dense(hparams.num_units)
    self.W2 = tf.keras.layers.Dense(hparams.num_units)
    self.V  = tf.keras.layers.Dense(1)

    self.num_layers = num_layers
    self.debug_mode = (True if debug_mode and hparams.debug_mode else False)
  
 #def call(self, query, values):
  def call(self, H, EO):  # H: Hidden State   , EO: Encoder Output
    
    printb(".....................................(ATTENTION STR).....................................\n", printable=self.debug_mode)
    
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    
    printb("(EO): {}\n".format(EO.shape), printable=self.debug_mode)
    printb("(H) : {}\n{}\n".format("", H), printable=self.debug_mode)

    H_exp = tf.expand_dims(H, 1)

    printb("(H_exp) (hidden after expand) [tf.expand_dims(H[self.num_layers - 1], 1)]: {}\n{}\n".format(H_exp.shape, H_exp), printable=self.debug_mode)

    
    # score shape == (batch_size, max_length, hidden_size)
    # applying tanh(FC(EO) + FC(H)) to self.V
    score = self.V(tf.nn.tanh(self.W1(EO) + self.W2(H_exp)))
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
