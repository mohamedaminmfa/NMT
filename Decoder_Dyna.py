import keras as k
import tensorflow as tf

from SL.lib.NMT_lib.RNN_Dyna import *
from SL.lib.NMT_lib.lang_util import *
from SL.lib.NMT_lib.LuongsAttention_Dyna  import *
from SL.lib.NMT_lib.BahdanauAttention_Dyna  import *
from tensorflow.python.layers import core as layers_core

class Decoder(tf.keras.Model):
#{
    def __init__(self,
                hparams,
                debug_mode=True,
                activation=None, 
                reuse=None, 
                kernel_initializer=None, 
                bias_initializer=None, 
                name=None, 
                dtype=None,
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
                scope=None):
    #{    
        super(Decoder, self).__init__()

        self.num_bi_layers           = hparams.num_bi_layers
        self.num_uni_layers          = hparams.num_uni_layers
        self.bidirectional_mapping   = hparams.bidirectional_mapping

        self.batch_size              = hparams.batch_size
        self.dec_units               = hparams.num_units	
        self.vocab_size              = len(hparams.targ_lang_train.word2idx)
        self.embedding_dim           = hparams.embedding_dim
        self.rnn_cell_type           = hparams.rnn_cell_type
        self.bridge_type             = hparams.bridge_type
        
        self.uni_residual_num_layers = hparams.num_residual_uni_layers        
        self.targ_lang_train         = hparams.targ_lang_train
        self.output_projection       = hparams.decoder_output_projection

        self.input_dropout           = hparams.input_dropout  
        self.output_dropout          = hparams.output_dropout 
        self.state_dropout           = hparams.state_dropout

        self.time_major              = hparams.time_major

        self.decoder_embedding_TimestepDropout  = hparams.decoder_embedding_TimestepDropout
        self.decoder_embedding_SpatialDropout1D = hparams.decoder_embedding_SpatialDropout1D      

        self.rnn_cell_mode           = hparams.rnn_cell_mode
        self.target_tensor_reverse   = hparams.target_tensor_reverse
      
        self.bidirectional_mode      = hparams.bidirectional_mode

        self.debug_mode             = (True if debug_mode and hparams.debug_mode else False)
      
        #================================================================================================================================================================
        
        # Output (Projection) Layer
        self.fc = tf.keras.layers.Dense( self.vocab_size )
        #self.fc = layers_core.Dense(self.vocab_size, use_bias=False, name="output_projection") #output_layer

        #================================================================================================================================================================

        # Embedding Layer
        self.embedding_enable  = hparams.decoder_embedding_enable
        if self.embedding_enable:
        #{
            self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
            #self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1, 1), name="embedding")
        #}

        #================================================================================================================================================================

        # RNN Cells
        # Double Units UNI-RNN Cells
        if hparams.num_bi_layers > 0 and self.bidirectional_mapping.upper() == "double_units".upper():  
        #{    
            self.num_decoder_layers = self.num_bi_layers + self.num_uni_layers
            
            self.uni_rnn_double_train, self.uni_rnn_double_eval= RNN().rnn_nn_dyna_layers(  units= self.dec_units*2, 
                                                                                            rnn_cell_type= self.rnn_cell_type,
                                                                                            learn_mode= None, 
                                                                                            num_layers= self.num_bi_layers,
                                                                                            num_residual_layers= 0,
                                                                                            activation= activation, 
                                                                                            reuse= reuse, 
                                                                                            kernel_initializer= kernel_initializer, 
                                                                                            bias_initializer= bias_initializer, 
                                                                                            name= name, 
                                                                                            dtype= dtype,
                                                                                            input_dropout= self.input_dropout,  
                                                                                            output_dropout= self.output_dropout, 
                                                                                            state_dropout= self.state_dropout, 
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
                                                                                            
                   
        # UNI-RNN Cells
        if hparams.num_uni_layers > 0: 
        #{
            if hparams.num_bi_layers > 0 and self.bidirectional_mapping.upper() == "double_layers".upper():
                self.uni_decoder_num_layers = (hparams.num_bi_layers * 2) + hparams.num_uni_layers
                self.num_decoder_layers     = (self.num_bi_layers * 2) + self.num_uni_layers
                
            else:
                self.uni_decoder_num_layers = self.num_uni_layers
                self.num_decoder_layers     = self.num_uni_layers
                

            self.uni_rnn_train, self.uni_rnn_eval= RNN().rnn_nn_dyna_layers(units= self.dec_units, 
                                                                            rnn_cell_type= self.rnn_cell_type,
                                                                            learn_mode= None, 
                                                                            num_layers= self.uni_decoder_num_layers,
                                                                            num_residual_layers= self.uni_residual_num_layers,
                                                                            activation= activation, 
                                                                            reuse= reuse, 
                                                                            kernel_initializer= kernel_initializer, 
                                                                            bias_initializer= bias_initializer, 
                                                                            name= name, 
                                                                            dtype= dtype,
                                                                            input_dropout= self.input_dropout,  
                                                                            output_dropout= self.output_dropout, 
                                                                            state_dropout= self.state_dropout, 
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

        #================================================================================================================================================================
        
        # Attention        
        self.attention_obj  = None
        self.attention_enable = hparams.attention_enable
        if self.attention_enable:
            
            self.attention_type = hparams.attention_type             

            if self.attention_type.upper() == "bahdanau".upper():
                self.attention_obj = BahdanauAttention(hparams, self.num_decoder_layers, self.debug_mode)

            elif self.attention_type.upper() == "luong".upper():
                self.attention_obj = LuongsAttention(hparams, self.num_decoder_layers, self.debug_mode) 

        #================================================================================================================================================================                                   
    #}

    def call(self, x, hidden, enc_output, LEARN_MODE, t_step=0):  #decoder(dec_input, dec_hidden, enc_output)
    #{
        printb("------------------------------------(DECODER {} STR)-------------------------------------\n".format(t_step), printable=self.debug_mode)

        printb("[DECODER] (x): {}".format(x.shape), printable=self.debug_mode)     
        
        # enc_output shape == (batch_size, max_length, hidden_size)

        decoder_state    = tuple()

        #================================================================================================================================================================

        a = x

        # Embedding Layer
        if self.embedding_enable:
        #{            
            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x_emb = self.embedding(x)
            #x_emb = tf.nn.embedding_lookup(self.embedding, x)

            printb("[DECODER] (x_emb) [embedding(x)]: {}".format(x_emb.shape), printable=self.debug_mode)     

            input_shape=K.shape(x_emb)	   

            if LEARN_MODE == tf.contrib.learn.ModeKeys.TRAIN:
            #{
                x_emb = tf.nn.dropout(x_emb, keep_prob= 1-self.decoder_embedding_TimestepDropout,  noise_shape=(input_shape[0], input_shape[1], 1)) # (TimestepDropout) then zero-out word embeddings
                x_emb = tf.nn.dropout(x_emb, keep_prob= 1-self.decoder_embedding_SpatialDropout1D, noise_shape=(input_shape[0], 1, input_shape[2])) # (SpatialDropout1D) and possibly drop some dimensions on every single embedding (timestep)
                printb("[DECODER] (x_emb) [after dropout_Embedding]: {}".format(x_emb.shape), printable=self.debug_mode)
            #}

            a = x_emb
        #}

        #================================================================================================================================================================

        # Bridge Type
        if self.bridge_type.upper() == "bridge_copy".upper():
            hidden = hidden

        elif self.bridge_type.upper() == "bridge_none".upper():
            hidden = None

        #================================================================================================================================================================

        # Attention
        if( self.attention_obj != None and self.bridge_type.upper() != "bridge_none".upper()):
        #{
            printb("hidden      :\n{}".format(hidden), printable=self.debug_mode)

            context_vector, attention_weights = self.attention_obj(hidden, enc_output)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            context_dec_input = tf.concat([tf.expand_dims(context_vector, 1), x_emb], axis=-1)
            printb("[DECODER] (context_dec_input) [tf.concat([tf.expand_dims(context_vector, 1), x_emb], axis=-1)]: {}".format(context_dec_input.shape), printable=self.debug_mode)
        #}

        else:
        #{
            context_dec_input = x_emb
            printb("[DECODER] (context_dec_input) [context_dec_input = x_emb]: {}".format(context_dec_input.shape), printable=self.debug_mode)
        #}
        
        #================================================================================================================================================================

        # passing the concatenated vector to the RNN cell         
        a = context_dec_input

        if LEARN_MODE == tf.contrib.learn.ModeKeys.TRAIN:
        #{
            if self.num_uni_layers > 0: 
                cell = self.uni_rnn_train

            if self.num_bi_layers > 0 and self.bidirectional_mapping.upper() == "double_units".upper():
                cell_double = self.uni_rnn_double_train
        #}
            
        elif LEARN_MODE == tf.contrib.learn.ModeKeys.EVAL or LEARN_MODE == tf.contrib.learn.ModeKeys.INFER:
        #{ 
            if self.num_uni_layers > 0:               
                cell = self.uni_rnn_eval

            if self.num_bi_layers > 0 and self.bidirectional_mapping.upper() == "double_units".upper():
                cell_double = self.uni_rnn_double_eval                 
        #}


        # Double UNI-Directional RNN Cells
        # Decoder UNI-directional Double

        # Double Units UNI-RNN Cells
        if self.num_bi_layers > 0 and self.bidirectional_mapping.upper() == "double_units".upper():
        #{
            printb("[DECODER] (UNI-Directional Trainable)", printable= self.debug_mode)

            if hidden != None:
                hidden_double = hidden[:self.num_bi_layers]

            a, state_double = tf.nn.dynamic_rnn(cell= cell_double, 
                                                inputs= a, 
                                                sequence_length= None, 
                                                initial_state= hidden_double, 
                                                dtype= tf.float32, 
                                                parallel_iterations= None, 
                                                swap_memory= False, 
                                                time_major= False, 
                                                scope= None)

            decoder_state += state_double

            hidden = hidden[self.num_bi_layers:]
        #}



        # Decoder Uni-directional
        if self.num_uni_layers > 0:
        #{
            a, state_normal = tf.nn.dynamic_rnn(cell= cell, 
                                                inputs= a, 
                                                sequence_length= None, 
                                                initial_state= hidden, 
                                                dtype= tf.float32, 
                                                parallel_iterations= None, 
                                                swap_memory= False, 
                                                time_major= False, 
                                                scope= None)
            
            decoder_state += state_normal
        #}
        

        decoder_state = tuple(decoder_state)

        if self.output_projection:
        #{
            # a shape == (batch_size * 1, hidden_size)
            a = tf.reshape(a, (-1, a.shape[2]))

            # output shape == (batch_size, vocab)
            output = self.fc(a)
        #}
        else:
            output = a
        

        printb("--------------------------------------(DECODER END)--------------------------------------\n", printable=self.debug_mode)

        return output, decoder_state
    #}

    def initialize_hidden_state(self):
    #{
        return tf.zeros((self.batch_size, self.dec_units))
    #}
#}