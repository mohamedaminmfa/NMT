import keras as k
import tensorflow as tf

from SL.lib.NMT_lib.RNN import *
from SL.lib.NMT_lib.lang_util import *
from SL.lib.NMT_lib.LuongsAttention  import *
from SL.lib.NMT_lib.BahdanauAttention  import *

#UNIDIR_LAYER , BIDIR_LAYER: 
#[GRU, LSTM]
#[  1,    0]

#[GRU, LSTM, GRU, LSTM, GRU, LSTM, GRU, LSTM]
#[  0,    2,   0,    2,   1,    0,   1,    1]

class Decoder(tf.keras.Model):

    def __init__(self,
                hparams,
                return_sequences=True,
                return_state=True,              
                activation='tanh', 
                recurrent_activation='hard_sigmoid', 
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                recurrent_initializer='orthogonal', 
                bias_initializer='zeros', 
                unit_forget_bias=True, 
                kernel_regularizer=None, 
                recurrent_regularizer=None,
                bias_regularizer=None, 
                activity_regularizer=None,
                kernel_constraint=None, 
                recurrent_constraint=None, 
                bias_constraint=None,                 
                implementation=1, 
                go_backwards=False, 
                stateful=False,
                unroll=False,
				reset_after= False,
                merge_mode="concat",
                backward_layer= None,
                weights=None,
                output_projection=True,
				debug_mode=False):

    #{    
        super(Decoder, self).__init__()
		
        self.bi_encoder_num_layers  = np.sum(hparams.model_layer[0])
        self.uni_encoder_num_layers = np.sum(hparams.model_layer[1])		
        
        self.bidirectional_mapping   = hparams.bidirectional_mapping

        if self.bidirectional_mapping.upper() == "double_layers".upper():
            self.num_uni_layers_decoder_double = 0
            self.num_uni_layers_decoder_normal = (self.bi_encoder_num_layers * 2) + self.uni_encoder_num_layers
            
        elif self.bidirectional_mapping.upper() == "double_units".upper():
            self.num_uni_layers_decoder_double = self.bi_encoder_num_layers
            self.num_uni_layers_decoder_normal = self.uni_encoder_num_layers

        
        self.batch_size              = hparams.batch_size
        self.dec_units               = hparams.num_units	
        self.vocab_size              = len(hparams.targ_lang_train.word2idx)
        self.embedding_dim           = hparams.embedding_dim
        self.rnn_cell_type           = hparams.rnn_cell_type
        self.bridge_type             = hparams.bridge_type

        self.residual_enabled        = hparams.residual_enabled
        self.num_residual_bi_layers  = hparams.num_residual_bi_layers         
        self.num_residual_uni_layers = hparams.num_residual_uni_layers  

        self.targ_lang_train         = hparams.targ_lang_train
        self.output_projection       = hparams.decoder_output_projection

        self.input_dropout           = hparams.input_dropout  
        self.output_dropout          = hparams.output_dropout 
        self.state_dropout           = hparams.state_dropout

        self.time_major              = hparams.time_major

        self.embedding_enable       = hparams.embedding_enable
        self.decoder_embedding_TimestepDropout  = hparams.decoder_embedding_TimestepDropout
        self.decoder_embedding_SpatialDropout1D = hparams.decoder_embedding_SpatialDropout1D      

        self.rnn_cell_mode           = hparams.rnn_cell_mode
        self.target_tensor_reverse   = hparams.target_tensor_reverse
      
        self.bidirectional_mode      = hparams.bidirectional_mode

        self.debug_mode             = (True if debug_mode and hparams.debug_mode else False)

        self.model_decoder_list     = []
        self.TRAINING               = False

        #================================================================================================================================================================

        # Embedding Layer
        self.embedding_enable  = hparams.embedding_enable
        if self.embedding_enable:
        #{
            self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
            #self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1, 1), name="embedding")
        #}

        #================================================================================================================================================================
        
        # Attention        
        self.attention_obj  = None
        self.attention_enable = hparams.attention_enable
        
        if self.attention_enable:

            self.attention_type = hparams.attention_type             

            if self.attention_type.upper() == "bahdanau".upper():
                self.attention_obj = BahdanauAttention(hparams, (self.num_uni_layers_decoder_double + self.num_uni_layers_decoder_normal), self.debug_mode)

            elif self.attention_type.upper() == "luong".upper():
                self.attention_obj = LuongsAttention(self.dec_units, self.debug_mode) 

        #================================================================================================================================================================
        
        # Output (Projection) Layer
        self.fc = tf.keras.layers.Dense( self.vocab_size )
        #self.fc = layers_core.Dense(self.vocab_size, use_bias=False, name="output_projection") #output_layer

        #================================================================================================================================================================

        numLayer_counter = 0
        
        # Decoder UNI-directional Double

        if self.bidirectional_mapping.upper() == "double_units".upper():
        #{
            for index, numLayer in enumerate(hparams.model_layer[0]):
            #{
                if ( (index+1) % 2 == 0 ): #UNI-LSTM              
                #{    
                    for n in range(numLayer):
                    #{    
                        Ni = numLayer_counter + n
                        Nr = numLayer_counter + n    

                        if len(hparams.input_dropout) == 1:
                            Ni = 0

                        if len(hparams.state_dropout) == 1:
                            Nr = 0

                        self.lstm_uni_double = RNN().lstm_keras(units= (self.dec_units*2), 
                                                                mode= self.rnn_cell_mode,
                                                                activation= activation, 
                                                                recurrent_activation= recurrent_activation, 
                                                                use_bias= use_bias, 
                                                                kernel_initializer= kernel_initializer, 
                                                                recurrent_initializer= recurrent_initializer, 
                                                                bias_initializer= bias_initializer, 
                                                                unit_forget_bias= unit_forget_bias, 
                                                                kernel_regularizer= kernel_regularizer, 
                                                                recurrent_regularizer= recurrent_regularizer,
                                                                bias_regularizer= bias_regularizer, 
                                                                activity_regularizer= activity_regularizer,
                                                                kernel_constraint= kernel_constraint, 
                                                                recurrent_constraint= recurrent_constraint, 
                                                                bias_constraint= bias_constraint, 
                                                                dropout= self.input_dropout[Ni],
                                                                recurrent_dropout= self.state_dropout[Nr],
                                                                implementation= implementation, 
                                                                return_sequences= return_sequences,
                                                                return_state= return_state,
                                                                go_backwards= go_backwards, 
                                                                stateful= stateful,
                                                                unroll= unroll)
                                        
                        self.model_decoder_list.append( self.lstm_uni_double )
                        print("[DECODER] UNI_LSTM")
                    #}    
                #} 

                else: #UNI-GRU
                #{    
                    for n in range(numLayer):
                    #{   
                        Ni = numLayer_counter + n
                        Nr = numLayer_counter + n    

                        if len(hparams.input_dropout) == 1:
                            Ni = 0

                        if len(hparams.state_dropout) == 1:
                            Nr = 0

                        gru_uni_double = RNN().gru_keras(units= (self.dec_units*2),
                                                        mode= self.rnn_cell_mode,
                                                        activation= activation, 
                                                        recurrent_activation= recurrent_activation, 
                                                        use_bias= use_bias, 
                                                        kernel_initializer= kernel_initializer, 
                                                        recurrent_initializer= recurrent_initializer,
                                                        bias_initializer= bias_initializer, 
                                                        kernel_regularizer= kernel_regularizer, 
                                                        recurrent_regularizer= recurrent_regularizer, 
                                                        bias_regularizer= bias_regularizer,
                                                        activity_regularizer= activity_regularizer,
                                                        kernel_constraint= kernel_constraint,
                                                        recurrent_constraint= recurrent_constraint,
                                                        bias_constraint= bias_constraint,
                                                        dropout= self.input_dropout[Ni], 
                                                        recurrent_dropout= self.state_dropout[Nr],
                                                        implementation= implementation, 
                                                        return_sequences= return_sequences,
                                                        return_state= return_state, 
                                                        go_backwards= go_backwards, 
                                                        stateful= stateful, 
                                                        unroll= unroll, 
                                                        reset_after=reset_after)
                                        
                        self.model_decoder_list.append( gru_uni_double ) 
                        print("[DECODER] UNI_GRU")
                    #}
                #}

                numLayer_counter += numLayer
            #}
        #}

        #=================================================================================================================================================

        # UNI-Directional Normal
        # Decoder UNI-directional Normal

        if self.bi_encoder_num_layers > 0 and self.bidirectional_mapping.upper() == "double_layers".upper():
            self.uni_decoder_layers = np.add( np.multiply(hparams.model_layer[0], 2), hparams.model_layer[1])
            
        else:
            self.uni_decoder_layers = hparams.model_layer[1]


        for index, numLayer in enumerate( self.uni_decoder_layers ):
        #{
            #UNI-LSTM 
            if ( (index+1) % 2 == 0 ):              
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    

                    if len(hparams.input_dropout) == 1:
                        Ni = 0

                    if len(hparams.state_dropout) == 1:
                        Nr = 0

                    lstm = RNN().lstm_keras(units= self.dec_units, 
                                            mode= self.rnn_cell_mode,
                                            activation= activation, 
                                            recurrent_activation= recurrent_activation, 
                                            use_bias= use_bias, 
                                            kernel_initializer= kernel_initializer, 
                                            recurrent_initializer= recurrent_initializer, 
                                            bias_initializer= bias_initializer, 
                                            unit_forget_bias= unit_forget_bias, 
                                            kernel_regularizer= kernel_regularizer, 
                                            recurrent_regularizer= recurrent_regularizer,
                                            bias_regularizer= bias_regularizer, 
                                            activity_regularizer= activity_regularizer,
                                            kernel_constraint= kernel_constraint, 
                                            recurrent_constraint= recurrent_constraint, 
                                            bias_constraint= bias_constraint, 
                                            dropout= self.input_dropout[Ni],
                                            recurrent_dropout= self.state_dropout[Nr],
                                            implementation= implementation, 
                                            return_sequences= return_sequences,
                                            return_state= return_state,
                                            go_backwards= go_backwards, 
                                            stateful= stateful,
                                            unroll= unroll)
                                    
                    self.model_decoder_list.append( lstm )
                    print("[DECODER] UNI_LSTM")
                #}    
            #} 

            
            #UNI-GRU
            else: 
            #{    
                for n in range(numLayer):
                #{   
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    

                    if len(hparams.input_dropout) == 1:
                        Ni = 0

                    if len(hparams.state_dropout) == 1:
                        Nr = 0

                    gru = RNN().gru_keras(  units=self.dec_units,
                                            mode= self.rnn_cell_mode,
                                            activation= activation, 
                                            recurrent_activation= recurrent_activation, 
                                            use_bias= use_bias, 
                                            kernel_initializer= kernel_initializer, 
                                            recurrent_initializer= recurrent_initializer,
                                            bias_initializer= bias_initializer, 
                                            kernel_regularizer= kernel_regularizer, 
                                            recurrent_regularizer= recurrent_regularizer, 
                                            bias_regularizer= bias_regularizer,
                                            activity_regularizer= activity_regularizer,
                                            kernel_constraint= kernel_constraint,
                                            recurrent_constraint= recurrent_constraint,
                                            bias_constraint= bias_constraint,
                                            dropout= self.input_dropout[Ni], 
                                            recurrent_dropout= self.state_dropout[Nr],
                                            implementation= implementation, 
                                            return_sequences= return_sequences,
                                            return_state= return_state, 
                                            go_backwards= go_backwards, 
                                            stateful= stateful, 
                                            unroll= unroll, 
                                            reset_after=reset_after)
                                    
                    self.model_decoder_list.append( gru ) 
                    print("[DECODER] UNI_GRU")
                #}
            #}

            numLayer_counter += numLayer        
        #}
    #}    

    def call(self, x, hidden, enc_output, LEARN_MODE, t_step):  #decoder(dec_input, dec_hidden, enc_output)
    #{    
        printb("------------------------------------(DECODER {} STR)-------------------------------------\n".format(t_step), printable=self.debug_mode)
        
        printb("[DECODER] (x): {}".format(x.shape), printable=self.debug_mode)     
        
        # enc_output shape == (batch_size, max_length, hidden_size)

        decoder_state = list()
                
        #================================================================================================================================================================        

        # Embedding Layer
        if self.embedding_enable:
        #{          
            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x_emb = self.embedding(x)
            printb("[DECODER] (x_emb) [embedding(x)]: {}".format(x_emb.shape), printable=self.debug_mode)     

            input_shape=K.shape(x_emb)	   

            if LEARN_MODE == tf.contrib.learn.ModeKeys.TRAIN:
            #{
                self.TRAINING = True

                x_emb = tf.nn.dropout(x_emb, keep_prob= 1-self.decoder_embedding_TimestepDropout,  noise_shape=(input_shape[0], input_shape[1], 1)) # (TimestepDropout) then zero-out word embeddings
                x_emb = tf.nn.dropout(x_emb, keep_prob= 1-self.decoder_embedding_SpatialDropout1D, noise_shape=(input_shape[0], 1, input_shape[2])) # (SpatialDropout1D) and possibly drop some dimensions on every single embedding (timestep)
                printb("[DECODER] (x_emb) [after dropout_Embedding]: {}".format(x_emb.shape), printable=self.debug_mode)
            #}
        #}

        #================================================================================================================================================================

        # Attention
        attention_weights = None
        if( self.attention_obj != None):
        #{            
            context_vector, attention_weights = self.attention_obj(hidden[ len(hidden)-1 ][0], enc_output)

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
      
        a      = context_dec_input

        dropout_step   = 0
        bi_index_step  = 0
        uni_index_step = 0

        for index, LayerObject in enumerate(self.model_decoder_list): 
        #{ 
            residual_x = a
            
            if( isinstance( LayerObject, tf.keras.layers.CuDNNLSTM ) or isinstance( LayerObject, tf.keras.layers.LSTM )):
                dec_hidden = hidden[index]

            elif( isinstance( LayerObject, tf.keras.layers.CuDNNGRU ) or isinstance( LayerObject, tf.keras.layers.GRU )):
                dec_hidden =hidden[index][0]

            a_state = LayerObject(inputs= a, initial_state= dec_hidden, training= self.TRAINING)

            if len(a_state) == 3:
                a      = a_state[0]
                stateH = a_state[1]
                stateC = a_state[2]
            else:
                a      = a_state[0]
                stateH = a_state[1]
                stateC = stateH

            decoder_state.append([stateC, stateH])


            if self.residual_enabled:
            #{    
                if self.bidirectional_mapping.upper() == "double_layers".upper():
                    if (bi_index_step < (self.bi_encoder_num_layers*2) and bi_index_step >= ((self.bi_encoder_num_layers*2) - self.num_residual_bi_layers)):            
                        a = tf.math.add(a, residual_x)  
                        bi_index_step += 1


                elif self.bidirectional_mapping.upper() == "double_units".upper():
                    if (bi_index_step < self.bi_encoder_num_layers) and (bi_index_step >= (self.bi_encoder_num_layers - self.num_residual_bi_layers)):            
                        a = tf.math.add(a, residual_x)  
                        bi_index_step += 1


                elif (uni_index_step < self.uni_encoder_num_layers) and (uni_index_step >= (self.uni_encoder_num_layers - self.num_residual_uni_layers)):            
                    a = tf.math.add(a, residual_x)
                    uni_index_step += 1
            #}

            printb("[DECODER] (a)      Decoder LSTM with Attention: {}".format(a.shape), printable=self.debug_mode)
            printb("[DECODER] (stateH) Decoder LSTM with Attention: {}".format(stateH.shape), printable=self.debug_mode)
            printb("[DECODER] (stateC) Decoder LSTM with Attention: {}".format(stateC.shape), printable=self.debug_mode)        
        

            #...............................................................................................................................................
            
            #Apply Output Dropout
            if(index % len(self.output_dropout) == 0):
                dropout_step =0
            else:
                dropout_step +=1

            if LEARN_MODE == tf.contrib.learn.ModeKeys.TRAIN:
                a = tf.nn.dropout(a, keep_prob= 1- self.output_dropout[dropout_step])
                printb("[ENCODER] (a)     Encoder Dropout Layer: {}".format(a.shape), printable=self.debug_mode)          
        #}

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

        return output, decoder_state, attention_weights
    #}

    def initialize_hidden_state(self):
    #{
        return tf.zeros((self.batch_size, self.dec_units))
    #}
#}