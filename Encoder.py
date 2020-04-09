import keras as k
import tensorflow as tf

from SL.lib.NMT_lib.RNN import *
from SL.lib.NMT_lib.lang_util import *

#UNIDIR_LAYER , BIDIR_LAYER: 
#[GRU, LSTM]
#[  1,    0]

#[GRU, LSTM, GRU, LSTM, GRU, LSTM, GRU, LSTM]
#[  0,    2,   0,    2,   1,    0,   1,    1]

#encoder_input_dropout                = [[0.2], [0.2, 0.3]]
#encoder_recurrent_dropout            = [[0.2], [0.2, 0.3]]

class Encoder(tf.keras.Model):
#{
    def __init__(self,
                hparams,
                return_sequences=True,
                return_state=True,
                BI_return_state=True,                                                                                   
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
				debug_mode=False):

    #{    
        super(Encoder, self).__init__()
               
        self.batch_size             = hparams.batch_size
        self.enc_units              = hparams.num_units

        self.bi_encoder_num_layers  = np.sum(hparams.model_layer[0])
        self.uni_encoder_num_layers = np.sum(hparams.model_layer[1])
		
        self.residual_enabled       = hparams.residual_enabled
        self.num_residual_bi_layers = hparams.num_residual_bi_layers
        self.num_residual_uni_layers= hparams.num_residual_uni_layers

        self.rnn_cell_mode          = hparams.rnn_cell_mode
        #self.rnn_cell_type          = hparams.rnn_cell_type

        self.source_tensor_reverse  = hparams.source_tensor_reverse
        self.vocab_size             = len(hparams.inp_lang_train.word2idx)
        self.embedding_dim          = hparams.embedding_dim
        self.embedding_enable       = hparams.embedding_enable
        self.attention_enable       = hparams.attention_enable
        self.bidirectional_mode     = hparams.bidirectional_mode
 
        self.input_dropout          = hparams.input_dropout  
        self.output_dropout         = hparams.output_dropout 
        self.state_dropout          = hparams.state_dropout

        self.encoder_embedding_TimestepDropout  = hparams.encoder_embedding_TimestepDropout
        self.encoder_embedding_SpatialDropout1D = hparams.encoder_embedding_SpatialDropout1D

        self.debug_mode             = (True if debug_mode and hparams.debug_mode else False)
        self.time_major             = hparams.time_major

        self.bidirectional_mapping  = hparams.bidirectional_mapping        


        self.model_encoder_list     = []
        self.TRAINING               = False

        #=================================================================================================================================================

        if self.embedding_enable:
        #{
            self.embedding= tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
            #self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1, 1), name="embedding")

            self.encoder_embedding_TimestepDropout  = self.encoder_embedding_TimestepDropout
            self.encoder_embedding_SpatialDropout1D = self.encoder_embedding_SpatialDropout1D
        #}
		
        #=================================================================================================================================================

        #Encoder BI-DIRECTIONAL
        numLayer_counter = 0        
        for index, numLayer in enumerate(hparams.model_layer[0]):
        #{    
            #BI-LSTM 
            if ( (index+1) % 2 == 0 ): 
            #{    
                for n in range(numLayer):                    
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(self.input_dropout) ==1:
                        Ni = 0
                    
                    if len(self.state_dropout) ==1:
                        Nr = 0

                    self.BI_lstm = RNN().bidirectional_lstm_keras(  units= self.enc_units,
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
                                                                    return_state= BI_return_state,
                                                                    go_backwards= go_backwards, 
                                                                    stateful= stateful,
                                                                    unroll= unroll,
                                                                    merge_mode= merge_mode,
                                                                    backward_layer= backward_layer,
                                                                    weights= weights)
                    
                                                    
                    self.model_encoder_list.append( self.BI_lstm )
                    print("[ENCODER] BI_LSTM")
                #}        
            #}
            

            #BI-GRU
            else:
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(self.input_dropout) ==1:
                        Ni = 0
                    
                    if len(self.state_dropout) ==1:
                        Nr = 0
                    
                    self.BI_gru = RNN().bidirectional_gru_keras(units =self.enc_units,
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
                                                                return_state= BI_return_state, 
                                                                go_backwards= go_backwards, 
                                                                stateful= stateful, 
                                                                unroll= unroll, 
                                                                reset_after= reset_after,
                                                                merge_mode= merge_mode,
                                                                backward_layer= backward_layer,
                                                                weights= weights)                           
                                                          
                    self.model_encoder_list.append( self.BI_gru )
                    print("[ENCODER]  BI_GRU")
                #}  
            #}
            
            numLayer_counter += numLayer
        #}           

        #=================================================================================================================================================

        #Encoder UNI-DIRECTIONAL        
        for index, numLayer in enumerate(hparams.model_layer[1]):
        #{
            #UNI-LSTM
            if ( (index+1) % 2 == 0 ):              
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(self.input_dropout) ==1:
                        Ni = 0
                    
                    if len(self.state_dropout) ==1:
                        Nr = 0                    
                    
                    self.lstm = RNN().lstm_keras(units= self.enc_units, 
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
                    
                    self.model_encoder_list.append( self.lstm )
                    print("[ENCODER] UNI_LSTM")
                #}
            #}
            
            
            #UNI-GRU
            else:
            #{    
                for n in range(numLayer):
                #{    
                    Ni = numLayer_counter + n
                    Nr = numLayer_counter + n    
                    
                    if len(self.input_dropout) ==1:
                        Ni = 0
                    
                    if len(self.state_dropout) ==1:
                        Nr = 0                    
                    
                    self.gru = RNN().gru_keras( units=self.enc_units,
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
                                    
                    self.model_encoder_list.append( self.gru )
                    print("[ENCODER] UNI_GRU")
                #}
            #}
            
            numLayer_counter += numLayer           
        #}                    
    #}
    
    def call(self, x, hidden, LEARN_MODE):  #encoder(inp, hidden)             called from NMT_Model.py
    #{
        printb("--------------------------------------(ENCODER STR)--------------------------------------\n", printable=self.debug_mode)
        
        printb("[ENCODER] (x):{}".format(x.shape) , printable=self.debug_mode)
        printb("[ENCODER] (hidden) : {}".format(hidden.shape), printable=self.debug_mode)
                
        encoder_state = list()
                
        if self.source_tensor_reverse:
            x = reverse_tensor(x)
        
        if self.embedding_enable:
        #{
            x_emb = self.embedding(x)
            printb("[ENCODER] (x_emb) [embedding(x)]: {}".format(x_emb.shape), printable=self.debug_mode)		

            input_shape=K.shape(x_emb)	   

            if LEARN_MODE == tf.contrib.learn.ModeKeys.TRAIN:
            #{
                self.TRAINING = True

                x_emb = tf.nn.dropout(x_emb, keep_prob= 1-self.encoder_embedding_TimestepDropout,  noise_shape=(input_shape[0], input_shape[1], 1)) # (TimestepDropout) then zero-out word embeddings
                x_emb = tf.nn.dropout(x_emb, keep_prob= 1-self.encoder_embedding_SpatialDropout1D, noise_shape=(input_shape[0], 1, input_shape[2])) # (SpatialDropout1D) and possibly drop some dimensions on every single embedding (timestep)
                printb("[ENCODER] (x_emb) [after dropout_Embedding]: {}".format(x_emb.shape), printable=self.debug_mode)
            #}
        #}

        a              = x_emb
        residual_x     = a
        stateH         = hidden
        stateC         = hidden
                
        dropout_step   = 0  
        bi_index_step  = 0
        uni_index_step = 0

        for index, LayerObject in enumerate(self.model_encoder_list): 
        #{
			#...............................................................................................................................................

            #Encoder BI-directional RNN		
            if(isinstance( LayerObject, tf.keras.layers.Bidirectional )):
            #{			
                residual_x = a
                 
                # Check if the Object is Bi-Directional GRU or Bi-Directional LSTM
                a_state_check = LayerObject(inputs= a, initial_state= None, training= self.TRAINING)
                
                if len(a_state_check) == 3:
                #{
                    # GRU
                    a_state = LayerObject(inputs= a, initial_state = None, training= self.TRAINING)
                    a      = a_state[0]
                    stateH = tf.concat([a_state[1], a_state[2]], 1)
                    stateC = stateH

                    encoder_state.append([stateC, stateH])
                #}

                else:
                #{
                    # LSTM
                    a_state = LayerObject(inputs= a, initial_state = None, training= self.TRAINING)
                    a      = a_state[0]
                    stateH = tf.concat([a_state[1], a_state[3]], 1)
                    stateC = tf.concat([a_state[2], a_state[4]], 1)

                    encoder_state.append([stateC, stateH])
                #}                    
                
                if self.residual_enabled and (bi_index_step < self.bi_encoder_num_layers) and (bi_index_step >= (self.bi_encoder_num_layers - self.num_residual_bi_layers)):            
                    a = tf.math.add(a, residual_x)
                    bi_index_step +=1
                                    

                printb("[ENCODER] (a)      Encoder BI-DIRECTIONAL: {}".format(a.shape), printable=self.debug_mode)
            #}

            #...............................................................................................................................................

            #Encoder UNI-directional RNN
            else:
            #{
                residual_x = a

                if( isinstance( LayerObject, tf.keras.layers.CuDNNLSTM ) or isinstance( LayerObject, tf.keras.layers.LSTM )):
                    enc_hidden = [hidden, hidden]

                elif( isinstance( LayerObject, tf.keras.layers.CuDNNGRU ) or isinstance( LayerObject, tf.keras.layers.GRU )):
                    enc_hidden = hidden

                a_state = LayerObject(inputs= a , initial_state= enc_hidden , training= self.TRAINING)

                if len(a_state) == 3:
                    a      = a_state[0]
                    stateH = a_state[1]
                    stateC = a_state[2]
                else:
                    a      = a_state[0]
                    stateH = a_state[1]
                    stateC = stateH

                encoder_state.append([stateC, stateH])                
            
                if self.residual_enabled and (uni_index_step < self.uni_encoder_num_layers) and (uni_index_step >= (self.uni_encoder_num_layers - self.num_residual_uni_layers)):            
                    a = tf.math.add(a, residual_x) 
                    uni_index_step +=1
                

                printb("[ENCODER] (a)      Encoder UNI: {}".format(a.shape), printable=self.debug_mode)
                printb("[ENCODER] (stateH) Encoder UNI: {}".format(stateH.shape), printable=self.debug_mode)
                printb("[ENCODER] (stateC) Encoder UNI: {}".format(stateC.shape), printable=self.debug_mode)
            #}          

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
        
        output = a

        printb("[ENCODER] (output) :{}".format(output.shape), printable=self.debug_mode)
        printb("[ENCODER] (stateH) :{}".format(stateH.shape), printable=self.debug_mode)
        printb("[ENCODER] (stateC) :{}".format(stateC.shape), printable=self.debug_mode)
        
        printb("--------------------------------------(ENCODER END)--------------------------------------\n", printable=self.debug_mode)

        return output, encoder_state    
    #}
    
    def initialize_hidden_state(self):
    #{    
        return tf.zeros((self.batch_size, self.enc_units))
    
        """
        https://www.kaggle.com/jaikishore7/tensorflow-eager-language-model?scriptVersionId=5784188
        
        if state == "train":
            x,_,_ = self.LSTM_1(train_values, initial_state = [self.hiddenH,self.hiddenC] )
            x,_,_ = self.LSTM_2(x, initial_state = [self.lstm2_ht,self.lstm2_ct] )
            x = self.out(x)
            return x

        else:
            x,lstm_1_ht,lstm_1_ct = self.LSTM_1(train_values,initial_state = [hiddenH,hiddenC])
            x,lstm_2_ht,lstm_2_ct = self.LSTM_2(x,initial_state = [lstm_2_ht,lstm_2_ct] )
            x = self.out(x)
            return x
        """
    #}
#}