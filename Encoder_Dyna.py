import keras as k
import tensorflow as tf

from SL.lib.NMT_lib.RNN_Dyna import *
from SL.lib.NMT_lib.lang_util import *

class Encoder(tf.keras.Model):
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
        super(Encoder, self).__init__()
        
        self.batch_size             = hparams.batch_size
        self.enc_units              = hparams.num_units

        self.bi_encoder_num_layers  = hparams.num_bi_layers
        self.uni_encoder_num_layers = hparams.num_uni_layers

        self.bi_residual_num_layers = hparams.num_residual_bi_layers
        self.uni_residual_num_layers= hparams.num_residual_uni_layers

        self.rnn_cell_mode          = hparams.rnn_cell_mode
        self.rnn_cell_type          = hparams.rnn_cell_type

        self.source_tensor_reverse  = hparams.source_tensor_reverse
        self.vocab_size             = len(hparams.inp_lang_train.word2idx)
        self.embedding_dim          = hparams.embedding_dim
        self.embedding_enable       = hparams.encoder_embedding_enable
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
        

        if self.embedding_enable:
        #{
            self.embedding= tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
            #self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1, 1), name="embedding")

            self.encoder_embedding_TimestepDropout  = self.encoder_embedding_TimestepDropout
            self.encoder_embedding_SpatialDropout1D = self.encoder_embedding_SpatialDropout1D
        #}
        
        if self.bi_encoder_num_layers > 0:
            #BI-DIRECTIONAL
            self.rnn_train_fw, self.rnn_eval_fw,  self.rnn_train_bw, self.rnn_eval_bw= RNN().bidirectional_rnn_nn_dyna_layers(  units= self.enc_units, 
                                                                                                                                rnn_cell_type= self.rnn_cell_type,
                                                                                                                                learn_mode= None, 
                                                                                                                                num_layers= self.bi_encoder_num_layers,
                                                                                                                                num_residual_layers= self.bi_residual_num_layers,
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
                                                                                                                                state_is_tuple= state_is_tuple,
																																bidirectional_mode= self.bidirectional_mode)
        if self.uni_encoder_num_layers > 0:
            #UNI-DIRECTIONAL        
            self.rnn_train, self.rnn_eval= RNN().rnn_nn_dyna_layers(units= self.enc_units, 
                                                                    rnn_cell_type= self.rnn_cell_type,
                                                                    learn_mode= None, 
                                                                    num_layers= self.uni_encoder_num_layers,
                                                                    num_residual_layers= self.uni_residual_num_layers,
                                                                    activation= activation, 
                                                                    reuse= reuse, 
                                                                    kernel_initializer= kernel_initializer, 
                                                                    bias_initializer= bias_initializer, 
                                                                    name= name, 
                                                                    dtype= dtype,
                                                                    input_dropout=  self.input_dropout,  
                                                                    output_dropout= self.output_dropout, 
                                                                    state_dropout=  self.state_dropout, 
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
    
    def call(self, x, hidden, LEARN_MODE= tf.contrib.learn.ModeKeys.TRAIN):  #encoder(inp, hidden, LEARN_MODE)             called from NMT_Model.py
    #{
        printb("--------------------------------------(ENCODER STR)--------------------------------------\n", printable=self.debug_mode)
        
        printb("[ENCODER] (x):{}".format(x.shape) , printable=self.debug_mode)
            
        if self.source_tensor_reverse:
        #{
            #x = np.fliplr(x)
            x = tf.reverse(x, [-1])
        #}

        a = x

        # Embedding Layer
        if self.embedding_enable:
        #{
            x_emb = self.embedding(x)
            #x_emb = tf.nn.embedding_lookup(self.embedding, x)
            
            printb("[ENCODER] (x_emb) [embedding(x)]: {}".format(x_emb.shape), printable=self.debug_mode)		

            input_shape=K.shape(x_emb)	   

            if LEARN_MODE == tf.contrib.learn.ModeKeys.TRAIN:
            #{
                x_emb = tf.nn.dropout(x_emb, keep_prob= 1-self.encoder_embedding_TimestepDropout,  noise_shape=(input_shape[0], input_shape[1], 1)) # (TimestepDropout) then zero-out word embeddings
                x_emb = tf.nn.dropout(x_emb, keep_prob= 1-self.encoder_embedding_SpatialDropout1D, noise_shape=(input_shape[0], 1, input_shape[2])) # (SpatialDropout1D) and possibly drop some dimensions on every single embedding (timestep)
                printb("[ENCODER] (x_emb) [after dropout_Embedding]: {}".format(x_emb.shape), printable=self.debug_mode)
            #}

            a = x_emb
        #}

        bi_state    = None
        bi_state_fw = None
        bi_state_bw = None
        uni_state   = None
        enc_state   = None 

        encoder_state = tuple()        

        if LEARN_MODE == tf.contrib.learn.ModeKeys.TRAIN:
        #{
            if self.bi_encoder_num_layers > 0:
                cell_fw = self.rnn_train_fw
                cell_bw = self.rnn_train_bw

            if self.uni_encoder_num_layers > 0:
                cell    = self.rnn_train
        #}
            
        elif LEARN_MODE == tf.contrib.learn.ModeKeys.EVAL or LEARN_MODE == tf.contrib.learn.ModeKeys.INFER:
        #{ 
            if self.bi_encoder_num_layers > 0:
                cell_fw = self.rnn_eval_fw
                cell_bw = self.rnn_eval_bw

            if self.uni_encoder_num_layers > 0:
                cell    = self.rnn_eval
        #}

        printb("[ENCODER] tf.contrib.learn.ModeKeys.TRAIN", printable=self.debug_mode)

        #BI-Directional RNN Cells
        if self.bi_encoder_num_layers > 0:
        #{
            printb("[ENCODER] (BI-Directional Trainable)", printable= self.debug_mode)

            if self.bidirectional_mode.upper() == "STACK_BIDIRECTIONAL".upper():
            #{
                initial_states_fw= [f_l.zero_state(self.batch_size, tf.float32) for f_l in cell_fw]
                initial_states_bw= [b_l.zero_state(self.batch_size, tf.float32) for b_l in cell_bw]

                (a, bi_encoder_state_fw, bi_encoder_state_bw) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn( cells_fw= cell_fw, 
                                                                                                                cells_bw= cell_bw,
                                                                                                                inputs= a,
                                                                                                                initial_states_fw= initial_states_fw, 
                                                                                                                initial_states_bw= initial_states_bw, 
                                                                                                                dtype= tf.float32, 
                                                                                                                sequence_length= None,
                                                                                                                parallel_iterations= None, 
                                                                                                                time_major= self.time_major, 
                                                                                                                scope= None, 
                                                                                                                swap_memory= False)
            #}

            elif self.bidirectional_mode.upper() == "UNSTACK_BIDIRECTIONAL".upper():
            #{
                initial_states_fw = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
                initial_states_bw = cell_bw.zero_state(self.batch_size, dtype=tf.float32)

                a, (bi_encoder_state_fw, bi_encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw= cell_fw, 
                                                                                                cell_bw= cell_bw,
                                                                                                inputs= a,
                                                                                                initial_state_fw= initial_states_fw, 
                                                                                                initial_state_bw= initial_states_bw,
                                                                                                dtype= tf.float32, 
                                                                                                sequence_length= None,
                                                                                                parallel_iterations= None, 
                                                                                                time_major= self.time_major, 
                                                                                                scope= None, 
                                                                                                swap_memory= False)

                a = tf.concat(a, -1)
            #}

            if self.bidirectional_mapping.upper() == "double_units".upper():
            #{                
                if self.rnn_cell_type.upper() == "gru".upper():
                    # GRU
                    stateH = tf.concat((bi_encoder_state_fw, bi_encoder_state_bw), -1)

                    encoder_state += tuple(stateH)

                elif self.rnn_cell_type.upper() == "lstm".upper():
                    # LSTM
                    stateH = tf.concat((bi_encoder_state_fw[0][0], bi_encoder_state_bw[0][0]), 1)
                    stateC = tf.concat((bi_encoder_state_fw[0][1], bi_encoder_state_bw[0][1]), 1)

                    encoder_state += (tf.contrib.rnn.LSTMStateTuple(c=stateC, h=stateH),)
            #}

            else:
                encoder_state += bi_encoder_state_fw
                encoder_state += bi_encoder_state_bw            
        #}

        #===============================================================================================================================================================================

        #UNI-Directional RNN Cells            
        if self.uni_encoder_num_layers > 0:
        #{
            printb("[ENCODER] (UNI-Directional Trainable)", printable= self.debug_mode)

            initial_state = self.rnn_train.zero_state(self.batch_size, dtype=tf.float32)

            a, uni_encoder_state  = tf.nn.dynamic_rnn(  cell= cell,
                                                        inputs= a,
                                                        sequence_length=None,
                                                        initial_state= initial_state,
                                                        dtype=tf.float32,
                                                        parallel_iterations=None,
                                                        swap_memory=False,
                                                        time_major= self.time_major,
                                                        scope=None)

            encoder_state += uni_encoder_state
        #}       

        #===============================================================================================================================================================================

        output = a
        encoder_state = tuple(encoder_state)
        
        #print("#encoder_state:\n",encoder_state)
        #print()
        
        #printb("[ENCODER] (output) :{}".format(output.shape), printable=self.debug_mode)
        #printb("[ENCODER] (state)  :{}".format( hidden[0].shape if isinstance(hidden, tuple) else hidden.shape), printable=self.debug_mode)

        printb("--------------------------------------(ENCODER END)--------------------------------------\n", printable=self.debug_mode)

        return output, encoder_state
    #}


    
    # Define the initial state
    def initialize_hidden_state(self):  
    #{      
        if self.rnn_cell_type.upper() == "lstm".upper():
            init_hidden_state = list()

            for i in range((self.bi_encoder_num_layers + self.uni_encoder_num_layers)):
                init_hidden_state.append( tf.nn.rnn_cell.LSTMStateTuple( (tf.zeros((self.batch_size, self.enc_units)),), (tf.zeros((self.batch_size, self.enc_units)),) ))

            return init_hidden_state


        elif self.rnn_cell_type.upper() == "gru".upper():
            init_hidden_state =  ( self.bi_encoder_num_layers + self.uni_encoder_num_layers ) * (tf.zeros((self.batch_size, self.enc_units)),)
            return [init_hidden_state]
    #}
#}