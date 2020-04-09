import keras as k
import tensorflow as tf

class RNN():
#{    
    """
    def __init__(self):
        super(RNN, self).__init__()
    """

    # Build the multi-layer RNN cells
    def rnn_nn(self, units, rnn_cell_type, num_layers, input_dropout=0.0,  output_dropout=0.0, state_dropout=0.0, variational_rec=False):
    #{
        if isinstance(input_dropout, list) and isinstance(output_dropout, list) and isinstance(state_dropout, list):
        #{
            if (len(input_dropout) < num_layers) and (len(output_dropout) < num_layers) and (len(state_dropout) < num_layers):
                input_dropout  = input_dropout * num_layers  
                output_dropout = output_dropout * num_layers
                state_dropout  = state_dropout * num_layers         

            return tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(  (tf.nn.rnn_cell.LSTMCell(units) if(rnn_cell_type.upper() == "LSTM") else (tf.nn.rnn_cell.GRUCell(units) if(rnn_cell_type.upper() == "GRU") else tf.nn.rnn_cell.RNNCell(units))),
                                                                                variational_recurrent= variational_rec, 
                                                                                input_keep_prob  = (1- input_dropout[l]), 
                                                                                output_keep_prob = (1- output_dropout[l]), 
                                                                                state_keep_prob  = (1- state_dropout[l]),
                                                                                dtype= tf.float32
                                                                            )
                                                for l in range(num_layers)]
                                            )
        #}                        
        else:
        #{
            return tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(  (tf.nn.rnn_cell.LSTMCell(units) if(rnn_cell_type.upper() == "LSTM") else (tf.nn.rnn_cell.GRUCell(units) if(rnn_cell_type.upper() == "GRU") else tf.nn.rnn_cell.RNNCell(units))),
                                                                                variational_recurrent= variational_rec, 
                                                                                input_keep_prob  = (1- input_dropout), 
                                                                                output_keep_prob = (1- output_dropout), 
                                                                                state_keep_prob  = (1- state_dropout),
                                                                                dtype= tf.float32
                                                                            )
                                                for _ in range(num_layers)]
                                            )
        #}
    #}

#...........................................................................................................................................................

    def rnn_nn_cell(self, 
                    units, 
                    rnn_cell_type,
                    learn_mode= None,
                    activation=None, 
                    reuse=None, 
                    kernel_initializer=None, 
                    bias_initializer=None, 
                    name=None, 
                    dtype=None,
                    input_dropout= 0.0,  
                    output_dropout= 0.0, 
                    state_dropout= 0.0, 
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
                    residual_connection=False,              
                    device_str=None):
    #{

        if rnn_cell_type.upper() == "GRU":
            single_cell = tf.nn.rnn_cell.GRUCell(   num_units= units, 
                                                    activation= activation, 
                                                    reuse= reuse, 
                                                    kernel_initializer= kernel_initializer, 
                                                    bias_initializer= bias_initializer, 
                                                    name= name, 
                                                    dtype= dtype
                                                )

        elif rnn_cell_type.upper() == "LSTM":
            single_cell = tf.nn.rnn_cell.LSTMCell(  num_units= units, 
                                                    use_peepholes= False, 
                                                    cell_clip= cell_clip, 
                                                    initializer= initializer, 
                                                    num_proj= num_proj,
                                                    proj_clip= proj_clip, 
                                                    num_unit_shards= num_unit_shards,
                                                    num_proj_shards= num_proj_shards,
                                                    forget_bias= forget_bias, 
                                                    state_is_tuple= state_is_tuple, 
                                                    activation= activation, 
                                                    reuse= reuse, 
                                                    name= name, 
                                                    dtype= dtype
                                                )   
        else:
            raise ValueError("Unknown cell type %s!" % rnn_cell_type)



        # Device Wrapper
        single_cell_droppable = tf.nn.rnn_cell.DropoutWrapper(cell= single_cell,
                                                            variational_recurrent= variational_recurrent, 
                                                            input_keep_prob  = (1.0 - input_dropout), 
                                                            output_keep_prob = (1.0 - output_dropout), 
                                                            state_keep_prob  = (1.0 - state_dropout),
                                                            dtype= dtype
                                                            )
        
        single_cell_nondroppable = tf.nn.rnn_cell.DropoutWrapper(cell= single_cell,
                                                                variational_recurrent= variational_recurrent, 
                                                                input_keep_prob  = 1.0, 
                                                                output_keep_prob = 1.0, 
                                                                state_keep_prob  = 1.0,
                                                                dtype= dtype
                                                                )

        # Residual Wrapper
        if residual_connection:
            single_cell_droppable     = tf.contrib.rnn.ResidualWrapper(single_cell_droppable)
            single_cell_nondroppable  = tf.contrib.rnn.ResidualWrapper(single_cell_nondroppable) 
            


        # Device Wrapper
        if device_str:
            single_cell_droppable    = tf.contrib.rnn.DeviceWrapper(single_cell_droppable, device_str)
            single_cell_nondroppable = tf.contrib.rnn.DeviceWrapper(single_cell_nondroppable, device_str)
        


        return  single_cell_droppable, single_cell_nondroppable 
    #}



#...........................................................................................................................................................

    def rnn_nn_dyna_layers( self, 
                            units, 
                            rnn_cell_type,                             
                            num_layers,
                            num_residual_layers,
                            activation=None, 
                            reuse=None, 
                            kernel_initializer=None, 
                            bias_initializer=None, 
                            name=None, 
                            dtype=None,
                            learn_mode= None,
                            input_dropout= 0.0,  
                            output_dropout= 0.0, 
                            state_dropout= 0.0, 
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
                            return_MultiRNNCell= True,
                            device_str= None):
    #{
        rnn_layers_train = []
        rnn_layers_eval  = []

        if isinstance(input_dropout, list) and isinstance(output_dropout, list) and isinstance(state_dropout, list):
        #{
            if (len(input_dropout) < num_layers) and (len(output_dropout) < num_layers) and (len(state_dropout) < num_layers):
                input_dropout  = input_dropout * num_layers  
                output_dropout = output_dropout * num_layers
                state_dropout  = state_dropout * num_layers   

            for l in range(num_layers):
            #{
                single_cell_droppable, single_cell_nondroppable = self.rnn_nn_cell(units= units, 
                                                                    rnn_cell_type= rnn_cell_type,
                                                                    activation= activation, 
                                                                    reuse= reuse, 
                                                                    kernel_initializer= kernel_initializer, 
                                                                    bias_initializer= bias_initializer, 
                                                                    name= name, 
                                                                    dtype= dtype,
                                                                    input_dropout= input_dropout[l],  
                                                                    output_dropout= output_dropout[l], 
                                                                    state_dropout= state_dropout[l], 
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
                                                                    residual_connection= (l >= (num_layers - num_residual_layers)),             
                                                                    device_str= device_str)
                rnn_layers_train.append(single_cell_droppable)
                rnn_layers_eval.append(single_cell_nondroppable)
            #}		
        #}

        else:
        #{
            for l in range(num_layers):
            #{

                single_cell_droppable, single_cell_nondroppable = self.rnn_nn_cell(units= units, 
                                                                    rnn_cell_type= rnn_cell_type,
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
                                                                    state_is_tuple= state_is_tuple,
                                                                    residual_connection= (l >= (num_layers - num_residual_layers)),             
                                                                    device_str= device_str) 

                rnn_layers_train.append(single_cell_droppable)
                rnn_layers_eval.append(single_cell_nondroppable)
            #}
        #}	

        if return_MultiRNNCell:
            return tf.contrib.rnn.MultiRNNCell(rnn_layers_train), tf.contrib.rnn.MultiRNNCell(rnn_layers_eval)
        else:
            return rnn_layers_train, rnn_layers_eval
    #}
    
#...........................................................................................................................................................

    def bidirectional_rnn_nn_dyna_layers(self, 
                                        units, 
                                        rnn_cell_type,
                                        num_layers,
                                        num_residual_layers,
                                        activation=None, 
                                        reuse=None, 
                                        kernel_initializer=None, 
                                        bias_initializer=None, 
                                        name=None, 
                                        dtype=None,
                                        learn_mode= None, 
                                        input_dropout= 0.0,  
                                        output_dropout= 0.0, 
                                        state_dropout= 0.0, 
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
										bidirectional_mode= "STACK_BIDIRECTIONAL",
                                        device_str= None):
    #{
        rnn_layers_fw_train = []
        rnn_layers_fw_eval  = []
        
        rnn_layers_bw_train = []
        rnn_layers_bw_eval  = []
        
        if isinstance(input_dropout, list) and isinstance(output_dropout, list) and isinstance(state_dropout, list):
        #{
            if (len(input_dropout) < num_layers) and (len(output_dropout) < num_layers) and (len(state_dropout) < num_layers):
                input_dropout  = input_dropout * num_layers  
                output_dropout = output_dropout * num_layers
                state_dropout  = state_dropout * num_layers

            for l in range(num_layers):
            #{
                # By giving a different variable scope to each layer, I've ensured that
                # the weights are not shared among the layers. If you want to share the
                # weights, you can do that by giving variable_scope as "encoder" but do
                # make sure first that reuse is set to tf.AUTO_REUSE

                single_cell_droppable_fw, single_cell_fw = self.rnn_nn_cell( units= units, 
                                                                            rnn_cell_type= rnn_cell_type,
                                                                            activation= activation, 
                                                                            reuse= reuse, 
                                                                            kernel_initializer= kernel_initializer, 
                                                                            bias_initializer= bias_initializer, 
                                                                            name= name, 
                                                                            dtype= dtype,
                                                                            input_dropout= input_dropout[l],  
                                                                            output_dropout= output_dropout[l], 
                                                                            state_dropout= state_dropout[l], 
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
                                                                            residual_connection= (l >= num_layers - num_residual_layers),       
                                                                            device_str= device_str)
                
                single_cell_droppable_bw, single_cell_bw = self.rnn_nn_cell( units= units, 
                                                                            rnn_cell_type= rnn_cell_type,
                                                                            activation= activation, 
                                                                            reuse= reuse, 
                                                                            kernel_initializer= kernel_initializer, 
                                                                            bias_initializer= bias_initializer, 
                                                                            name= name, 
                                                                            dtype= dtype,
                                                                            input_dropout= input_dropout[l],  
                                                                            output_dropout= output_dropout[l], 
                                                                            state_dropout= state_dropout[l], 
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
                                                                            residual_connection= (l >= num_layers - num_residual_layers),          
                                                                            device_str= device_str)                

                rnn_layers_fw_train.append(single_cell_droppable_fw)
                rnn_layers_fw_eval.append(single_cell_fw)                

                rnn_layers_bw_train.append(single_cell_droppable_bw) 
                rnn_layers_bw_eval.append(single_cell_bw)                               
            #}               
        #}

        else:
        #{
            for l in range(num_layers):
            #{
                single_cell_droppable_fw, single_cell_fw = self.rnn_nn_cell( units= units, 
                                                                            rnn_cell_type= rnn_cell_type,
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
                                                                            state_is_tuple= state_is_tuple,
                                                                            residual_connection= (l >= num_layers - num_residual_layers),
                                                                            device_str= device_str)
                
                single_cell_droppable_bw, single_cell_bw = self.rnn_nn_cell( units= units, 
                                                                            rnn_cell_type= rnn_cell_type,
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
                                                                            state_is_tuple= state_is_tuple,
                                                                            residual_connection= (l >= num_layers - num_residual_layers),          
                                                                            device_str= device_str)                

                rnn_layers_fw_train.append(single_cell_droppable_fw)
                rnn_layers_fw_eval.append(single_cell_fw)                

                rnn_layers_bw_train.append(single_cell_droppable_bw) 
                rnn_layers_bw_eval.append(single_cell_bw)                               
            #}            
        #}

        if bidirectional_mode.upper() == "STACK_BIDIRECTIONAL".upper():
		
            return rnn_layers_fw_train, rnn_layers_fw_eval, rnn_layers_bw_train, rnn_layers_bw_eval

        else:
			
            rnn_layers_fw_train = tf.contrib.rnn.MultiRNNCell(rnn_layers_fw_train)
            rnn_layers_fw_eval  = tf.contrib.rnn.MultiRNNCell(rnn_layers_fw_eval)
            rnn_layers_bw_train = tf.contrib.rnn.MultiRNNCell(rnn_layers_bw_train)
            rnn_layers_bw_eval  = tf.contrib.rnn.MultiRNNCell(rnn_layers_bw_eval)

            return rnn_layers_fw_train, rnn_layers_fw_eval, rnn_layers_bw_train, rnn_layers_bw_eval
    #}


        """
        https://stackoverflow.com/questions/49242266/difference-between-bidirectional-dynamic-rnn-and-stack-bidirectional-dynamic-rnn
        https://stackoverflow.com/questions/46189318/how-to-use-multilayered-bidirectional-lstm-in-tensorflow
        
        output = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(rnn_layers_fw_train, 
                                                                rnn_layers_bw_train, 
                                                                INPUT,
                                                                sequence_length=LENGTHS, 
                                                                dtype=tf.float32)



        Concat the forward and backward outputs
        output = tf.concat(outputs,2)
        """	


#...........................................................................................................................................................

    def rnn_keras_cell( self,
                        units,
                        rnn_cell_type,                                           
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
                        kernel_constraint=None, 
                        recurrent_constraint=None, 
                        bias_constraint=None, 
                        dropout=0.0, 
                        recurrent_dropout=0.0, 
                        implementation=1,
                        reset_after=False
                        ):
    #{

        if rnn_cell_type.upper() == "LSTM":
            single_cell = tf.keras.layers.LSTMCell( units= units, 
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
                                                    kernel_constraint= kernel_constraint, 
                                                    recurrent_constraint= recurrent_constraint, 
                                                    bias_constraint= bias_constraint, 
                                                    dropout= dropout, 
                                                    recurrent_dropout= recurrent_dropout, 
                                                    implementation= implementation
                                                  ) 
        elif rnn_cell_type.upper() == "GRU":
            single_cell = tf.keras.layers.GRUCell(  units= units, 
                                                    activation= activation,
                                                    recurrent_activation= recurrent_activation, 
                                                    use_bias= use_bias, 
                                                    kernel_initializer=kernel_initializer, 
                                                    recurrent_initializer= recurrent_initializer, 
                                                    bias_initializer= bias_initializer, 
                                                    kernel_regularizer= kernel_regularizer, 
                                                    recurrent_regularizer= recurrent_regularizer, 
                                                    bias_regularizer= bias_regularizer,
                                                    kernel_constraint= kernel_constraint, 
                                                    recurrent_constraint= recurrent_constraint, 
                                                    bias_constraint= bias_constraint, 
                                                    dropout= dropout, 
                                                    recurrent_dropout= recurrent_dropout, 
                                                    implementation= implementation, 
                                                    reset_after= reset_after
                                                 ) 
        else:
            raise ValueError("Unknown cell type %s!" % rnn_cell_type)               

        return single_cell
    #}

#...........................................................................................................................................................

    def rnn_keras_dyna_layers(  self,
                                units,
                                rnn_cell_type,
                                num_layers,
                                num_residual_layers,                                           
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
                                kernel_constraint=None, 
                                recurrent_constraint=None, 
                                bias_constraint=None, 
                                dropout=0.0, 
                                recurrent_dropout=0.0, 
                                implementation=1,
                                reset_after=False,
                                return_sequences= True,
                                return_state= True,
                                go_backwards= False,
                                stateful= False,
                                unroll= False,
                                layerWrapper_normalize_input= False,
                                layerWrapper_normalize_output= False,
                                layerWrapper_input_dropout= 0.0,
                                layerWrapper_output_dropout= 0.0
                            ):
        """
        rnn_layers= [
                tf.keras.layers.GRUCell(units),
                tf.keras.layers.GRUCell(units),
                tf.keras.layers.GRUCell(units)
            ]
        """
        #self.dropout = tf.Variable(RNNARg.dropout)

        rnn_layers= []
        for l in range(num_layers):
        #{        
            single_cell= self.rnn_keras_cell(units= units,
                                            rnn_cell_type= rnn_cell_type,
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
                                            kernel_constraint= kernel_constraint, 
                                            recurrent_constraint= recurrent_constraint, 
                                            bias_constraint= bias_constraint, 
                                            dropout= dropout, 
                                            recurrent_dropout= recurrent_dropout, 
                                            implementation= implementation,
                                            reset_after= reset_after
                                            ) 
            
            single_cell= RNNCellWrapper(cell= single_cell, 
                                        input_dropout= layerWrapper_input_dropout, 
                                        output_dropout= layerWrapper_output_dropout, 
                                        residual_connection= (l >= num_layers - num_residual_layers))
            
            #Xsingle_cell= ResidualWrapper(single_cell, True)  

            rnn_layers.append(single_cell)
        #}                
 
        #<----
        """
        Xstack_cells = tf.keras.layers.RNN( stack_cells, 
                                            return_sequences= return_sequences, 
                                            return_state= return_state,
                                            go_backwards= go_backwards,
                                            stateful= stateful,
                                            unroll= unroll
                                        )
        """
        return tf.keras.layers.StackedRNNCells(rnn_layers)
    #}        
#...........................................................................................................................................................

    def bidirectional_rnn_keras_dyna_layers(self,
                                            units,
                                            rnn_cell_type,
                                            num_layers, 
                                            merge_mode,
                                            backward_layer,                                           
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
                                            kernel_constraint=None, 
                                            recurrent_constraint=None, 
                                            bias_constraint=None, 
                                            dropout=0.0, 
                                            recurrent_dropout=0.0, 
                                            implementation=1,
                                            reset_after=False,
                                            return_sequences= True,
                                            return_state= True,
                                            go_backwards= False,
                                            stateful= False,
                                            unroll= False,                                            
                                            layerWrapper_normalize_input= False,
                                            layerWrapper_normalize_output= False,
                                            layerWrapper_input_dropout= 0.0,
                                            layerWrapper_output_dropout= 0.0
                                            ):
        rnn_layers= []
        for l in range(num_layers):
        #{        
            
            single_cell= self.rnn_keras_cell(units= units,
                                            rnn_cell_type= rnn_cell_type,
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
                                            kernel_constraint= kernel_constraint, 
                                            recurrent_constraint= recurrent_constraint, 
                                            bias_constraint= bias_constraint, 
                                            dropout= dropout, 
                                            recurrent_dropout= recurrent_dropout, 
                                            implementation= implementation,
                                            reset_after= reset_after,
                                            layerWrapper_normalize_input= layerWrapper_normalize_input,
                                            layerWrapper_normalize_output= layerWrapper_normalize_output,
                                            layerWrapper_input_dropout= layerWrapper_input_dropout,
                                            layerWrapper_output_dropout= layerWrapper_output_dropout,
                                            layerWrapper_residual_connection= (l >= num_layers - num_residual_layers)                                             
                                            ) 
            
            rnn_layers.append(single_cell)
        #}
                
        stack_cells = tf.keras.layers.StackedRNNCells(rnn_layers)
        
        return tf.keras.layers.Bidirectional(tf.keras.layers.RNN(stack_cells, 
                                                                return_sequences= return_sequences, 
                                                                return_state= return_state,
                                                                go_backwards= go_backwards,
                                                                stateful= stateful,
                                                                unroll= unroll),                                                                
                                            merge_mode= merge_mode,
                                            backward_layer= merge_mode)
    #} 

#...........................................................................................................................................................

    def rnn_keras(  self, 
                    units, 
                    rnn_cell_type,
                    cell_mode= "NATIVE",
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
                    dropout=0.0, 
                    recurrent_dropout=0.0, 
                    implementation=1, 
                    return_sequences= True,
                    return_state= True, 
                    go_backwards=False, 
                    stateful=False, 
                    unroll=False,
                    reset_after=False
                ):
    #{   
        # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
        # the code automatically does that.
        if( tf.test.is_gpu_available() and  cell_mode.upper() == "CuDNN".upper() ):
        #{
            if rnn_cell_type.upper() == "GRU":
                print(">>>>> Keras CuDNN GRU")
                return tf.keras.layers.CuDNNGRU(units= units, 
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
                                                return_sequences= return_sequences, 
                                                return_state= return_state, 
                                                go_backwards= go_backwards, 
                                                stateful= stateful
                                                ) 
                
            elif rnn_cell_type.upper() == "LSTM":   
                print(">>>>> Keras CuDNN LSTM")
                return tf.keras.layers.CuDNNLSTM(units= units, 
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
                                                return_sequences= return_sequences,
                                                return_state= return_state, 
                                                go_backwards= go_backwards, 
                                                stateful= stateful,
                                                )                                
        #}
        elif cell_mode.upper() == "NATIVE".upper(): 
        #{
            if rnn_cell_type.upper() == "GRU":
                print(">>>>> Keras Native GRU")
                return tf.keras.layers.GRU( units= units, 
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
                                            dropout= dropout, 
                                            recurrent_dropout= recurrent_dropout, 
                                            implementation= implementation, 
                                            return_sequences= return_sequences, 
                                            return_state= return_state, 
                                            go_backwards= go_backwards, 
                                            stateful= stateful, 
                                            unroll= unroll, 
                                            reset_after= reset_after
                                            )
                
            elif rnn_cell_type.upper() == "LSTM":   
                print(">>>>> Keras Native LSTM")
                return tf.keras.layers.LSTM(units= units,
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
                                            dropout= dropout,
                                            recurrent_dropout= recurrent_dropout, 
                                            implementation= implementation, 
                                            return_sequences= return_sequences, 
                                            return_state= return_state, 
                                            go_backwards= go_backwards, 
                                            stateful= stateful, 
                                            unroll= unroll
                                            )
        #}

        else:
            raise ValueError("Unknown cell mode %s!" % cell_mode)
    #}

#...........................................................................................................................................................

    def rnn_keras_stack(self, 
                        units, 
                        num_layers,
                        rnn_cell_type,
                        cell_mode= "NATIVE",
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
                        dropout=0.0, 
                        recurrent_dropout=0.0, 
                        implementation=1, 
                        return_sequences= True,
                        return_state= True, 
                        go_backwards=False, 
                        stateful=False, 
                        unroll=False,
                        reset_after=False
                    ):
    #{

        rnn_layers=[]  

        for l in range(num_layers):
            rnn_layers.append( self.rnn_keras(  units, 
                                                rnn_cell_type,
                                                cell_mode,
                                                activation,
                                                recurrent_activation,
                                                use_bias, 
                                                kernel_initializer, 
                                                recurrent_initializer, 
                                                bias_initializer, 
                                                unit_forget_bias, 
                                                kernel_regularizer,
                                                recurrent_regularizer, 
                                                bias_regularizer, 
                                                activity_regularizer, 
                                                kernel_constraint,
                                                recurrent_constraint, 
                                                bias_constraint, 
                                                dropout, 
                                                recurrent_dropout, 
                                                implementation, 
                                                return_sequences,
                                                return_state, 
                                                go_backwards, 
                                                stateful, 
                                                unroll,
                                                reset_after
                                                ))

        return rnn_layers
    #}

#--------------------------------------------------------------------------------------------
    
    def rnn_keras_layers(self, 
                        units, 
                        num_layers,
                        rnn_cell_type,
                        cell_mode,
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
                        dropout=0.0, 
                        recurrent_dropout=0.0, 
                        implementation=1, 
                        return_sequences= True,
                        return_state= True, 
                        go_backwards=False, 
                        stateful=False, 
                        unroll=False,
                        reset_after=False
                    ):
    #{

        return rnn_keras_layers_Wrapper(self.rnn_keras_stack(units, 
                                                            num_layers,
                                                            rnn_cell_type,
                                                            cell_mode,
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
                                                            dropout= dropout, 
                                                            recurrent_dropout= recurrent_dropout, 
                                                            implementation= implementation, 
                                                            return_sequences= return_sequences,
                                                            return_state= return_state, 
                                                            go_backwards= go_backwards, 
                                                            stateful= stateful, 
                                                            unroll= unroll,
                                                            reset_after= reset_after
                                                            ))
    #}
   
#--------------------------------------------------------------------------------------------
   
    def bidirectional_rnn_keras_stack( self, 
                                        units,
                                        num_layers,
                                        rnn_cell_type,
                                        cell_mode,
                                        merge_mode,
                                        backward_layer,
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
                                        dropout=0.0, 
                                        recurrent_dropout=0.0, 
                                        implementation=1, 
                                        return_sequences= True,
                                        return_state= True, 
                                        go_backwards=False, 
                                        stateful=False, 
                                        unroll=False,
                                        reset_after=False
                                        ):
    #{
        rnn_layers=[]  

        for l in range(num_layers):
            rnn_layers.append( tf.keras.layers.Bidirectional(self.rnn_keras(units, 
                                                                            rnn_cell_type,
                                                                            cell_mode,
                                                                            activation,
                                                                            recurrent_activation,
                                                                            use_bias, 
                                                                            kernel_initializer, 
                                                                            recurrent_initializer, 
                                                                            bias_initializer, 
                                                                            unit_forget_bias, 
                                                                            kernel_regularizer,
                                                                            recurrent_regularizer, 
                                                                            bias_regularizer, 
                                                                            activity_regularizer, 
                                                                            kernel_constraint,
                                                                            recurrent_constraint, 
                                                                            bias_constraint, 
                                                                            dropout, 
                                                                            recurrent_dropout, 
                                                                            implementation, 
                                                                            return_sequences,
                                                                            return_state, 
                                                                            go_backwards, 
                                                                            stateful, 
                                                                            unroll,
                                                                            reset_after),
                                                                merge_mode= merge_mode,
                                                                backward_layer= merge_mode))
        return rnn_layers
    #}    

#...........................................................................................................................................................    

    def bidirectional_rnn_keras_layers( self, 
                                        units, 
                                        num_layers,
                                        rnn_cell_type,
                                        cell_mode,
                                        merge_mode,
                                        backward_layer,
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
                                        dropout=0.0, 
                                        recurrent_dropout=0.0, 
                                        implementation=1, 
                                        return_sequences= True,
                                        return_state= True, 
                                        go_backwards=False, 
                                        stateful=False, 
                                        unroll=False,
                                        reset_after=False
                                    ):
    #{

        return rnn_keras_layers_Wrapper(self.bidirectional_rnn_keras_stack( units= units, 
                                                                            num_layers= num_layers,
                                                                            rnn_cell_type= rnn_cell_type,
                                                                            cell_mode= cell_mode,
                                                                            merge_mode= merge_mode,
                                                                            backward_layer= merge_mode,
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
                                                                            dropout= dropout, 
                                                                            recurrent_dropout= recurrent_dropout, 
                                                                            implementation= implementation, 
                                                                            return_sequences= return_sequences,
                                                                            return_state= return_state, 
                                                                            go_backwards= go_backwards, 
                                                                            stateful= stateful, 
                                                                            unroll= unroll,
                                                                            reset_after= reset_after))
    #}
   
#...........................................................................................................................................................


    def gru_keras(  self, 
                    units,
					mode="NATIVE",
                    activation='tanh', 
                    recurrent_activation='hard_sigmoid', 
                    use_bias=True, 
                    kernel_initializer='glorot_uniform', 
                    recurrent_initializer='orthogonal',
                    bias_initializer='zeros', 
                    kernel_regularizer=None, 
                    recurrent_regularizer=None, 
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    recurrent_constraint=None,
                    bias_constraint=None,
                    dropout=0.0, 
                    recurrent_dropout=0.0,
                    implementation=1, 
                    return_sequences=True,
                    return_state=True, 
                    go_backwards=False, 
                    stateful=False, 
                    unroll=False, 
                    reset_after=False
                ):
    #{   
        # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
        # the code automatically does that.
        if( tf.test.is_gpu_available() and  mode.upper() == "CuDNN".upper() ):
        #{
            print(">>>>> Keras CuDNN GRU")
            return tf.keras.layers.CuDNNGRU(units= units, 
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
                                            return_sequences= return_sequences, 
                                            return_state= return_state, 
                                            go_backwards= go_backwards, 
                                            stateful= stateful
                                            )       
        #}
        elif mode.upper() == "NATIVE".upper(): 
        #{
            print(">>>>> Keras Native GRU")
            return tf.keras.layers.GRU( units= units, 
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
                                        dropout= dropout, 
                                        recurrent_dropout= recurrent_dropout, 
                                        implementation= implementation, 
                                        return_sequences= return_sequences, 
                                        return_state= return_state, 
                                        go_backwards= go_backwards, 
                                        stateful= stateful, 
                                        unroll= unroll, 
                                        reset_after= reset_after
                                        )
            
        else:
            raise ValueError("Unknown cell Mode Type %s!" % mode)  

        #}
    #}
   
#...........................................................................................................................................................
    
    def lstm_keras( self, 
                    units, 
					mode="NATIVE",
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
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    implementation=1, 
                    return_sequences=True,
                    return_state=True,
                    go_backwards=False, 
                    stateful=False,
                    unroll=False
                    ):
    #{   
        # If you have a GPU, we recommend using CuDNNLSTM(provides a 3x speedup than LSTM)
        # the code automatically does that.
        if( tf.test.is_gpu_available() and  mode.upper() == "CuDNN".upper() ):
        #{
            print(">>>>> Keras CuDNN LSTM")
            return tf.keras.layers.CuDNNLSTM(units= units, 
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
                                            return_sequences= return_sequences,
                                            return_state= return_state, 
                                            go_backwards= go_backwards, 
                                            stateful= stateful,
                                            )  
        #}
        elif  mode.upper() == "NATIVE".upper():
        #{
            print(">>>>> Keras Native LSTM")
            return tf.keras.layers.LSTM(units= units,
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
                                        dropout= dropout,
                                        recurrent_dropout= recurrent_dropout, 
                                        implementation= implementation, 
                                        return_sequences= return_sequences, 
                                        return_state= return_state, 
                                        go_backwards= go_backwards, 
                                        stateful= stateful, 
                                        unroll= unroll
                                        )
            
        else:
            raise ValueError("Unknown cell Mode Type %s!" % mode)            
        #}
    #}   
#...........................................................................................................................................................    
    
    def bidirectional_lstm_keras(self, 
                                units,
                                mode="NATIVE",
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
                                dropout=0.0,
                                recurrent_dropout=0.0,
                                implementation=1, 
                                return_sequences=True,
                                return_state=True,
                                go_backwards=False, 
                                stateful=False,
                                unroll=False,
                                merge_mode= "concat",
                                backward_layer= None,
                                weights=None
                                ):                                            
    #{
        return tf.keras.layers.Bidirectional(self.lstm_keras(units= units,
															mode= mode,		
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
                                                            dropout= dropout,
                                                            recurrent_dropout= recurrent_dropout,
                                                            implementation= implementation, 
                                                            return_sequences= return_sequences,
                                                            return_state= return_state,
                                                            go_backwards= go_backwards, 
                                                            stateful= stateful,
                                                            unroll= unroll), 
                                                merge_mode= merge_mode,
                                                backward_layer= backward_layer,
                                                weights= weights)
    #}


#...........................................................................................................................................................

    def bidirectional_gru_keras(self, 
                                units,
                                mode="NATIVE",
                                activation='tanh', 
                                recurrent_activation='hard_sigmoid', 
                                use_bias=True, 
                                kernel_initializer='glorot_uniform', 
                                recurrent_initializer='orthogonal',
                                bias_initializer='zeros', 
                                kernel_regularizer=None, 
                                recurrent_regularizer=None, 
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                recurrent_constraint=None,
                                bias_constraint=None,
                                dropout=0.0, 
                                recurrent_dropout=0.0,
                                implementation=1, 
                                return_sequences=True,
                                return_state=True, 
                                go_backwards=False, 
                                stateful=False, 
                                unroll=False, 
                                reset_after=False,
                                merge_mode="concat",
                                backward_layer= None,
                                weights=None                                
                                ):

        return tf.keras.layers.Bidirectional( self.gru_keras(units= units,
                                                            mode= mode,	
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
                                                            dropout= dropout, 
                                                            recurrent_dropout= recurrent_dropout,
                                                            implementation= implementation, 
                                                            return_sequences= return_sequences,
                                                            return_state= return_state, 
                                                            go_backwards= go_backwards, 
                                                            stateful= stateful, 
                                                            unroll= unroll, 
                                                            reset_after= reset_after), 
                                                merge_mode= merge_mode,
                                                backward_layer= backward_layer,
                                                weights= weights)
    #}               

#...........................................................................................................................................................

#}