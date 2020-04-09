import keras as k
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from SL.lib.NMT_lib.lang_util import *

class NMT_Model(tf.keras.Model):
#{    
    def __init__(self, hparams, encoder_inst, decoder_inst, optimizer_inst, debug_mode=False):
    #{	
        super(NMT_Model, self).__init__()
        
        self.encoder = encoder_inst
        self.decoder = decoder_inst
        self.optimizer_inst = optimizer_inst
		
        self.batch_size  = hparams.batch_size

        self.training_type         = hparams.training_type
        self.sampling_probability  = hparams.sampling_probability  

        self.targ_lang_train       = hparams.targ_lang_train

        self.debug_mode = (True if debug_mode and hparams.debug_mode else False)
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


    def _compute_loss(self, real, logits):
    #{
        """Compute optimization loss."""

        batch_size = real.shape[0]
        max_time   = real.shape[1]

        crossent   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=logits)

        target_weights = tf.sequence_mask( [max_time]*real.shape[0] , max_time, dtype=logits.dtype)

        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)

        return loss
    #}


    def call(self, inputs, hidden, targets, loss, LEARN_MODE):    
    #{
        printb("-------------------------------------(NMT_MODEL STR)-------------------------------------\n", printable=self.debug_mode)
        #Encoder inputs.shape       =(BS, Tx)                   ex: inputs       : (16, 55)
        #Encoder hidden.shape       =(BS, units)                ex: hidden       : (16, 1024)
        #Encoder enc_output.shape   =(BS, Tx, units)            ex: enc_output   : (16, 55, 1024)
        #Encoder enc_hidden_H.shape =(BS, units)                ex: enc_hidden_H : (16, 1024)
        #Encoder enc_hidden_C.shape =(BS, units)                ex: enc_hidden_C : (16, 1024)
        #dec_input.shape            =(BS, 1)                    ex: enc_hidden_C : (16, 1)
        #start.shape                =(BS, Target_Vocab_Size)    ex: enc_hidden_C : (16, 1237)

        outs        = [] 
        enc_output, enc_hidden = self.encoder(inputs, hidden, LEARN_MODE)                                   #<**********************************************************************
        dec_hidden  = enc_hidden 

        printb("[NMT_MODEL] (inputs)       -> Encoder : {}".format(inputs.shape), printable=self.debug_mode)
        printb("[NMT_MODEL] (hidden)       -> Encoder : {}".format(hidden.shape), printable=self.debug_mode)
        printb("[NMT_MODEL] (LEARN_MODE)     -> Encoder : {}".format(LEARN_MODE), printable=self.debug_mode)
        printb("[NMT_MODEL] (enc_output)   <- Encoder : {}".format(enc_output.shape), printable=self.debug_mode)
        #printb("[NMT_MODEL] (enc_hidden_H) <- Encoder : {}".format(enc_hidden_H.shape), printable=self.debug_mode)
        #printb("[NMT_MODEL] (enc_hidden_C) <- Encoder : {}".format(enc_hidden_C.shape), printable=self.debug_mode)
        
        targ_lang_train = self.decoder.targ_lang_train

        if targ_lang_train != None:
            dec_input = tf.expand_dims([targ_lang_train.word2idx['<start>']] * self.batch_size, 1)            
            printb("[NMT_MODEL] (dec_input) [tf.expand_dims([targ_lang_train.word2idx['<start>']] * batch_size, 1)] : {}".format(dec_input.shape), printable=self.debug_mode)

            start = tf.one_hot( np.full((self.batch_size),targ_lang_train.word2idx['<start>']), len(targ_lang_train.word2idx))
            printb("[NMT_MODEL] (start)[tf.one_hot( np.full((batch_size),targ_lang_train.word2idx['<start>']), len(targ_lang_train.word2idx))] : {}".format(start.shape), printable=self.debug_mode)

            outs.append( start )
        else:
            dec_input = tf.expand_dims([0] * self.batch_size, 1)

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

            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output, LEARN_MODE, t)   #<**********************************************************************

            printb("[NMT_MODEL] (dec_input     -> Decoder : {}".format(dec_input.shape), printable=self.debug_mode)
            printb("[NMT_MODEL] (enc_output)   -> Decoder : {}".format(enc_output.shape), printable=self.debug_mode)
            printb("[NMT_MODEL] (LEARN_MODE)   -> Decoder : {}".format(LEARN_MODE), printable=self.debug_mode)
            printb("[NMT_MODEL] (predictions)  <- Decoder : {}".format(predictions.shape), printable=self.debug_mode)
            #printb("[NMT_MODEL] (dec_hidden_H) <- Decoder : {}".format(dec_hidden_H.shape), printable=self.debug_mode)
            #printb("[NMT_MODEL] (dec_hidden_C) <- Decoder : {}".format(dec_hidden_C.shape), printable=self.debug_mode)
		

            loss += self.loss_function(targets[:, t], predictions, self.optimizer_inst)                   #<**********************************************************************
            #loss += self._compute_loss(targets[:, t], predictions)

            outs.append(predictions)

            if LEARN_MODE == tf.contrib.learn.ModeKeys.TRAIN:
            #{
                #https://www.nextremer.com/blog/5556/
				
				#==================================================================================================================================================================================================

                if self.training_type.upper() == "free_running".upper():
                #{                    
                    # using free running
                    predicted_id = tf.argmax(predictions, axis=-1)          
                    dec_input    = tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))
                    printb("[NMT_MODEL] (dec_input) = [dec_input = tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))]".format(dec_input.shape), printable=self.debug_mode)					
                #}

                #==================================================================================================================================================================================================

                elif self.training_type.upper() == "teacher_forcing".upper():
                #{    
                    # using teacher forcing
                    dec_input = tf.expand_dims(targets[:, t], 1)		
                    printb("[NMT_MODEL] (dec_input) = [tf.expand_dims(targets[:, t], 1)]".format(dec_input.shape), printable=self.debug_mode)					
                #}

                #==================================================================================================================================================================================================

                elif self.training_type.upper() == "SEMI_TEACHER_FORCING".upper():
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
                    a = self.sampling_probability					
                    Y_t_1  = a * Y_t_1
                    Yh_t_1 = (1-a) * Yh_t_1
                    
                    printb("[NMT_MODEL] (Y_t_1)  = [a * Y_t_1]".format(dec_input.shape), printable=self.debug_mode)	
                    printb("[NMT_MODEL] (Yh_t_1) = [(1-a) * Yh_t_1]".format(dec_input.shape), printable=self.debug_mode)	

                    dec_input = tf.concat(Y_t_1, Yh_t_1)
                    printb("[NMT_MODEL] (dec_input) = [tf.concat(Y_t_1, Yh_t_1)]".format(dec_input.shape), printable=self.debug_mode)
                #}

                #==================================================================================================================================================================================================

                elif self.training_type.upper() == "scheduled_sampling".upper():
                #{                    
                    do_sampling = random.random() < self.sampling_probability

                    if do_sampling:
                    #{
                        # using free running
                        predicted_id = tf.argmax(predictions, axis=-1)     
                        printb("[NMT_MODEL] (predicted_id) [tf.argmax(predictions, axis=-1) ]".format(predicted_id.shape), printable=self.debug_mode)
                        
                        dec_input    = tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))	
                        printb("[NMT_MODEL] (dec_input) [tf.convert_to_tensor(np.reshape(predicted_id, (len(predicted_id),1)))]".format(dec_input.shape), printable=self.debug_mode)
                    #}
                    else:
                    #{
                        # using teacher forcing
                        dec_input = tf.expand_dims(targets[:, t], 1)
                        printb("[NMT_MODEL] (dec_input) = [tf.expand_dims(targets[:, t], 1)]".format(dec_input.shape), printable=self.debug_mode)
                    #}                   
                #}

                #==================================================================================================================================================================================================

                else:
                #{
                    raise ValueError("Unknown LEARN_MODE %s!" % training_type)
                #}
            #}

            elif LEARN_MODE == tf.contrib.learn.ModeKeys.EVAL:
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