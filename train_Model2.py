from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import keras as k
import tensorflow as tf
import tensorflow.contrib.eager as tfe

#tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution()

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
#from nltk.translate.bleu_score import corpus_bleu
#from nltk.translate.gleu_score import corpus_gleu
from sklearn.model_selection import train_test_split

from .text_processing_util import *
from .lang_util import *
from .graph_plotting_util import *
from .RNN import *
from .BahdanauAttention import *
from .LuongsAttention import *
from .Encoder import *
from .Decoder import *
from .Optimizer import *
from .NMT_Model import *


"""
train_Model(enable= True,
            num_epochs= NUM_EPOCHS,
            batch_size= BATCH_SIZE,
            train_dataset= train_dataset,
            val_dataset= val_dataset,
            inp_lang_train= inp_lang_train,
            targ_lang_train= targ_lang_train,
            encoder_inst= encoder,
            decoder_inst= decoder,
            model_inst= NMT_MODEL,
            optimizer_inst= optimizer,
            checkpoint_inst= checkpoint,
            plotting_enable= PLOTTING_ENABLE,
            gradient_clipping_enabled= GRADIENT_CLIPPING_ENABLED,
            max_gradient_norm= MAX_GRADIENT_NORM,
            checkpoint_dir= CHECKPOINT_DIR,
            log_file= CHECKPOINT_DIR +"logger.log",
            config_file= CHECKPOINT_DIR +"config.txt",
            save_checkpoint= SAVE_CHECKPOINT,
            epoch_logger_mode= EPOCH_LOGGER_MODE,
            eval_validation_set= EVAL_VALIDATION_SET,
            eval_validation_set_epc= EVAL_VALIDATION_SET_EPC,
            data_partition= data_partition,
            model_desc= MODEL_DESC,
            kernel= KERNEL,
            progress_bar= PROGRESS_BAR
            )
"""


def data_generator(X, Y, step, batch_size):    
#{ 
    X = X[step:step+batch_size]
    Y = Y[step:step+batch_size]

    XY = list(zip(X, Y))

    random.shuffle( XY )

    X, Y = zip(*XY)

    return X, Y
#}	

def train_Model(enable,
                num_epochs,
                batch_size,
                train_dataset,
                val_dataset,
                inp_lang_train,
                targ_lang_train,				
                encoder_inst,
                decoder_inst,
                model_inst,
                optimizer_inst,
                checkpoint_inst,
                plotting_enable, 
                gradient_clipping_enabled, 
                max_gradient_norm,
                checkpoint_dir,
                log_file,
                config_file,
                save_checkpoint,
                epoch_logger_mode,
                eval_validation_set, 
                eval_validation_set_epc,
                data_partition=None,
                model_desc="",
                kernel=None,
                progress_bar=None
                ):
#{
    if enable:
    #{  
        train_X, train_Y = zip(*train_dataset)
        val_X,   val_Y   = zip(*val_dataset)

        print("train_X", train_X)
        print("train_Y", train_Y)


        LAST_MODEL_VER   = readConfiguration( config_file, 'DEFAULT', 'MODEL_VER' )         

        if LAST_MODEL_VER == None:  
            save_log(log_file, data_partition.dataset_split_info)
            save_log(log_file, data_partition.load_dataset_spliter_info)
            save_log(log_file, model_desc)			
            print()

        train_loss_avg_result_list     = read_serialize_object(checkpoint_dir+'/serialize/train_loss_avg_result_list.pkl', objectType=list)
        train_accuracy_result_list     = read_serialize_object(checkpoint_dir+'/serialize/train_accuracy_result_list.pkl', objectType=list)
        train_perplexity_result_list   = read_serialize_object(checkpoint_dir+'/serialize/train_perplexity_result_list.pkl', objectType=list)
        train_monitor_BLEU_score_list  = read_serialize_object(checkpoint_dir+'/serialize/train_monitor_BLEU_score_list.pkl', objectType=list)
        train_monitor_ROUGE_score_list = read_serialize_object(checkpoint_dir+'/serialize/train_monitor_ROUGE_score_list.pkl', objectType=list)

        val_loss_result_list           = read_serialize_object(checkpoint_dir+'/serialize/val_loss_result_list.pkl', objectType=list)
        val_accuracy_result_list       = read_serialize_object(checkpoint_dir+'/serialize/val_accuracy_result_list.pkl', objectType=list)
        val_perplexity_result_list     = read_serialize_object(checkpoint_dir+'/serialize/val_perplexity_result_list.pkl', objectType=list)
        val_monitor_BLEU_score_list    = read_serialize_object(checkpoint_dir+'/serialize/val_monitor_BLEU_score_list.pkl', objectType=list)
        val_monitor_ROUGE_score_list   = read_serialize_object(checkpoint_dir+'/serialize/val_monitor_ROUGE_score_list.pkl', objectType=list)

        diff_loss_result_list          = read_serialize_object(checkpoint_dir+'/serialize/diff_loss_result_list.pkl', objectType=list)

        MODEL_VER=1

        from sklearn.metrics import accuracy_score

        if(plotting_enable):
            simpleTrainingCurves  = SimpleTrainingCurves("cross-entropy", "accuracy")
            simpleTrainingCurves2 = SimpleTrainingCurves("BLEU-SCORE", "ROUGE-SCORE")

        if kernel.upper() == "KAGGLE".upper():
            LOG_HISTORY = read_file(log_file)  
            if LOG_HISTORY != None:
                save_log(log_file, LOG_HISTORY)

        LAST_MODEL_VER   = readConfiguration( config_file, 'DEFAULT', 'MODEL_VER' )         
        if LAST_MODEL_VER != None:
            MODEL_VER      = int(LAST_MODEL_VER)   + 1


        TOTAL_EPOCHS = readConfiguration( config_file, 'DEFAULT', 'TOTAL_EPOCHS')    
        if TOTAL_EPOCHS == None:
            TOTAL_EPOCHS = 0    


        BEST_TRAIN_LOSS_CONFIG = readConfiguration( config_file, 'DEFAULT', 'BEST_TRAIN_LOSS' )
        if BEST_TRAIN_LOSS_CONFIG == None:
            BEST_TRAIN_LOSS          =  float(100.000)
            BEST_EPOCH_TAINING_LOSS  = 1        
        else:
            BEST_TRAIN_LOSS          = BEST_TRAIN_LOSS_CONFIG[:BEST_TRAIN_LOSS_CONFIG.find("|")]
            BEST_EPOCH_TAINING_LOSS  = BEST_TRAIN_LOSS_CONFIG[BEST_TRAIN_LOSS_CONFIG.find("|")+1:]


        BEST_TRAIN_ACCURACY_CONFIG = readConfiguration( config_file, 'DEFAULT', 'BEST_TRAIN_ACCURACY' )
        if BEST_TRAIN_ACCURACY_CONFIG == None:
            BEST_TRAIN_ACCURACY = float(0.0)
            BEST_EPOCH_TAINING_ACCURACY = 1
        else:
            BEST_TRAIN_ACCURACY         = BEST_TRAIN_ACCURACY_CONFIG[:BEST_TRAIN_ACCURACY_CONFIG.find("|")]
            BEST_EPOCH_TAINING_ACCURACY = BEST_TRAIN_ACCURACY_CONFIG[BEST_TRAIN_ACCURACY_CONFIG.find("|")+1:]


        BEST_VALIDATION_LOSS_CONFIG = readConfiguration( config_file, 'DEFAULT', 'BEST_VALIDATION_LOSS' )     
        if BEST_VALIDATION_LOSS_CONFIG == None:
            BEST_VALIDATION_LOSS  = float(100.000)
            BEST_EPOCH_VALID_LOSS = 1
        else:
            BEST_VALIDATION_LOSS  = BEST_VALIDATION_LOSS_CONFIG[:BEST_VALIDATION_LOSS_CONFIG.find("|")]
            BEST_EPOCH_VALID_LOSS = BEST_VALIDATION_LOSS_CONFIG[BEST_VALIDATION_LOSS_CONFIG.find("|")+1:]


        BEST_VALIDATION_ACCURACY_CONFIG = readConfiguration( config_file, 'DEFAULT', 'BEST_VALIDATION_ACCURACY' )    
        if BEST_VALIDATION_ACCURACY_CONFIG == None:
            BEST_VALIDATION_ACCURACY = float(0.0)
            BEST_EPOCH_VALID_ACCURACY   = 1
        else:
            BEST_VALIDATION_ACCURACY  = BEST_VALIDATION_ACCURACY_CONFIG[:BEST_VALIDATION_ACCURACY_CONFIG.find("|")]
            BEST_EPOCH_VALID_ACCURACY = BEST_VALIDATION_ACCURACY_CONFIG[BEST_VALIDATION_ACCURACY_CONFIG.find("|")+1:]


        BEST_VALIDATION_BLEU_CONFIG = readConfiguration( config_file, 'DEFAULT', 'BEST_VALIDATION_BLEU' )    
        if BEST_VALIDATION_BLEU_CONFIG == None:
            BEST_VALIDATION_BLEU = float(0.0)
            BEST_EPOCH_VALID_BLEU   = 1
        else:
            BEST_VALIDATION_BLEU  = BEST_VALIDATION_BLEU_CONFIG[:BEST_VALIDATION_BLEU_CONFIG.find("|")]
            BEST_EPOCH_VALID_BLEU = BEST_VALIDATION_BLEU_CONFIG[BEST_VALIDATION_BLEU_CONFIG.find("|")+1:]


        BEST_VALIDATION_ROUGE_CONFIG = readConfiguration( config_file, 'DEFAULT', 'BEST_VALIDATION_BLEU' )    
        if BEST_VALIDATION_ROUGE_CONFIG == None:
            BEST_VALIDATION_ROUGE = float(0.0)
            BEST_EPOCH_VALID_ROUGE   = 1
        else:
            BEST_VALIDATION_ROUGE  = BEST_VALIDATION_ROUGE_CONFIG[:BEST_VALIDATION_ROUGE_CONFIG.find("|")]
            BEST_EPOCH_VALID_ROUGE = BEST_VALIDATION_ROUGE_CONFIG[BEST_VALIDATION_ROUGE_CONFIG.find("|")+1:]        


        BEST_DIFFERENCE_LOSS_CONFIG = readConfiguration( config_file, 'DEFAULT', 'BEST_DIFFERENCE_LOSS' )     
        if BEST_DIFFERENCE_LOSS_CONFIG == None:
            BEST_DIFFERENCE_LOSS  = float(100.000)
            BEST_EPOCH_DIFFERENCE_LOSS = 1
        else:
            BEST_DIFFERENCE_LOSS       = BEST_DIFFERENCE_LOSS_CONFIG[:BEST_DIFFERENCE_LOSS_CONFIG.find("|")]
            BEST_EPOCH_DIFFERENCE_LOSS = BEST_DIFFERENCE_LOSS_CONFIG[BEST_DIFFERENCE_LOSS_CONFIG.find("|")+1:]


        val_loss_result     = 0
        val_accuracy_result = 0
        diff_loss_result    = 0

        log_header = ("\n-----------------------------------------------| MODEL VERSION: %s |-----------------------------------------------\n"% (MODEL_VER))

        epoch_log  = log_header            
        total_log  = log_header 

        print(log_header, end="")

        for epoch in range(num_epochs):
        #{
            #actual_val_monitor_list, predicted_val_monitor_list = list(), list()                    #add        
            actual_val_monitor_fileName     = checkpoint_dir + "eval_val_monitor_actual.txt"        #add
            predicted_val_monitor_fileName  = checkpoint_dir + "eval_val_monitor_predicted.txt"     #add        
            actual_val_monitor_outputs, predicted_val_monitor_outputs  = list(), list()             #add  

            #actual_train_monitor_list, predicted_train_monitor_list = list(), list()                #add
            actual_train_monitor_fileName     = checkpoint_dir + "eval_train_monitor_actual.txt"    #add
            predicted_train_monitor_fileName  = checkpoint_dir + "eval_train_monitor_predicted.txt" #add        
            actual_train_monitor_outputs, predicted_train_monitor_outputs  = list(), list()         #add

            val_line_num, train_line_num = 1, 1


            TOTAL_EPOCHS      = int(TOTAL_EPOCHS) + 1

            start = time.time()

            batchCounter      = 0

            train_loss_avg    = tfe.metrics.Mean()
            valid_loss_avg    = tfe.metrics.Mean()

            train_accuracy    = tfe.metrics.Accuracy()
            valid_accuracy    = tfe.metrics.Accuracy()

            hidden = encoder_inst.initialize_hidden_state()

            train_total_loss  = 0
            valid_total_loss  = 0


            train_batch_loss  = []
            valid_batch_loss  = []

            print( "Epoch %d "%( epoch+1 ) , end="\n")

            #for (train_step, (train_input, train_label)) in enumerate(train_dataset):
            for train_step in range(0, len(list(train_dataset)), batch_size):
            #{
                if train_step+batch_size > len(list(train_dataset)):
                    break

                train_input, train_label = data_generator(train_X, train_Y, train_step, batch_size)

                batchCounter += (train_step+1)

                if progress_bar:
                #{
                    if(eval_validation_set and (epoch+1) % eval_validation_set_epc == 0):
                        print("[{}%]".format(int(( (train_step+1) / (TRAIN_N_BATCH + VALIDATION_N_BATCH) ) *100)), end="\r")
                    else:    
                        print("[{}%]".format(int(( (train_step+1) / (TRAIN_N_BATCH) ) *100)), end="\r")
                #}

                loss = 0

                with tf.GradientTape() as tape:            
                    outs, loss = model_inst(train_input, hidden, train_label,  train_label.shape[1], batch_size, encoder_inst, decoder_inst, loss, optimizer_inst, targ_lang_train, training=True)   #<---------------


                train_batch_loss  = (loss / int(train_label.shape[1]))
                train_total_loss += train_batch_loss

                variables = encoder_inst.variables + decoder_inst.variables        
                gradients = tape.gradient(loss, variables)        

                del loss

                if(gradient_clipping_enabled):
                    gradients, gradient_norm_summary = gradient_clip(gradients, max_gradient_norm= MAX_GRADIENT_NORM)

                optimizer_inst.optimizer.apply_gradients(zip(gradients, variables)) 

                # Track progress 
                train_loss_avg(train_batch_loss)  # add current batch loss 

                # compare predicted label to actual label  
                train_predictions  = tf.argmax(outs, axis=-1)
                del outs

                train_predictions  = np.array(train_predictions)
                train_predictions  = train_predictions.swapaxes(0,1) 
                train_accuracy(train_predictions, train_label)  

                if train_step % 100 == 0:
                #{
                    monitor = 'Epoch {} Batch {} Loss {:.4f}\n'.format(epoch + 1, train_step, train_batch_loss.numpy())
                    print(monitor, end="")   

                    epoch_log += monitor            
                    total_log += monitor        
                #}


                for (label, pre) in zip(train_label, train_predictions):#add
                #{    
                    actual_sent = tensor_to_sentence(label, targ_lang_train)
                    #print("actual_sent:", actual_sent)
                    #actual_train_monitor_outputs.append( actual_sent + "(" + str(train_line_num) + ")" )
                    actual_train_monitor_outputs.append( clean_tag(actual_sent) )
                    #actual_train_monitor_list.append( split(clean_tag(actual_sent)) )

                    predicted_sent = tensor_to_sentence(pre, targ_lang_train)                    
                    #predicted_train_monitor_outputs.append( predicted_sent + "(" + str(train_line_num) + ")" )
                    predicted_train_monitor_outputs.append( clean_tag(predicted_sent) )
                    #predicted_train_monitor_list.append( split(clean_tag(predicted_sent)) )                    

                    train_line_num += 1
                #}  

                # END TRAINING BATCH             
            #}    

            save_file_fromList(actual_train_monitor_fileName,    actual_train_monitor_outputs,    writeMode= "w") #add
            save_file_fromList(predicted_train_monitor_fileName, predicted_train_monitor_outputs, writeMode= "w") #add

            #train_monitor_BLEU_score  = calc_bleu(actual_train_monitor_fileName,  predicted_train_monitor_fileName, bpe_delimiter=" ") #add
            #train_monitor_ROUGE_score = calc_rouge(actual_train_monitor_fileName, predicted_train_monitor_fileName, bpe_delimiter=" ") #add
            train_monitor_BLEU_score  = 0.1
            train_monitor_ROUGE_score = 0.1

            if np.squeeze(train_loss_avg.result()) < float(BEST_TRAIN_LOSS):
                BEST_TRAIN_LOSS = np.squeeze(train_loss_avg.result()) 
                BEST_EPOCH_TAINING_LOSS = TOTAL_EPOCHS

            if np.squeeze(train_accuracy.result()) > float(BEST_TRAIN_ACCURACY):
                BEST_TRAIN_ACCURACY = np.squeeze(train_accuracy.result())
                BEST_EPOCH_TAINING_ACCURACY = TOTAL_EPOCHS 

            if(eval_validation_set and ( (epoch+1) % eval_validation_set_epc == 0 )):
            #{   
                #for (val_step, (val_input, val_label)) in enumerate(val_dataset):
                for val_step in range(0, len(list(val_dataset)), batch_size):
                #{
                    if val_step+batch_size > len(list(val_dataset)):
                        break

                    val_input, val_label = data_generator(val_X, val_Y, val_step, batch_size)

                    val_predictions, valid_batch_loss = model_inst(val_input, hidden, val_label, val_label.shape[1], batch_size, encoder_inst, decoder_inst, 0, optimizer_inst, targ_lang_train, training=False)  #<---------------

                    valid_batch_loss  = (valid_batch_loss / int(val_label.shape[1]))
                    valid_total_loss += valid_batch_loss        

                    # Track progress 
                    valid_loss_avg(valid_batch_loss)  # add current batch loss

                    val_predictions    = tf.argmax(val_predictions, axis=-1)
                    val_predictions    = np.array(val_predictions)
                    val_predictions    = val_predictions.swapaxes(0,1)  
                    valid_accuracy(val_predictions, val_label)

                    for (label, pre) in zip(val_label, val_predictions):#add
                    #{    
                        actual_sent = tensor_to_sentence(label, targ_lang_train)                    
                        #actual_val_monitor_outputs.append( actual_sent + "(" + str(val_line_num) + ")" )
                        actual_val_monitor_outputs.append( clean_tag(actual_sent) )                    
                        #actual_val_monitor_list.append( split( clean_tag(actual_sent)) )

                        predicted_sent= tensor_to_sentence(pre, targ_lang_train)                    
                        #predicted_val_monitor_outputs.append( predicted_sent + "(" + str(val_line_num) + ")" )
                        predicted_val_monitor_outputs.append( clean_tag(predicted_sent) )                    
                        #predicted_val_monitor_list.append( split(clean_tag(predicted_sent)) )                    

                        val_line_num += 1
                    #}   

                    # END VALIDATION BATCH                
                #} 


                save_file_fromList(actual_val_monitor_fileName,    actual_val_monitor_outputs,    writeMode= "w") #add
                save_file_fromList(predicted_val_monitor_fileName, predicted_val_monitor_outputs, writeMode= "w") #add

                #val_BLEU_score = calc_BLEU(actual_Val_monitor_fileName, predicted_Val_monitor_fileName, actual_Val_monitor_list[0], predicted_Val_monitor_list[0])  #add
                #val_monitor_BLEU_score  = calc_bleu(actual_val_monitor_fileName,  predicted_val_monitor_fileName, bpe_delimiter=" ") #add
                #val_monitor_ROUGE_score = calc_rouge(actual_val_monitor_fileName, predicted_val_monitor_fileName, bpe_delimiter=" ") #add	
                val_monitor_BLEU_score  = 0.1 #add
                val_monitor_ROUGE_score = 0.1 #add					

                val_loss_result     = np.squeeze(valid_loss_avg.result())
                val_accuracy_result = np.squeeze(valid_accuracy.result())            

                if val_loss_result < float(BEST_VALIDATION_LOSS):
                #{
                    BEST_VALIDATION_LOSS  = val_loss_result
                    BEST_EPOCH_VALID_LOSS = TOTAL_EPOCHS
                #}

                if val_accuracy_result > float(BEST_VALIDATION_ACCURACY):
                #{
                    BEST_VALIDATION_ACCURACY  = val_accuracy_result
                    BEST_EPOCH_VALID_ACCURACY = TOTAL_EPOCHS
                #} 


                if val_monitor_BLEU_score > float(BEST_VALIDATION_BLEU):
                #{
                    BEST_VALIDATION_BLEU  = val_monitor_BLEU_score
                    BEST_EPOCH_VALID_BLEU = TOTAL_EPOCHS
                #} 


                if val_monitor_ROUGE_score > float(BEST_VALIDATION_ROUGE):
                #{
                    BEST_VALIDATION_ROUGE  = val_monitor_ROUGE_score
                    BEST_EPOCH_VALID_ROUGE = TOTAL_EPOCHS
                #} 


                diff_loss_result = np.absolute(val_loss_result - train_loss_avg.result())

                if ( diff_loss_result < float(BEST_DIFFERENCE_LOSS) ):
                #{
                    BEST_DIFFERENCE_LOSS       = diff_loss_result
                    BEST_EPOCH_DIFFERENCE_LOSS = TOTAL_EPOCHS
                #}    

                BEST_TRAIN_LOSS_STRING          = "{:.4f}|{}".format(float(BEST_TRAIN_LOSS), BEST_EPOCH_TAINING_LOSS)
                BEST_TRAIN_ACCURACY_STRING      = "{:.4f}|{}".format(float(BEST_TRAIN_ACCURACY), BEST_EPOCH_TAINING_ACCURACY)                

                BEST_VALIDATION_LOSS_STRING     = "{:.4f}|{}".format(float(BEST_VALIDATION_LOSS),     BEST_EPOCH_VALID_LOSS)    
                BEST_VALIDATION_ACCURACY_STRING = "{:.4f}|{}".format(float(BEST_VALIDATION_ACCURACY), BEST_EPOCH_VALID_ACCURACY)
                BEST_VALIDATION_BLEU_STRING     = "{:.4f}|{}".format(float(BEST_VALIDATION_BLEU),     BEST_EPOCH_VALID_BLEU)
                BEST_VALIDATION_ROUGE_STRING    = "{:.4f}|{}".format(float(BEST_VALIDATION_ROUGE),    BEST_EPOCH_VALID_ROUGE)                
                BEST_DIFFERENCE_LOSS_STRING     = "{:.4f}|{}".format(float(BEST_DIFFERENCE_LOSS),     BEST_EPOCH_DIFFERENCE_LOSS)

                saveConfiguration(config_file, 'DEFAULT', 'BEST_TRAIN_LOSS', BEST_TRAIN_LOSS_STRING) 
                saveConfiguration(config_file, 'DEFAULT', 'BEST_TRAIN_ACCURACY', BEST_TRAIN_ACCURACY_STRING)

                saveConfiguration(config_file, 'DEFAULT', 'BEST_VALIDATION_LOSS',     BEST_VALIDATION_LOSS_STRING)  
                saveConfiguration(config_file, 'DEFAULT', 'BEST_VALIDATION_ACCURACY', BEST_VALIDATION_ACCURACY_STRING) 
                saveConfiguration(config_file, 'DEFAULT', 'BEST_VALIDATION_BLEU',     BEST_VALIDATION_BLEU_STRING)
                saveConfiguration(config_file, 'DEFAULT', 'BEST_VALIDATION_ROUGE',    BEST_VALIDATION_ROUGE_STRING)
                saveConfiguration(config_file, 'DEFAULT', 'BEST_DIFFERENCE_LOSS',     BEST_DIFFERENCE_LOSS_STRING)     

                if(plotting_enable):
                    simpleTrainingCurves.add(np.squeeze(train_loss_avg.result()), val_loss_result, np.squeeze(train_accuracy.result()), val_accuracy_result, True)
                    simpleTrainingCurves2.add(train_monitor_BLEU_score, val_monitor_BLEU_score, train_monitor_ROUGE_score, val_monitor_ROUGE_score, False)

                print_sample(actual_train_monitor_outputs, predicted_train_monitor_outputs, 4, "Training")
                print_sample(actual_val_monitor_outputs, predicted_val_monitor_outputs, 4, "Validation")

                train_loss_avg_result_list.append( np.squeeze(train_loss_avg.result()) )
                train_accuracy_result_list.append( np.squeeze(train_accuracy.result()) )
                train_perplexity_result_list.append( tf.exp(train_loss_avg.result()) )                        
                train_monitor_BLEU_score_list.append( train_monitor_BLEU_score )
                train_monitor_ROUGE_score_list.append( train_monitor_ROUGE_score )

                val_loss_result_list.append( val_loss_result )
                val_accuracy_result_list.append( val_accuracy_result )
                val_perplexity_result_list.append( tf.exp(valid_loss_avg.result()) )
                val_monitor_BLEU_score_list.append( val_monitor_BLEU_score )
                val_monitor_ROUGE_score_list.append( val_monitor_ROUGE_score )

                diff_loss_result_list.append( diff_loss_result ) 

                monitor =tabulate(\
                [['Optimizer        :', optimizer_inst.optimizer_name,                     'Learning Rate   :', optimizer_inst.learning_rate],
                ['Epoch             :', epoch+1,                                           'Total_Epoch     :', TOTAL_EPOCHS], 
                ['Train_Loss        :', "{:.4f}".format(train_loss_avg.result()),          'Val_Loss        :', "{:.4f}".format(valid_loss_avg.result()), 'DIFF_Loss :', "{:.4f}".format(diff_loss_result)],
                ['Train_Perplexity  :', "{:.4f}".format(tf.exp(train_loss_avg.result())),  'Val_Perplexity  :', "{:.4f}".format(tf.exp(valid_loss_avg.result()))],
                ['Train_Accuracy    :', "{:.4f}".format(train_accuracy.result()),          'Val_Accuracy    :', "{:.4f}".format(valid_accuracy.result())],
                ['Train_BLEU_score  :', "{:.4f}".format(train_monitor_BLEU_score),         'Val_BLEU_score  :', "{:.4f}".format(val_monitor_BLEU_score)],
                ['Train_ROUGE_score :', "{:.4f}".format(train_monitor_ROUGE_score),        'Val_ROUGE_score :', "{:.4f}".format(val_monitor_ROUGE_score)]])

                monitor += '\nTime taken for 1 epoch {} sec\n\n'.format(time.time() - start)

                epoch_log += monitor            
                total_log += monitor


                STAT_LOG = "Best Train Loss          : {}\n"\
                           "Best Train Accuracy      : {}\n"\
                           "Best Validation Loss     : {}\n"\
                           "Best Validation Accuracy : {}\n"\
                           "Best Difference Loss     : {}\n"\
                           "Best Validation BLEU     : {}\n"\
                           "Best Validation ROUGE    : {}\n".format(  BEST_TRAIN_LOSS_STRING,
                                                                      BEST_TRAIN_ACCURACY_STRING,
                                                                      BEST_VALIDATION_LOSS_STRING,  
                                                                      BEST_VALIDATION_ACCURACY_STRING,
                                                                      BEST_DIFFERENCE_LOSS_STRING,
                                                                      BEST_VALIDATION_BLEU_STRING,
                                                                      BEST_VALIDATION_ROUGE_STRING )       


                # saving (checkpoint) the model#add every N epochs
                if (epoch + 1) % save_checkpoint == 0:
                #{
                    serialize_object( train_loss_avg_result_list, checkpoint_dir+'/serialize/train_loss_avg_result_list.pkl')
                    serialize_object( train_accuracy_result_list, checkpoint_dir+'/serialize/train_accuracy_result_list.pkl')
                    serialize_object( train_perplexity_result_list, checkpoint_dir+'/serialize/train_perplexity_result_list.pkl')
                    serialize_object( train_monitor_BLEU_score_list, checkpoint_dir+'/serialize/train_monitor_BLEU_score_list.pkl')
                    serialize_object( train_monitor_ROUGE_score_list, checkpoint_dir+'/serialize/train_monitor_ROUGE_score_list.pkl')

                    serialize_object( val_loss_result_list, checkpoint_dir+'/serialize/val_loss_result_list.pkl')
                    serialize_object( val_accuracy_result_list, checkpoint_dir+'/serialize/val_accuracy_result_list.pkl')
                    serialize_object( val_perplexity_result_list, checkpoint_dir+'/serialize/val_perplexity_result_list.pkl')
                    serialize_object( val_monitor_BLEU_score_list, checkpoint_dir+'/serialize/val_monitor_BLEU_score_list.pkl')
                    serialize_object( val_monitor_ROUGE_score_list, checkpoint_dir+'/serialize/val_monitor_ROUGE_score_list.pkl')

                    serialize_object( diff_loss_result_list, checkpoint_dir+'/serialize/diff_loss_result_list.pkl')

                    saver_info = "-EP:{}-LR:{}-TRAIN_LOSS:{:.4f}-VAL_LOSS:{:.4f}-"\
                                 "TRAIN_ACC:{:.4f}-VAL_ACC:{:.4f}-DIFF_LOSS:{:.4f}"\
                                 "TRAIN_BLEU:{:.4f}-VAL_BLEU:{:.4f}-TRAIN_ROUGE:{:.4f}-VAL_ROUGE:{:.4f}".format(  TOTAL_EPOCHS, optimizer_inst.learning_rate,
                                                                                                                  train_loss_avg.result(), valid_loss_avg.result(),
                                                                                                                  train_accuracy.result(),valid_accuracy.result(),diff_loss_result,                                                                                                            
                                                                                                                  train_monitor_BLEU_score, val_monitor_BLEU_score,
                                                                                                                  train_monitor_ROUGE_score, val_monitor_ROUGE_score
                                                                                                               )         
                    checkpoint_inst.checkpoint.save(file_prefix = checkpoint_inst.checkpoint_prefix  + saver_info )


                    saveConfiguration(config_file, 'DEFAULT', 'TOTAL_EPOCHS', str(TOTAL_EPOCHS))            
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_TRAINING_LOSS', "{:.4f}".format(train_loss_avg.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_TRAINING_ACCURACY', "{:.4f}".format(train_accuracy.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'MODEL_VER', str(MODEL_VER))

                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_VALIDATION_LOSS', "{:.4f}".format(valid_loss_avg.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_VALIDATION_ACCURACY', "{:.4f}".format(valid_accuracy.result()))

                    epoch_log += "Model Saved.\n\n"
                    total_log += "Model Saved.\n\n" 

                    # END SAVER
                #}         
            #}        

            else:
            #{
                simpleTrainingCurves.add(np.squeeze(train_loss_avg.result()), 0, np.squeeze(train_accuracy.result()), 0)

                print_sample(actual_train_monitor_outputs, predicted_train_monitor_outputs, 4, "Training")

                train_loss_avg_result_list.append( np.squeeze(train_loss_avg.result()) )
                train_accuracy_result_list.append( np.squeeze(train_accuracy.result()) )
                train_perplexity_result_list.append( tf.exp(train_loss_avg.result()) )                        
                train_monitor_BLEU_score_list.append( train_monitor_BLEU_score )
                train_monitor_ROUGE_score_list.append( train_monitor_ROUGE_score )

                BEST_TRAIN_LOSS_STRING     = "{:.4f}|{}".format(float(BEST_TRAIN_LOSS), BEST_EPOCH_TAINING_LOSS)
                BEST_TRAIN_ACCURACY_STRING = "{:.4f}|{}".format(float(BEST_TRAIN_ACCURACY), BEST_EPOCH_TAINING_ACCURACY)

                saveConfiguration(config_file, 'DEFAULT', 'BEST_TRAIN_LOSS', BEST_TRAIN_LOSS_STRING) 
                saveConfiguration(config_file, 'DEFAULT', 'BEST_TRAIN_ACCURACY', BEST_TRAIN_ACCURACY_STRING)


                monitor =tabulate(\
                [['Optimizer        :', optimizer_inst.optimizer_name,   'Learning Rate   :', optimizer_inst.learning_rate],
                ['Epoch             :', epoch+1,      'Total_Epoch     :', TOTAL_EPOCHS], 
                ['Train_Loss        :', "{:.4f}".format(train_loss_avg.result())],
                ['Train_Perplexity  :', "{:.4f}".format(tf.exp(train_loss_avg.result()))],
                ['Train_Accuracy    :', "{:.4f}".format(train_accuracy.result())],            
                ['Train_BLEU_score  :', "{:.4f}".format(train_monitor_BLEU_score)],
                ['Train_ROUGE_score :', "{:.4f}".format(train_monitor_ROUGE_score)]])

                monitor += '\nTime taken for 1 epoch {} sec\n\n'.format(time.time() - start)                                                    

                epoch_log += monitor            
                total_log += monitor 


                STAT_LOG = "Best Train Loss          : {}\n"\
                           "Best Train Accuracy      : {}\n".format(BEST_TRAIN_LOSS_STRING, BEST_TRAIN_ACCURACY_STRING)


                # saving (checkpoint) the model#add every N epochs
                if (epoch + 1) % save_checkpoint == 0:
                #{
                    serialize_object( train_loss_avg_result_list, checkpoint_dir+'/serialize/train_loss_avg_result_list.pkl')
                    serialize_object( train_accuracy_result_list, checkpoint_dir+'/serialize/train_accuracy_result_list.pkl')
                    serialize_object( train_perplexity_result_list, checkpoint_dir+'/serialize/train_perplexity_result_list.pkl')              

                    saver_info = "-EP:{}-LR:{}-TRAIN_LOSS:{:.4f}-TRAIN_ACC:{:.4f}-TRAIN_BLEU:{:.4f}-TRAIN_ROUGE:{:.4f}".format(  TOTAL_EPOCHS, optimizer_inst.learning_rate,
                                                                                                                                 train_loss_avg.result(), train_accuracy.result(),
                                                                                                                                 train_monitor_BLEU_score, train_monitor_ROUGE_score
                                                                                                                              )

                    checkpoint_inst.checkpoint.save(file_prefix = checkpoint_inst.checkpoint_prefix + saver_info)

                    saveConfiguration(config_file, 'DEFAULT', 'TOTAL_EPOCHS', str(TOTAL_EPOCHS))            
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_TRAINING_LOSS', "{:.4f}".format(train_loss_avg.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_TRAINING_ACCURACY', "{:.4f}".format(train_accuracy.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'MODEL_VER', str(MODEL_VER))

                    epoch_log += "Model Saved.\n\n"
                    total_log += "Model Saved.\n\n" 

                    # END SAVER
                #}
            #}


            total_epochs_monitor =\
            "............................................\n"\
            "Total Epoch Done\t : {}\n"\
            "............................................\n".format(TOTAL_EPOCHS)


            total_epochs_model_saved_value = readConfiguration( config_file, 'DEFAULT', 'TOTAL_EPOCHS' )
            if total_epochs_model_saved_value == None:
                total_epochs_model_saved_value = 0

            total_epochs_model_saved_monitor =\
            "Total Epochs Model Saved : {}\n"\
            "............................................\n\n".format(total_epochs_model_saved_value)


            log_footer = STAT_LOG + total_epochs_monitor + total_epochs_model_saved_monitor

            #print(log_footer)

            epoch_log += log_footer
            total_log += log_footer


            if plotting_enable:
                print(total_log, end="")

            else:
                print(monitor, end="") 


            save_log(log_file, epoch_log, epoch_logger_mode)        
            epoch_log   = ""

            # END EPOCH
        #}

        save_log(log_file, total_log, (not epoch_logger_mode))

        # END TRAINING
    #}
#}