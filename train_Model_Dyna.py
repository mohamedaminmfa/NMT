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
import datetime
import numpy as np
import configparser
import _pickle as pickle
import matplotlib.pyplot as plt

from tabulate import tabulate

from SL.lib.NMT_lib.text_processing_util import *
from SL.lib.NMT_lib.lang_util import *
from SL.lib.NMT_lib.graph_plotting_util import *

def train_Model(hparams,
                train_dataset,
                val_dataset,				
                model_inst,
                checkpoint_inst
                ):
#{

    if hparams.do_training:
    #{
        num_epochs                = hparams.num_epochs
        batch_size                = hparams.batch_size
        inp_lang_train            = hparams.inp_lang_train
        targ_lang_train           = hparams.targ_lang_train
        plotting_enable           = hparams.plotting_enable
        gradient_clipping_enabled = hparams.gradient_clipping_enabled
        max_gradient_norm         = hparams.max_gradient_norm
        checkpoint_dir            = hparams.checkpoint_dir
        log_file                  = hparams.log_file
        config_file               = hparams.config_file

        data_partition            = hparams.dataset_split
        progress_bar              = hparams.progress_bar
        model_desc                = hparams.model_desc
        save_checkpoint           = hparams.save_checkpoint
        epoch_logger_mode         = hparams.epoch_logger_mode
        eval_validation_set       = hparams.eval_validation_set
        eval_validation_set_epc   = hparams.eval_validation_set_epc
        dataset_split_info        = hparams.dataset_split_info
        load_dataset_spliter_info = hparams.load_dataset_spliter_info
        spec_chk                  = hparams.specific_checkpoint
        epc_chk                   = None

        LAST_MODEL_VER            = readConfiguration( config_file, 'DEFAULT', 'MODEL_VER' )
       
       
        if spec_chk != None: 
            epc_chk = int(spec_chk[spec_chk.find("EP:")+3: spec_chk.find("-LR")])
            print("epc_chk:", epc_chk)

        if LAST_MODEL_VER == None:  
            save_log(log_file, dataset_split_info)
            save_log(log_file, load_dataset_spliter_info)
            save_log(log_file, model_desc)			
            print()


        global_step                    = tf.train.get_or_create_global_step()

        if spec_chk:
            global_step = tf.assign( ref=global_step, value=epc_chk)


        train_loss_avg_result_list     = read_serialize_object(checkpoint_dir+'/serialize/train_loss_avg_result_list.pkl', objectType=list)
        train_loss_avg_result_list     = train_loss_avg_result_list[:epc_chk] if spec_chk else train_loss_avg_result_list

        train_accuracy_result_list     = read_serialize_object(checkpoint_dir+'/serialize/train_accuracy_result_list.pkl', objectType=list)
        train_accuracy_result_list     = train_accuracy_result_list[:epc_chk] if spec_chk else train_accuracy_result_list

        train_perplexity_result_list   = read_serialize_object(checkpoint_dir+'/serialize/train_perplexity_result_list.pkl', objectType=list)
        train_perplexity_result_list   = train_perplexity_result_list[:epc_chk] if spec_chk else train_perplexity_result_list

        train_monitor_BLEU_score_list  = read_serialize_object(checkpoint_dir+'/serialize/train_monitor_BLEU_score_list.pkl', objectType=list)
        train_monitor_BLEU_score_list  = train_monitor_BLEU_score_list[:epc_chk] if spec_chk else train_monitor_BLEU_score_list

        train_monitor_ROUGE_score_list = read_serialize_object(checkpoint_dir+'/serialize/train_monitor_ROUGE_score_list.pkl', objectType=list)
        train_monitor_ROUGE_score_list = train_monitor_ROUGE_score_list[:epc_chk] if spec_chk else train_monitor_ROUGE_score_list

        val_loss_result_list           = read_serialize_object(checkpoint_dir+'/serialize/val_loss_result_list.pkl', objectType=list)
        val_loss_result_list           = val_loss_result_list[:epc_chk] if spec_chk else  val_loss_result_list

        val_accuracy_result_list       = read_serialize_object(checkpoint_dir+'/serialize/val_accuracy_result_list.pkl', objectType=list)
        val_accuracy_result_list       = val_accuracy_result_list[:epc_chk] if spec_chk else val_accuracy_result_list

        val_perplexity_result_list     = read_serialize_object(checkpoint_dir+'/serialize/val_perplexity_result_list.pkl', objectType=list)
        val_perplexity_result_list     = val_perplexity_result_list[:epc_chk] if spec_chk else val_perplexity_result_list

        val_monitor_BLEU_score_list    = read_serialize_object(checkpoint_dir+'/serialize/val_monitor_BLEU_score_list.pkl', objectType=list)
        val_monitor_BLEU_score_list    = val_monitor_BLEU_score_list[:epc_chk] if spec_chk else val_monitor_BLEU_score_list

        val_monitor_ROUGE_score_list   = read_serialize_object(checkpoint_dir+'/serialize/val_monitor_ROUGE_score_list.pkl', objectType=list)
        val_monitor_ROUGE_score_list   = val_monitor_ROUGE_score_list[:epc_chk] if spec_chk else val_monitor_ROUGE_score_list

        diff_loss_result_list          = read_serialize_object(checkpoint_dir+'/serialize/diff_loss_result_list.pkl', objectType=list)
        diff_loss_result_list          = diff_loss_result_list[:epc_chk] if spec_chk else diff_loss_result_list

        MODEL_VER = 1
        
        from sklearn.metrics import accuracy_score

        if(plotting_enable):
        #{
            simpleTrainingCurves  = SimpleTrainingCurves("cross-entropy", "accuracy")
            simpleTrainingCurves2 = SimpleTrainingCurves("BLEU-SCORE", "ROUGE-SCORE")

            accuracy_result_len   = min(len(val_accuracy_result_list), len(train_accuracy_result_list))
            loss_result_len       = min(len(val_loss_result_list), len(train_loss_avg_result_list))
            BLEU_score_len        = min(len(val_monitor_BLEU_score_list), len(train_monitor_BLEU_score_list))
            ROUGE_score_len       = min(len(val_monitor_ROUGE_score_list), len(train_monitor_ROUGE_score_list))            
            plotting_len          = epc_chk if spec_chk else min(accuracy_result_len, loss_result_len, BLEU_score_len, ROUGE_score_len)

            for i in range(plotting_len):
            #{
                if(eval_validation_set):
                #{   
                    simpleTrainingCurves.add(train_loss_avg_result_list[i], val_loss_result_list[i], train_accuracy_result_list[i], val_accuracy_result_list[i], True)
                    simpleTrainingCurves2.add(train_monitor_BLEU_score_list[i], val_monitor_BLEU_score_list[i], train_monitor_ROUGE_score_list[i], val_monitor_ROUGE_score_list[i], False)
                #}
                else:
                #{
                    simpleTrainingCurves.add(train_loss_avg_result_list[i], 0, train_accuracy_result_list[i], 0, True)
                    simpleTrainingCurves2.add(train_monitor_BLEU_score_list[i], 0, train_monitor_ROUGE_score_list[i], 0, False)
                #}
            #}
        #}


        LAST_MODEL_VER = readConfiguration( config_file, 'DEFAULT', 'MODEL_VER' )         
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
        log_header += "Batch Size   : {}\nLearning Rate: {}\n".format( batch_size, model_inst.optimizer_inst.learning_rate)
        

        dropout_str = "Dropout      : [I:{}, S:{}, O:{}]\nEnc_Dropout  : [T:{}, S:{}]\nDec_Dropout  : [T:{}, S:{}]\n".format(hparams.input_dropout,
                                                                                                                            hparams.state_dropout,
                                                                                                                            hparams.output_dropout,
                                                                                                                            hparams.encoder_embedding_TimestepDropout,
                                                                                                                            hparams.encoder_embedding_SpatialDropout1D,
                                                                                                                            hparams.decoder_embedding_TimestepDropout,
                                                                                                                            hparams.decoder_embedding_SpatialDropout1D)

        log_header += dropout_str

        epoch_log  = log_header            
        total_log  = log_header 

        print(log_header, end="")

        encoder_checkpoint_dir    = checkpoint_dir+ '/weights_encoder'
        decoder_checkpoint_dir    = checkpoint_dir+ '/weights_decoder'
        encoder_checkpoint_prefix = os.path.join(encoder_checkpoint_dir, "ckpt")
        decoder_checkpoint_prefix = os.path.join(decoder_checkpoint_dir, "ckpt")

        for epoch in range(num_epochs):
        #{
            global_step = tf.assign_add(global_step, 1, name ='increment_global_step')

            train_loss_avg    = tf.keras.metrics.Mean()
            valid_loss_avg    = tf.keras.metrics.Mean()

            train_accuracy    = tf.keras.metrics.Accuracy()
            valid_accuracy    = tf.keras.metrics.Accuracy()
           
            #actual_val_monitor_list, predicted_val_monitor_list = list(), list()                    #add        
            actual_val_monitor_fileName     = checkpoint_dir + "/eval_val_monitor_actual.txt"        #add
            predicted_val_monitor_fileName  = checkpoint_dir + "/eval_val_monitor_predicted.txt"     #add        
            actual_val_monitor_outputs, predicted_val_monitor_outputs  = list(), list()              #add  
            
            #actual_train_monitor_list, predicted_train_monitor_list = list(), list()                #add
            actual_train_monitor_fileName     = checkpoint_dir + "/eval_train_monitor_actual.txt"    #add
            predicted_train_monitor_fileName  = checkpoint_dir + "/eval_train_monitor_predicted.txt" #add        
            actual_train_monitor_outputs, predicted_train_monitor_outputs  = list(), list()          #add
            
            val_line_num, train_line_num = 1, 1
            
            
            TOTAL_EPOCHS      = int(TOTAL_EPOCHS) + 1

            start = time.time()

            batchCounter      = 0
            
            hidden = model_inst.encoder.initialize_hidden_state()

            train_total_loss  = 0
            valid_total_loss  = 0

            train_batch_loss  = []
            valid_batch_loss  = []

            print( "Epoch %d "%( epoch+1 ) , end="\n")

            for (train_step, (train_input, train_label)) in enumerate(train_dataset):
            #{
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
                    outs, loss = model_inst(train_input, 
                                            hidden, 
                                            train_label,
                                            loss,
                                            LEARN_MODE= tf.contrib.learn.ModeKeys.TRAIN)   #<================================================
                    
                
                train_batch_loss  = (loss / int(train_label.shape[1]))
                train_total_loss += train_batch_loss

                variables = model_inst.encoder.trainable_variables + model_inst.decoder.trainable_variables
                
                if(gradient_clipping_enabled):
                    # Gradient clipping to avoid "exploding gradients"
                    gradients, _ = tf.clip_by_global_norm(tape.gradient(loss, variables), max_gradient_norm)
                else:
                    gradients = tape.gradient(loss, variables) 
                    
                model_inst.optimizer_inst.optimizer.apply_gradients(zip(gradients, variables)) 

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
            
            #} END TRAINING BATCH    
    
            save_file_fromList(actual_train_monitor_fileName,    actual_train_monitor_outputs,    writeMode= "w") #add
            save_file_fromList(predicted_train_monitor_fileName, predicted_train_monitor_outputs, writeMode= "w") #add

            train_monitor_BLEU_score  = calc_bleu(actual_train_monitor_fileName,  predicted_train_monitor_fileName, bpe_delimiter=" ") #add
            train_monitor_ROUGE_score = calc_rouge(actual_train_monitor_fileName, predicted_train_monitor_fileName, bpe_delimiter=" ") #add
            
            if np.squeeze(train_loss_avg.result()) < float(BEST_TRAIN_LOSS):
                BEST_TRAIN_LOSS = np.squeeze(train_loss_avg.result()) 
                BEST_EPOCH_TAINING_LOSS = TOTAL_EPOCHS
                
            if np.squeeze(train_accuracy.result()) > float(BEST_TRAIN_ACCURACY):
                BEST_TRAIN_ACCURACY = np.squeeze(train_accuracy.result())
                BEST_EPOCH_TAINING_ACCURACY = TOTAL_EPOCHS 


            # Evaluation
            if(eval_validation_set and ( (epoch+1) % eval_validation_set_epc == 0 )):
            #{   
                for (val_step, (val_input, val_label)) in enumerate(val_dataset):                
                #{
                    val_predictions, valid_batch_loss = model_inst(val_input, 
                                                                hidden, 
                                                                val_label, 
                                                                0, 
                                                                LEARN_MODE= tf.contrib.learn.ModeKeys.EVAL)  #<================================================
                                                                

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
                #} END VALIDATION BATCH 

                
                save_file_fromList(actual_val_monitor_fileName,    actual_val_monitor_outputs,    writeMode= "w") #add
                save_file_fromList(predicted_val_monitor_fileName, predicted_val_monitor_outputs, writeMode= "w") #add
                    
                #val_BLEU_score = calc_BLEU(actual_Val_monitor_fileName, predicted_Val_monitor_fileName, actual_Val_monitor_list[0], predicted_Val_monitor_list[0])  #add
                val_monitor_BLEU_score  = calc_bleu(actual_val_monitor_fileName,  predicted_val_monitor_fileName, bpe_delimiter=" ") #add
                val_monitor_ROUGE_score = calc_rouge(actual_val_monitor_fileName, predicted_val_monitor_fileName, bpe_delimiter=" ") #add	
                        
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
                [['Optimizer         :', model_inst.optimizer_inst.optimizer_name,         'Learning Rate   :', model_inst.optimizer_inst.learning_rate],
                ['Epoch             :', epoch+1,                                           'Total_Epoch     :', TOTAL_EPOCHS], 
                ['Train_Loss        :', "{:.4f}".format(train_loss_avg.result()),          'Val_Loss        :', "{:.4f}".format(valid_loss_avg.result()), 'DIFF_Loss :', "{:.4f}".format(diff_loss_result)],
                ['Train_Perplexity  :', "{:.4f}".format(tf.exp(train_loss_avg.result())),  'Val_Perplexity  :', "{:.4f}".format(tf.exp(valid_loss_avg.result()))],
                ['Train_Accuracy    :', "{:.4f}".format(train_accuracy.result()),          'Val_Accuracy    :', "{:.4f}".format(valid_accuracy.result())],
                ['Train_BLEU_score  :', "{:.4f}".format(train_monitor_BLEU_score),         'Val_BLEU_score  :', "{:.4f}".format(val_monitor_BLEU_score)],
                ['Train_ROUGE_score :', "{:.4f}".format(train_monitor_ROUGE_score),        'Val_ROUGE_score :', "{:.4f}".format(val_monitor_ROUGE_score)]])
            
                now = datetime.datetime.now()
                monitor += '\nDate/Time {}\nTime taken for 1 epoch {} sec\n\n'.format( str(now)[:19], (time.time() - start))
            
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

                    now = datetime.datetime.now()
                    saver_info = "-DT:{}-GS:{}-EP:{}-LR:{}-TRAIN_LOSS:{:.4f}-VAL_LOSS:{:.4f}-"\
                                 "TRAIN_ACC:{:.4f}-VAL_ACC:{:.4f}-DIFF_LOSS:{:.4f}"\
                                 "TRAIN_BLEU:{:.4f}-VAL_BLEU:{:.4f}-TRAIN_ROUGE:{:.4f}-VAL_ROUGE:{:.4f}".format( str(now)[:19], global_step.numpy(), TOTAL_EPOCHS, model_inst.optimizer_inst.learning_rate,
                                                                                                                train_loss_avg.result(), valid_loss_avg.result(),
                                                                                                                train_accuracy.result(),valid_accuracy.result(),diff_loss_result,                                                                                                            
                                                                                                                train_monitor_BLEU_score, val_monitor_BLEU_score,
                                                                                                                train_monitor_ROUGE_score, val_monitor_ROUGE_score)
                                                                                                               

                    if(isinstance(checkpoint_inst, tf.train.Checkpoint)):
                        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
                        checkpoint_inst.save(file_prefix= checkpoint_prefix  + saver_info )    
                    else:
                        checkpoint_inst.checkpoint.save(file_prefix = checkpoint_inst.checkpoint_prefix  + saver_info )

                    #model_inst.encoder.save_weights(encoder_checkpoint_prefix)
                    #model_inst.decoder.save_weights(decoder_checkpoint_prefix)
                    
                    #global_step   = tf.train.get_or_create_global_step()
                    #all_variables = ( model_inst.encoder.variables + model_inst.decoder.variables + model_inst.optimizer_inst.optimizer.variables() + [global_step] + [train_loss_avg] + [valid_loss_avg] + [train_accuracy] + [valid_accuracy] )
                    #tfe.Saver(all_variables).save(checkpoint_inst.checkpoint_prefix + saver_info)
    
                    
                    saveConfiguration(config_file, 'DEFAULT', 'TOTAL_EPOCHS', str(TOTAL_EPOCHS))            
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_TRAINING_LOSS', "{:.4f}".format(train_loss_avg.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_TRAINING_ACCURACY', "{:.4f}".format(train_accuracy.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'MODEL_VER', str(MODEL_VER))
                    
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_VALIDATION_LOSS', "{:.4f}".format(valid_loss_avg.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_VALIDATION_ACCURACY', "{:.4f}".format(valid_accuracy.result()))

                    epoch_log += "Model Saved.\n\n"
                    total_log += "Model Saved.\n\n" 

                #}  END SAVER       
            #}  END Evaluation 


            # Training Only
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
                [['Optimizer        :', model_inst.optimizer_inst.optimizer_name,   'Learning Rate   :', model_inst.optimizer_inst.learning_rate],
                ['Epoch             :', epoch+1,      'Total_Epoch     :', TOTAL_EPOCHS], 
                ['Train_Loss        :', "{:.4f}".format(train_loss_avg.result())],
                ['Train_Perplexity  :', "{:.4f}".format(tf.exp(train_loss_avg.result()))],
                ['Train_Accuracy    :', "{:.4f}".format(train_accuracy.result())],            
                ['Train_BLEU_score  :', "{:.4f}".format(train_monitor_BLEU_score)],
                ['Train_ROUGE_score :', "{:.4f}".format(train_monitor_ROUGE_score)]])
        
                now = datetime.datetime.now()
                monitor += '\nDate/Time {}\nTime taken for 1 epoch {} sec\n\n'.format( str(now)[:19], (time.time() - start))                                                  
                
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
                    
                    now = datetime.datetime.now()
                    saver_info = "-DT:{}-GS:{}-EP:{}-LR:{}-TRAIN_LOSS:{:.4f}-TRAIN_ACC:{:.4f}-TRAIN_BLEU:{:.4f}-TRAIN_ROUGE:{:.4f}".format(   str(now)[:19], global_step.numpy(), TOTAL_EPOCHS, model_inst.optimizer_inst.learning_rate,
                                                                                                                                        train_loss_avg.result(), train_accuracy.result(),
                                                                                                                                        train_monitor_BLEU_score, train_monitor_ROUGE_score)
                    
                    if(isinstance(checkpoint_inst, tf.train.Checkpoint)):
                        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
                        checkpoint_inst.save(file_prefix= checkpoint_prefix  + saver_info )    
                    else:
                        checkpoint_inst.checkpoint.save(file_prefix = checkpoint_inst.checkpoint_prefix  + saver_info )

                    #model_inst.encoder.save_weights(encoder_checkpoint_prefix)
                    #model_inst.decoder.save_weights(decoder_checkpoint_prefix)
                    
					#global_step = tf.train.get_or_create_global_step()
                    #all_variables = ( encoder.variables + decoder.variables + optimizer.variables() + [global_step] + [train_loss_avg] + [train_accuracy] )
                    #tfe.Saver(all_variables).save(checkpoint_inst.checkpoint_prefix + saver_info)

                    saveConfiguration(config_file, 'DEFAULT', 'TOTAL_EPOCHS', str(TOTAL_EPOCHS))            
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_TRAINING_LOSS', "{:.4f}".format(train_loss_avg.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'CURRENT_TRAINING_ACCURACY', "{:.4f}".format(train_accuracy.result()))
                    saveConfiguration(config_file, 'DEFAULT', 'MODEL_VER', str(MODEL_VER))
                    
                    epoch_log += "Model Saved.\n\n"
                    total_log += "Model Saved.\n\n"
                    
                #} END SAVER
                
            #} END Training Only
                                                                                    
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
           
        #} END EPOCH

        save_log(log_file, total_log, (not epoch_logger_mode))
        
    #} END TRAINING
#}