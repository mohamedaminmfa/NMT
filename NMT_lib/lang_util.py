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

import tensorflow as tf
from tabulate import tabulate
from keras import backend as K
from keras.layers import Dropout
from nltk.corpus import stopwords

from NMT_lib.bleu import *
from NMT_lib.rouge import *
from NMT_lib.text_processing_util import *

#...........................................................................................................................................................

def get_word2idx(string, lang_dic):
	try:
		index = lang_dic.word2idx[string]
	
	except Exception as ex:
		index = lang_dic.word2idx["<unk>"]

	return index

#...........................................................................................................................................................

def get_word2idx_OOV(string, lang_dic, counter):
	try:
		index = lang_dic.word2idx[string]
	
	except Exception as ex:
		index = lang_dic.word2idx["<unk>"]
		counter = counter + 1

	return index, counter
	
#...........................................................................................................................................................

def get_OOV(string, lang_dic, counter):
	try:
		index = lang_dic.word2idx[string]
	
	except Exception as ex:
		counter = counter + 1

	return counter	

#...........................................................................................................................................................

def get_idx2wordxx(index,lang_dic):

    return lang_dic.idx2word.get(index, "<unk>")

#...........................................................................................................................................................

def get_idx2word(word_id,lang_dic):

    return lang_dic.idx2word.get(tf.squeeze(word_id).numpy(), "<unk>")

#...........................................................................................................................................................

def sentence_to_tensor(sentence, langModel):
    
    return [ langModel.word2idx[s] for s in sentence.split(" ")]

#...........................................................................................................................................................

def tensor_to_sentence(tensor, langModel):
    words = []
    for word_id in tensor:
        
        word = get_idx2word(word_id, langModel)
        
        if(word.lower() == "<end>"):
             words.append(" " + "<end>")
             break
        
        elif(word.lower() != "<start>" and word.lower() != "<pad>"):
            words.append(" " + word)

        
    return "".join(words).strip()

#...........................................................................................................................................................

def tensor_to_sentence2(tensor, lang):
    
    result = ""
    for word_id in tensor:
        result += lang.idx2word[tf.squeeze(word_id).numpy()] + ' '
        
    return result.strip()

#...........................................................................................................................................................

def indexes2Sentence(list, langModel ):
    sentence =""
    for i in list:
        sentence += "".join(langModel.idx2word[i] +" ")

    return sentence.strip()

#...........................................................................................................................................................

def printb(value, printable=False):
    if printable:
        print(value)

#...........................................................................................................................................................
        
def print_sample(actual, predicted, num_of_sample, desc):        
    print("+"*200)      
    print("-------------------------------------")
    print(desc + " Actual Sentences:")
    print("-------------------------------------")                        
    print("\n".join(actual[:num_of_sample]))
    print("-------------------------------------")
    print(desc + " Predicted Sentences:")
    print("-------------------------------------")                        
    print("\n".join(predicted[:num_of_sample]))                    
    print("+"*200)  
    
#...........................................................................................................................................................        

def plot(List, Ylabel, axis=None):

    plt.ylabel(Ylabel)
    
    if axis:
        plt.axis(axis)
        
    for l in List: 
        plt.plot(l)
        
    plt.show()
    
#...........................................................................................................................................................        

def readConfiguration(file, section, option):
    try:
        config = configparser.ConfigParser()
        config.read(file)
        return config[section][option]

    except Exception as e:
        #print("EXCEPTION Read Configuration:", e)
        return None

#...........................................................................................................................................................
    
def saveConfiguration(file, section, option, value):    
    config = configparser.ConfigParser()
    
    file_dir = file[:file.rfind('/')+1]
    
    if not (os.path.exists(file_dir)):            
        os.makedirs(file_dir)
            
    if os.path.exists(file):
        config.read(file)
    
    cfgfile = open(file, 'w')
    config.set(section, option, value)
    config.write(cfgfile)
    cfgfile.close()
        
#...........................................................................................................................................................      

def save_file_fromList(filePath, List, writeMode= "a"):  
    file_dir = filePath[:filePath.rfind('/')+1]
    
    if( not (os.path.exists(file_dir)) and file_dir != ""):            
        os.makedirs(file_dir)

    if(os.path.exists(filePath) and writeMode.lower() == "w".lower()):
        os.remove(filePath)

    with open(filePath, writeMode) as out:
        for l in List:
            out.write("%s\n" % l)
#...........................................................................................................................................................

def save_file(filePath, text, writeMode= "a"):  
    file_dir = filePath[:filePath.rfind('/')+1]
    
    if( not (os.path.exists(file_dir)) and file_dir != ""):            
        os.makedirs(file_dir)

    if(os.path.exists(filePath) and writeMode.lower() == "w".lower()):
        os.remove(filePath)

    with open(filePath, writeMode) as out:
        out.writelines(text)
        out.write("\n")

#...........................................................................................................................................................

def read_file(file):     
    lines = None
    if(os.path.exists(file)):
        with open(file, 'r') as f:
            lines = f.readlines()    
    return lines

#...........................................................................................................................................................  

def serialize_object(Object, filePath, mode="wb"):
    
    file_dir = filePath[:filePath.rfind('/')+1]
    
    if( not (os.path.exists(file_dir)) and file_dir != ""):            
        os.makedirs(file_dir)
        
    with open(filePath, mode) as f:
        pickle.dump(Object, f)  
    
#...........................................................................................................................................................              
             
def read_serialize_object(filePath, mode="rb", objectType= None):
    
    if( not (os.path.exists(filePath))):

        if objectType == list:
            return list()
        else:
            return None
        
    with open(filePath, mode) as f:
        entry = pickle.load(f)
    return entry
              
#...........................................................................................................................................................
              
def save_log(file, text, enabled=True, writeMode= "a"):    
    if enabled:  
        save_file(file, text, writeMode)
            
#...........................................................................................................................................................

def deleteDirectory(directoryPath, ENABLED=True):
    try:
        if ENABLED:
            shutil.rmtree(directoryPath)            
            print("Directory Deleted [DONE]")

    except Exception as e:
        print(" deleteDirectory EXCEPTION : ", e)
        
#...........................................................................................................................................................        
        
def RESET_TF_SESSION():
    curr_session = tf.get_default_session()
   
    # close current session
    if curr_session is not None:
        curr_session.close()
    
    # reset graph
    K.clear_session()
    
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    
    return s
  
#...........................................................................................................................................................  
  
def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm), tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))]

    return clipped_gradients, gradient_norm_summary

#...........................................................................................................................................................

def clip_gradients(grads_and_vars, clip_ratio):
    gradients, variables = zip(*grads_and_vars)
    clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
    return zip(clipped, variables)

#...........................................................................................................................................................  
  
def _clean(sentence, bpe_delimiter):
    """Clean and handle BPE delimiter."""
    sentence = sentence.strip()

    # BPE
    if bpe_delimiter:
        sentence = re.sub(bpe_delimiter + " ", "", sentence)

    return sentence
	
#...........................................................................................................................................................	
    
def calc_rouge(ref_file, summarization_file, bpe_delimiter=None):
    """Compute ROUGE scores and handling BPE."""

    references = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
        for line in fh:
            references.append(_clean(line, bpe_delimiter))

    hypotheses = []
    with codecs.getreader("utf-8")(
            tf.gfile.GFile(summarization_file, "rb")) as fh:
        for line in fh:
            hypotheses.append(_clean(line, bpe_delimiter))

    rouge_score_map = rouge(hypotheses, references)
    
    return rouge_score_map["rouge_l/f_score"]	
  
#...........................................................................................................................................................  
  
# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def calc_bleu(ref_file, trans_file, bpe_delimiter=None):
    """Compute BLEU scores and handling BPE."""

    bleu_outputs = []
    
    max_order = 4
    for cc in range(1, 5):
        smooth = False
        max_order = cc

        ref_files = [ref_file]
        reference_text = []
        for reference_filename in ref_files:
            with codecs.getreader("utf-8")(
                    tf.gfile.GFile(reference_filename, "rb")) as fh:
                reference_text.append(fh.readlines())

        per_segment_references = []
        for references in zip(*reference_text):
            reference_list = []
            for reference in references:
                reference = _clean(reference, bpe_delimiter)
                reference_list.append(split(reference))
            per_segment_references.append(reference_list)

        translations = []
        with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
            for line in fh:
                line = _clean(line, bpe_delimiter)
                translations.append(split(line))

        # bleu_score, precisions, bp, ratio, translation_length, reference_length
        bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)
        
        bleu_outputs.append(bleu_score)
        
        #print(100 * bleu_score)
        
    return bleu_outputs[0]  


def _bleu2(per_segment_references, translations, bpe_delimiter=None):
    """Compute BLEU scores and handling BPE."""

    max_order = 4
    for cc in range(1, 5):
        smooth = False
        max_order = cc


        # bleu_score, precisions, bp, ratio, translation_length, reference_length
        bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)

        print(100 * bleu_score)
    return 100 * bleu_score 

#...........................................................................................................................................................

def wer_score(hyp, ref, print_matrix=False):
  N = len(hyp)
  M = len(ref)
  L = np.zeros((N,M))
  for i in range(0, N):
    for j in range(0, M):
      if min(i,j) == 0:
        L[i,j] = max(i,j)
      else:
        deletion = L[i-1,j] + 1
        insertion = L[i,j-1] + 1
        sub = 1 if hyp[i] != ref[j] else 0
        substitution = L[i-1,j-1] + sub
        L[i,j] = min(deletion, min(insertion, substitution))
        # print("{} - {}: del {} ins {} sub {} s {}".format(hyp[i], ref[j], deletion, insertion, substitution, sub))
  if print_matrix:
    print("WER matrix ({}x{}): ".format(N, M))
    print(L)
  return int(L[N-1, M-1])

#...........................................................................................................................................................

def wer(h, r):
    """
    https://martin-thoma.com/word-error-rate-calculation/
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

#...........................................................................................................................................................

def reverse_tensor(tensor):
    #tensor_reversed = np.fliplr(tensor)
    tensor_reversed = tf.reverse(tensor, [-1])
    return tensor_reversed

#...........................................................................................................................................................

def read_corpus(data_path, num_examples):
#{  
    if isinstance(data_path, list):
        if( num_examples == -1 ):
            corpus = data_path
        else:
            corpus = data_path[:num_examples]
    else:
        
        if( num_examples == -1 ):
            corpus = open(data_path, encoding='UTF-8').read().strip().split('\n')
        else:
            corpus = open(data_path, encoding='UTF-8').read().strip().split('\n')[:num_examples]
    
    return corpus
#}

#...........................................................................................................................................................
	
class data_partitioning(object):
#{
    def __init__(self):
	#{
        super(data_partitioning, self).__init__()

        self.dataset_size       = 0
        self.NUM_TRAIN_EXAMPLES = 0
        self.NUM_DEV_EXAMPLES   = 0
        self.NUM_TEST_EXAMPLES  = 0

        self.training_from      = 0
        self.training_to        = 0

        self.validate_from      = 0
        self.validate_to 		= 0

        self.test_from 			= 0
        self.test_to 			= 0

        self.TRAIN_PERCENTAGE   = 0
        self.VALID_PERCENTAGE   = 0
        self.TEST_PERCENTAGE    = 0

        self.dataset_split_info = ""
        self.load_dataset_spliter_info = ""	
	#}
	
    def __call__(self, dataset, dataset_split, num_examples=-1, seed=None):    
	#{	
        if(sum(dataset_split) > 1.0):
            raise Exception('dataset_split should not be more than 100%')
			
        corpus = read_corpus(dataset, num_examples)

        num_examples      = len(corpus)	
        self.dataset_size = len(corpus)

        if(seed != None):
            corpus.sort()    
            random.seed(seed)
            random.shuffle(corpus)
		
        self.NUM_TRAIN_EXAMPLES  = int( num_examples * dataset_split[0] )
        self.NUM_DEV_EXAMPLES    = int( num_examples * dataset_split[1] )
        self.NUM_TEST_EXAMPLES   = int( num_examples - (self.NUM_TRAIN_EXAMPLES + self.NUM_DEV_EXAMPLES) )

        self.TRAIN_PERCENTAGE    = ( self.NUM_TRAIN_EXAMPLES / num_examples )
        self.VALID_PERCENTAGE    = ( self.NUM_DEV_EXAMPLES   / num_examples )
        self.TEST_PERCENTAGE     = ( self.NUM_TEST_EXAMPLES  / num_examples )
		

        self.dataset_split_info = str(dataset_split) + "\n"\
        "********************************************************\n"\
        "NUM_EXAMPLES            : {}\n"\
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"\
        "NUM_TRAIN_EXAMPLES      : {}\t[{:.2%}]\n"\
        "--------------------------------------------------------\n"\
        "NUM_DEV_EXAMPLES        : {}\t[{:.2%}]\n"\
        "--------------------------------------------------------\n"\
        "NUM_TEST_EXAMPLES       : {}\t[{:.2%}]\n"\
        "********************************************************\n".format(num_examples, 
                                                                            self.NUM_TRAIN_EXAMPLES, self.TRAIN_PERCENTAGE,
                                                                            self.NUM_DEV_EXAMPLES,   self.VALID_PERCENTAGE,
                                                                            self.NUM_TEST_EXAMPLES,  self.TEST_PERCENTAGE)

        self.training_from  = 0
        self.training_to    = self.NUM_TRAIN_EXAMPLES

        self.validate_from  = self.training_to
        self.validate_to    = self.validate_from + self.NUM_DEV_EXAMPLES

        self.test_from      = self.validate_to
        self.test_to        = num_examples


        self.load_dataset_spliter_info = "\n"\
        "********************************************************\n"\
        "Training-set            : [{} : {}]\n"\
        "--------------------------------------------------------\n"\
        "Validation-set          : [{} : {}]\n"\
        "--------------------------------------------------------\n"\
        "Test-set                : [{} : {}]\n"\
        "--------------------------------------------------------\n"\
        "Data Usage              : {:.2%}\n"\
        "********************************************************\n".format(self.training_from, self.training_to,
                                                                            self.validate_from, self.validate_to,
                                                                            self.test_from,     self.test_to,
                                                                            (self.TRAIN_PERCENTAGE + self.VALID_PERCENTAGE + self.TEST_PERCENTAGE))

        training_dataset   = corpus[ self.training_from: self.training_to ] 																		
        validation_dataset = corpus[ self.validate_from: self.validate_to ]
        test_dataset       = corpus[ self.test_from:     self.test_to     ]

        return training_dataset, validation_dataset, test_dataset
	#}
#}

#...........................................................................................................................................................

def preprocess_dataset(dataset, 
                       inp_sent_casesensitive="", 
                       targ_sent_casesensitive="",
					   remove_digit=False, 
                       remove_punctuation=False, 
                       remove_stopwords=False,
                       inp_remove_digit=False, 
                       inp_remove_punctuation=False, 
                       inp_remove_stopwords=False,
					   targ_remove_digit=False, 
                       targ_remove_punctuation=False, 
                       targ_remove_stopwords=False,
					   start_tag="<start>", 
					   end_tag="<end>"):
#{
    sent_list = []
    for l in dataset:
    #{
        pair  = l.split('\t')

        targ_sent = preprocess_sentence(pair[0], targ_sent_casesensitive, (remove_digit != targ_remove_digit), (remove_punctuation != targ_remove_punctuation), (remove_stopwords != targ_remove_stopwords), start_tag, end_tag)
        inp_sent  = preprocess_sentence(pair[1], inp_sent_casesensitive,  (remove_digit != inp_remove_digit),  (remove_punctuation != inp_remove_punctuation),  (remove_stopwords != inp_remove_stopwords),  start_tag, end_tag)

        sent_list.append("{}\t{}".format(targ_sent, inp_sent))
    #}

    return sent_list
#}

#...........................................................................................................................................................

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [TL ,SL]

def create_dataset_pairs(dataset):
#{    
    sent_pairs = [[w for w in l.split('\t')]  for l in dataset]
    
    return sent_pairs
#}

#...........................................................................................................................................................	
	
# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,

class LanguageIndex():
#{    
    def __init__(self, lang, min_fw):
    #{
        self.lang          = lang
        self.word2idx      = {}
        self.idx2word      = {}

        self.vocab         = set()
        self.Allvocab      = set() 

        self.Allwords      = []
        self.running_words = []

        self.line = 1

        self.keep_frequency_more_than = min_fw
        self.create_vocab()
    #}

    def create_vocab(self):
    #{
        try:
        #{    
            tags = ["<start>", "<end>", "?", "!", ",", ".", "¿"]
                
            for phrase in self.lang:  

                self.line = self.line + 1
                self.vocab.update( split(phrase) )

                for word in split(phrase):
                #{    
                    self.Allwords.append(word)
                    
                    if( word not in tags ):
                        self.running_words.append(word)                        
                #}

            self.Allvocab = self.vocab 
            
            self.vocab = get_freqMoreThan(self.keep_frequency_more_than, self.Allwords)
            
            self.vocab.add( "<unk>")
            self.Allvocab.add( "<unk>")
            
            self.create_index()
        #}    
        except Exception as e:                
            print( "PHRASE EXCPTION {} \t {}: ".format(e, phrase) )
            raise e
	#}
    
    def create_index(self):
	#{
        try:
		#{

            self.vocab = sorted(self.vocab)

            self.word2idx['<pad>'] = 0  

            for index, word in enumerate(self.vocab):
                self.word2idx[word] = index + 1

            for word, index in self.word2idx.items():
                self.idx2word[index] = word
		#}
        except Exception as e:
            print("EXCEPTION : {} Error in line = {}".format(e, self.line) )	
	#}
#}

#...........................................................................................................................................................

def max_length(tensor):
    return max(len(t) for t in tensor)

#...........................................................................................................................................................

def min_length(tensor):
    return min(len(t) for t in tensor)
  
#...........................................................................................................................................................

def load_RAW_dataset(dataset, num_examples=-1, seed= None):
#{  
    corpus = read_corpus(dataset, num_examples)
    
    if(seed != None):
        corpus.sort()    
        random.seed(seed)
        random.shuffle(corpus)

    return corpus
#}

#...........................................................................................................................................................

"""
Prepare the dataset

IX-2p WANT BUTTER IX-2p	                 Do you want butter?
IX-1p NOT WANT READ+ IX-1p WANT DEPART	 I don't want to read, I want to leave


1. Add a *start* and *end* token to each sentence.
2. Clean the sentences by removing special characters.
3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
4. Pad each sentence to a maximum length.
"""

def load_dataset(dataset, inp_lang= None, targ_lang= None, Tx=None, Ty=None, seed= None, inp_min_fw=0, targ_min_fw=0, reverse_src_targ=False ):
#{    
    pairs = create_dataset_pairs(dataset)
	
    if(inp_lang == None and targ_lang == None):
    #{
        # index language using the class defined above 
        
        if reverse_src_targ:
            inp_lang  = LanguageIndex( (targ for src, targ in pairs), inp_min_fw )
            targ_lang = LanguageIndex( (src  for src, targ in pairs), targ_min_fw )
        else:
            inp_lang  = LanguageIndex( (src  for src, targ in pairs), inp_min_fw )
            targ_lang = LanguageIndex( (targ for src, targ in pairs), targ_min_fw )            
    #}
    
    
    # Vectorize the input and target languages    
    # Source sentences  
    #input_tensor  = [ [ get_word2idx(s, inp_lang) for s in split(tar) ] for src, tar in pairs ]

    unk_counter  = 0
    inp          = ""
    input_tensor = []
    for src, targ in pairs:
        
        if reverse_src_targ:
            inp = targ
        else:
            inp = src
            
        sent = []
        for s in split(inp):
            index , unk_counter = get_word2idx_OOV(s, inp_lang, unk_counter)
            
            sent.append(index)
        
        input_tensor.append(sent)
      
    input_OOV = unk_counter
    
    
    # Target sentences       
    #target_tensor = [ [ get_word2idx(s, targ_lang) for s in split(src) ] for src, tar in pairs ]
    
    unk_counter   = 0     
    out           = ""
    target_tensor = []
    for src, targ in pairs:
        
        if reverse_src_targ:
            out = src
        else:
            out = targ
            
        sent = []
        for s in split(out):
            index , unk_counter = get_word2idx_OOV(s, targ_lang, unk_counter)
            sent.append(index)
        
        target_tensor.append(sent)    
    
    target_OOV    = unk_counter
    
    unk_counter   = 0
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    
    min_length_inp, min_length_targ = min_length(input_tensor), min_length(target_tensor)
    max_length_inp, max_length_targ = max_length(input_tensor), max_length(target_tensor)
    
    if Tx is None:
        Tx = max_length_inp

    if Ty is None:         
        Ty = max_length_targ
        
    
    # Padding the input and output tensor to the maximum length
    input_tensor  = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=Tx, padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=Ty, padding='post')

    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ, min_length_inp, min_length_targ, input_OOV, target_OOV
#}

#...........................................................................................................................................................
	
def search_best_seed(path_to_file, number_examples, dataset_split, seed_from, seed_to):

    best_seed_input_dic = {}
    best_seed_targ_dic  = {}
    best_seed_ALL_dic   = {}
    
    best_input_OOV  = int(number_examples * (dataset_split[1] + dataset_split[2]))
    best_target_OOV = int(number_examples * (dataset_split[1] + dataset_split[2]))
    best_ALL_OOV    = best_input_OOV + best_target_OOV
    
    for i in range(seed_from, seed_to+1):
        
        print(i, end=", ")
        
        corpus = read_corpus(path_to_file, number_examples)

        corpus.sort()    
        random.seed(i)
        random.shuffle(corpus)

        
        _, _, _,\
        load_dataset_training_from, load_dataset_training_to,\
        load_dataset_validate_from, load_dataset_validate_to,\
        load_dataset_test_from,load_dataset_test_to,\
        _, _ = data_partitioning(number_examples, dataset_split)



        input_tensor_train, target_tensor_train,\
        inp_lang_train, targ_lang_train,\
        max_length_inp_train, max_length_targ_train,\
        min_length_inp_train, min_length_targ_train,\
        input_OOV_train, target_OOV_train = load_dataset(corpus, load_dataset_training_from , load_dataset_training_to)        
        
        
        
        input_tensor_test, target_tensor_test,\
        _, _,\
        max_length_inp_test, max_length_targ_test,\
        min_length_inp_test, min_length_targ_test,\
        input_OOV, target_OOV = load_dataset(corpus, load_dataset_validate_from, load_dataset_test_to, inp_lang_train, targ_lang_train)
        
        if(input_OOV <= best_input_OOV):
            best_input_OOV = input_OOV
            best_seed_input_dic[i]  = best_input_OOV

        if(target_OOV <= best_target_OOV):
            best_target_OOV = target_OOV
            best_seed_targ_dic[i]   = best_target_OOV

        ALL_OOV = input_OOV + target_OOV
        
        if(ALL_OOV <= best_ALL_OOV):
            best_ALL_OOV = ALL_OOV
            best_seed_ALL_dic[i] = best_ALL_OOV
        
        
        if(i%100 == 0):            
            seed_info = "\n{}\n{}\n{}\n\n".format(best_seed_input_dic, best_seed_targ_dic, best_seed_ALL_dic)

            print(seed_info)
            save_file("/content/drive/My Drive/" + MODEL_NAME + "_seed_search/seed_search.txt", seed_info, writeMode= "a")
                        
    return best_seed_input_dic, best_seed_targ_dic, best_seed_ALL_dic          		

#...........................................................................................................................................................