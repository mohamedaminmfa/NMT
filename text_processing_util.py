import re
import os
import nltk
import numpy as np
import unicodedata


def split(s, sep=" "):
    return s.split(sep)

#...........................................................................................................................................................

def split2(s, sep=" ", comments=False, posix=False):
    lex = shlex.shlex(s, posix=posix)
    lex.whitespace_split = True
    lex.whitespace = sep

    if not comments:
        lex.commenters = ''

    return list(lex)

#...........................................................................................................................................................

def isDigit(str):    
    try:
        f = float(str)  
        return True
    
    except Exception as e:               
        return False  

#...........................................................................................................................................................

def removeDigit(str):
    
    clean_sen = []
    
    for s in split(str):
        if not isDigit(s):
            clean_sen.append(s)
    
    return " ".join(clean_sen)

#...........................................................................................................................................................

def remove_unwantedList(List, unwantedList):
    
    clean_list = []
    
    for l in List:
        if( l not in unwantedList):
            clean_list.append(v)
    
    return clean_list

#...........................................................................................................................................................

def stopWordRemoval(text, path_to_stopwords):
	stop_word_file = open(path_to_stopwords, "r")
	stop_word_list = split( stop_word_file.read(), "\n" )
	
	clean_sen = []
	
	words = split(text, " ")
	
	for w in words:
		if(w.lower() not in stop_word_list ):
			clean_sen.append(w)
			
	return " ".join(clean_sen)

#...........................................................................................................................................................

def stopWordRemoval(text, stop_word_list):
	
	clean_sen = []
	
	words = split(text, " ")
	
	for w in words:
		if(w.lower() not in stop_word_list ):
			clean_sen.append(w)
			
	return " ".join(clean_sen)

#...........................................................................................................................................................

def getFreqDist(wordsList): 
    
    fd = nltk.FreqDist(wordsList)
    return fd

#...........................................................................................................................................................

def get_freqEqual(num, wordsList):
  
    fd = nltk.FreqDist(wordsList)
    
    filter_words_list = set()
        
    for item in fd.items():
        if item[1] == num:
            filter_words_list.add(item[0])
    
    return filter_words_list

#...........................................................................................................................................................

def get_freqLessThan(num, wordsList):
  
    fd = nltk.FreqDist(wordsList)
    
    filter_words_list = set()
        
    for item in fd.items():
        if item[1] < num:
            filter_words_list.add(item[0])
    
    return filter_words_list

#...........................................................................................................................................................

def get_freqLessEqualThan(num, wordsList):
  
    fd = nltk.FreqDist(wordsList)
    
    filter_words_list = set()
        
    for item in fd.items():
        if item[1] <= num:
            filter_words_list.add(item[0])
    
    return filter_words_list
#...........................................................................................................................................................

def get_freqMoreThan(num, wordsList):
  
    fd = nltk.FreqDist(wordsList)
    
    filter_words_list = set()
        
    for item in fd.items():
        if item[1] > num:
            filter_words_list.add(item[0])
    
    return filter_words_list

#...........................................................................................................................................................

def get_freqMoreEqualThan(num, wordsList):
  
    fd = nltk.FreqDist(wordsList)
    
    filter_words_list = set()
        
    for item in fd.items():
        if item[1] >= num:
            filter_words_list.add(item[0])
    
    return filter_words_list

#...........................................................................................................................................................

def get_freqBetween(num1, num2, wordsList):
  
    if(num1 > num2):
        print("ERROR: num1 should be less than num2")
        return
    
    fd = nltk.FreqDist(wordsList)
    
    filter_words_list = set()
        
    for item in fd.items():
        if item[1] >= num1 and item[1] <=num2:
            filter_words_list.add(item[0])
    
    return filter_words_list

#...........................................................................................................................................................

def get_singletons(wordsList): 

    return get_freqEqual(1, wordsList)

#...........................................................................................................................................................

def get_singletons2(wordsList): 
    
    fd = nltk.FreqDist(wordsList)
    return fd.hapaxes()

#...........................................................................................................................................................

def get_doubletons(wordsList): 

    return get_freqEqual(2, wordsList)

#...........................................................................................................................................................

def remove_singletons(wordsList):
    
    return get_freqMoreThan(1, wordsList)

#...........................................................................................................................................................

def remove_doubletons(wordsList):
    
    return get_freqMoreThan(2, wordsList)

#...........................................................................................................................................................

def stemming(text):
	st = ISRIStemmer()
	stemmed_words = []
	
	words = split(text, " ")
	
	for w in words:
		stemmed_words.append(st.stem(w))
	
	return " ".join(stemmed_words)

#...........................................................................................................................................................

def lemmatizing(text):
    
    stemmer = nltk.stem.WordNetLemmatizer()
    #" ".join( stemmer.lemmatize(w) for w in text.split(" ") )

    stemmed_words = []

    words = split(text)

    for w in words:
        stemmed_words.append(stemmer.lemmatize(w))

    return " ".join(stemmed_words)

#...........................................................................................................................................................

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    
    latin_code = 'NFD'
    arabic_code = 'NFKD'
    return ''.join(c for c in unicodedata.normalize(latin_code , s) if unicodedata.category(c) != 'Mn')

#...........................................................................................................................................................

def custom_arabic_normalize(text):
    text = re.sub(r"[إأٱآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r'[^ا-ي ]', "", text)    
    
    text = re.sub(r'([ا]+)', "ا", text)

    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return text

#...........................................................................................................................................................

def clean_arabic_specialCharacter(w):
    
    w = custom_arabic_normalize(w)
    
    w = re.sub(r"([a-zA-Z0-9/%$@<>\'^+\\=#|~\؏\t?¿`!&*{\_}\-/\"»«×;،,:؛])","", w)
    w = re.sub(r"([][[])","", w)
    w = re.sub(r"([()])","", w)
    w = re.sub(r"([.,؟])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    return w
    
#...........................................................................................................................................................

def clean_tag(obj):
    
    if isinstance(obj, list):        
        sents_list =  []
        for l in obj:
            sent = []
            for s in l.split(" "):    
                
                if s.strip().lower() != "<start>" and s.strip().lower() != "<end>":
                    sent.append(s)
            sents_list.append(" ".join(sent))
        
        return sents_list
    
    else:      
        sent = []
        for i in obj.split(" "):

            if i.strip().lower() != "<start>" and i.strip().lower() != "<end>":
                sent.append(i)

        return " ".join(sent)
    
#...........................................................................................................................................................

def preprocess_sentence(w, casesensitive, remove_digit, remove_punctuation, remove_stopwords, start_tag="<start>", end_tag="<end>"):     
    
    if casesensitive.lower() == "lower":
        w = w.lower()
    elif casesensitive.lower() == "upper":
        w = w.upper()

    w = unicode_to_ascii(w.strip())

    #...........................................Stop Words...............................................................

    if remove_stopwords:
        stop_word_list = ["the", "a", "an", "of"]    
        w = stopWordRemoval(w, stop_word_list)   #Remove Stop Words

    #...........................................Space & others Matching............................................................

    #Replace more than one space or other punctuation with one
    w = re.sub(r'[" "]+', " ", w) 
    w = re.sub(r'["."]+', ".", w)
    w = re.sub(r'["?"]+', "?", w)
    w = re.sub(r'["!"]+', "!", w)
    w = re.sub(r'["_"]+', "_", w)
    w = re.sub(r'[","]+', ",", w)
    
    #........................................Punctuation Matching..........................................................
    
    if remove_punctuation:
        w = re.sub(r"([?!,.¿])", "", w)         #Remove Punctuation   
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    
    w = re.sub(r"([?!,¿])", r" \1 ", w)                   #Keep Punctuation and put space around it.  
       
    w = re.sub(r"(\()", r" \1", w)                        #Put space before (      ex: "(" to be " ("
    w = re.sub(r"(\))", r"\1 ", w)                        #Put space after  )      ex: ")" to be ") "
    
    w = re.sub(r"(?<=([(]))\s+", "", w)                   #Remove space after  (   ex: "( " to be "("
    w = re.sub(r"\s+(?=([)]))", "", w)                    #Remove space before )   ex: " )" to be ")"           
    
    
    w = re.sub(r"((?![0-9])\.(?![0-9]))", r" \1 ", w)     #Keep . and put space around it. and check for word arount it not digits.

    #Take space around string between double quote "xxxx" , ex: I love"ML" > I love "ML"
    w = re.sub(r'([\"].*[\"])', r' \1 ' , w) 
    
    w = re.sub(r'\s+(?=[^(\)]*\))', r'-' , w)             #Replace space inside () with -
     
    
    #.....................................................................................................................
  
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",",....etc)
    w = re.sub(r"[^a-zA-Z0-9?!,.¿\-:_'(+)\"#/\\<>]+", " ", w)
    
    #........................................Digits Matching.............................................................
    
    #w = re.sub(r"([0-9])",r" \1 ", w)                            #Remove Digits (Integer)
    #w = re.sub(r"((\d*\.)?\d+)", r" \1 ", w)                     #Remove Digits (Integer and Decimal)
    #w = re.sub(r"((([^a-zA-Z](\d*\.)?\d+)[^a-zA-Z]))", r" ", w)  #Remove Digits (Integer and Decimal) except digit between character like java2, j2ee, 1st, 8th
    
    if remove_digit:
        w = removeDigit(w)
    
    #.....................................................................................................................
    
    w = w.rstrip().strip()
    
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = start_tag + " " + w + " " + end_tag
    
    return w.rstrip().strip()