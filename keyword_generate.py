# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:11:36 2019

@author: Noman
"""

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
# Import adjustText, initialize list of texts
from adjustText import adjust_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import itertools
import re
import os
from keras import models
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,LSTM, Dropout, Flatten
from keras.layers import Conv1D,Conv2D, MaxPooling2D, MaxPooling1D
from keras import backend as K
from keras.optimizers import SGD,RMSprop,Adam
from keras.regularizers import l2

from keras.callbacks import CSVLogger
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import preprocessor as p

import nltk
from nltk.corpus import stopwords
stopword = stopwords.words('english')

#Seed Random Numbers with the TensorFlow Backend
#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)


import natsort 
import pre_processing as pp
import DataSetText as ds
pre_pro = pp.PreProcessing()
import Word_Embedding as we
import timeit

dstext = ds.DataSet_Text_Conversion()
w2v = we.WordEmbeddings()


Number_OF_Documents = 290


#read excel file for emoticons
excelpath = "F:\Codes\\Depression_Paper_Time_Frame\\word_emoti.xlsx"
df= dstext.Load_DataSet_From_EXCEL(excelpath , False, ['Words'], False)
#emoticons = dstext.Load_DataSet_From_EXCEL(excelpath , False, [0,15], False)
emoticons = df['Words'].tolist()
print( "Emoticons words:" + str(len(emoticons)))





directory = "F:\Codes\\Depression_Paper_Time_Frame\\code\\Data\\"


filesnames = []
try:
            #exists function work for both file and directory
    if os.path.exists(directory):
        print("\n\nDirectory Discovered!\n\n")
        filesnames = os.listdir(directory)
    else:
        print("\n\nDirectory Not Found!\n\n")
except FileNotFoundError:
    print("Directory not Found Error!")
finally:
    print("\nTotal Files:" + str(len(filesnames)))
    filesnames = natsort.natsorted(filesnames) 
    print(filesnames[0:10])
        

stopwords = ["still","would","that","want","word","ok","in","was","your","these","just","will","video","yeah","cant","there","they","make","come","have","need","from","dont","when","some","know","with","what","your","that","Words","\'","http","httpst","a","aa","anything","ago","ask","act","ang","amp","actually","accurate","accurate","account","i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"]        
suicide_words = ["good","fear" ,"happy" ,"hope" ,"tragedy" ,"dissatisfaction" ,"shock" ,"shcoking" ,"stress" ,"exhausted" ,"anxiety" ,"anger" ,"suffering" ,"despair" ,"pain" ,"hurt" ,"worry" ,"heart" ,"coflict" ,"suspicion" ,"blame" ,"crisis" ,"feel" ,"sorry" ,"suicide" ,"depression" ,"sad","upset" ,"lonely" ,"mad" ,"hate" ,"happy" ,"tired" ,"frustrated" ,"crazy" ,"like","irritaed"]
#https://www.researchgate.net/figure/Emotional-words-most-frequently-associated-with-suicide-This-diagram-shows-the-words_fig3_281634366



documents_dictionary = []
documents_sentences_corpus = []
count = 0
for filename in filesnames:
    print("File Started:" + str(count))
    keywords_dictionary = []
    sen_dictionary = []
    path = "F:\Codes\\Depression_Paper_Time_Frame\\code\\Data\\" + filename
    #read csv file
    df = pd.read_csv(path)
    #convert date according to the requirement
    df = df[pd.notnull(df['text'])]
    #df['Tweets'] = df['text']
    #print(df['Tweets'])
    
    #df['Date'] = pd.to_datetime(df['created_at']).dt.date
    #print(df['Date'])
    
    # Create new columns
    #df['day'] = pd.to_datetime(df['created_at']).dt.day
    #df['month'] = pd.to_datetime(df['created_at']).dt.month
    #df['year'] = pd.to_datetime(df['created_at']).dt.year
    #print(df.shape)
    #tcount = 0 
    start = timeit.timeit()
    for index,row in df.iterrows():
        #print("Tweet Processed:" + str(tcount))
        #print(row['text'])        
        #text = pre_pro.Text_Cleaning(row['text'])
        #print(text)
        text = row['text']
        text = text.replace("b\'"," ")
        
        sentence = pre_pro.Clean_Text(text) 
        #print(sentence)
        #print("\n")
        text = pre_pro.Replace_Worlds(text)
        text = pre_pro.Replace_Repeating_Character(text)
        text, words_ = pre_pro.Correct_Words(sentence)  #it takes alot of time
        #print(text)
        
        emoti_sen = pre_pro.Emotionally_Words_Sentences(emoticons, text)
        #print(emoti_sen)
        
        if emoti_sen == True:
            #print(text)
           
            
            #words_list = pre_pro.Pick_Known_Words(text) 
            #print(words_list)
              
            #text = ' '.join(text)
            #print(text)


            #for s in stopwords:
            #    text = re.sub(r'\b' + s + r'\b'  , ' ', text)   
            
            #store sentences
            text = text.replace("rt"," ")
            sen_dictionary.append(text)
        
            
            #list_of_words = set(text.split(" "))
            list_of_words = words_
            
            #print(list_of_words)
            
            doc_words = [word for word in list_of_words if len(word) >= 3]
            #print(doc_words)  
            keywords_dictionary.append(doc_words)
            #tcount = tcount + 1   
            
            
            
    end = timeit.timeit()
    print("Time:" + str(end - start))
    keywords_list_single = [item for sublist in keywords_dictionary for item in sublist]
    documents_dictionary.append(keywords_list_single)    
    documents_sentences_corpus.append(sen_dictionary)
    keywords_dictionary.clear()      
    
    print ("File number Read Successfully!:" + str(count + 1))
    count = count + 1                          
 
print(len(documents_dictionary))     
#print(documents_dictionary[0])



print(type(documents_dictionary))

#file saved with keywords
#with open('F:\Codes\\Suicide_Depression_Paper\\code\\listkeywords.txt', 'w') as filehandle:
#    for item in documents_dictionary:
#        filehandle.write("%s\n" % item)


import csv
with open('F:\Codes\\Depression_Paper_Time_Frame\\code\\listkeywords.csv', mode='w', newline='', encoding='utf-8') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(documents_dictionary)

#file saved with keywords
#with open('F:\Codes\\Suicide_Depression_Paper\\code\\listsentences.txt', mode='w', newline='', encoding='utf-8') as myfile:
#    for item in documents_sentences_corpus:
#        filehandle.write("%s\n" % item)

import csv
with open('F:\Codes\\Depression_Paper_Time_Frame\\code\\listsentences.csv', mode='w', newline='', encoding='utf-8') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(documents_sentences_corpus)

def list_flatten(l):
    result = list()
    for item in l:
        if isinstance(item, (list, tuple)):
            result.extend(item)
        else:
            result.append(item)
    return result


sentences_ = []
for doc_sen in documents_sentences_corpus:
    sentences_list = list_flatten(doc_sen)    
    result = ''.join(sentences_list[0:500])
    #print(result)
    sentences_.append(result)

#print(sentences_[0])

import csv
with open('F:\Codes\\Depression_Paper_Time_Frame\\code\\listsentences_scatter.csv', mode='w', newline='', encoding='utf-8') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(sentences_)


print("Successfully Written!")