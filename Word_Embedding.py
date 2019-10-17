# -*- coding: utf-8 -*-
"""
@author: Noman Ashraf

Note!!!

This file contains the implementation of w2vec generation on pre trained google news model. 
Please do not consider it as a project. If you want to see complete applications, please contact me on email. 

"""

from sklearn.model_selection import train_test_split
import numpy
import numpy as np
import theano
import os
import os.path
import sys
import re, unicodedata
import pandas as pd
import csv
import _pickle as cPickle
from collections import defaultdict

from sklearn.manifold import TSNE
# Import adjustText, initialize list of texts
from adjustText import adjust_text
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from gensim import models
from gensim.models import KeyedVectors

from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import DataSetText as ds


import pre_processing as pp
pre_pro = pp.PreProcessing()

class WordEmbeddings(object):
    
    def generate_googlepretrained_w2v_doc_or_tweet_keywords(self, w2v_file , keywords, y_train, vectors_per_document= 1):
        w2vmodel = models.KeyedVectors.load_word2vec_format(w2v_file, binary=True, limit=100000)
        print("Word 2 Vector File Loaded!")        
    
        vector = w2vmodel['easy']
        print( "Shape of Vector:" + str(vector.shape))
        
        
        X_train_Vector = []
        for kl in keywords:
            vector_list = []
            for word in kl[0:vectors_per_document]:
                word = pre_pro.Clean_Text(word)
                if word in w2vmodel.vocab:
                    vector_list.append(w2vmodel[word])
                else:
                    vector_list.append(np.random.uniform(-0.1, 0.1, 300))
            
            X_train_Vector.append(vector_list)
        
        X = numpy.array(X_train_Vector)
        print( "length of Training Vectors" + str(len(X_train_Vector)))
        
        xTrain, xTest, yTrain, yTest = train_test_split(X, y_train, test_size = 0.2)
        return xTrain, xTest, yTrain, yTest
    
    
   
    
    
    def generate_tfidf_w2v_doc_keywords(self, w2v_file , keywords, y_train, Number_OF_Words, Number_OF_TF_IDF_Words ,Number_OF_Documents):
        w2vmodel = models.KeyedVectors.load_word2vec_format(w2v_file, binary=True, limit=100000)
        print("Word 2 Vector File Loaded!")   
        
        cv=CountVectorizer()
        word_count_vector=cv.fit_transform(keywords)
        print(word_count_vector.shape)
        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(word_count_vector)
        df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
        df_idf.sort_values(by=['idf_weights'], ascending=False) 
        count_vector=cv.transform(keywords)
        tf_idf_vector=tfidf_transformer.transform(count_vector)
        feature_names = cv.get_feature_names()  
        first_document_vector=tf_idf_vector[0]
        df = pd.DataFrame(first_document_vector.T.todense(),index=feature_names, columns=["tfidf"])
        df['Words'] = df.index.values
        #print(df.nlargest(100, ['tfidf']).sort_index(by=["tfidf"],ascending=False).iloc[:100, 1: ])
        
        documents_tf_idf = []
        for i in range(0,len(keywords)):
            first_document_vector=tf_idf_vector[i]
            df = pd.DataFrame(first_document_vector.T.todense(),index=feature_names, columns=["tfidf"])
            df['Words'] = df.index.values
            documents_tf_idf.append(df.nlargest(Number_OF_TF_IDF_Words, ['tfidf']).iloc[:Number_OF_TF_IDF_Words, 1: ].to_string(index = False))
            df.drop('Words', axis=1, inplace=True)
            #print(i)
        print(len(documents_tf_idf))
        
        
        count = 0
        X_train_Vector = []
        #remove size limit from function to load whole words 
        for data in documents_tf_idf:
            words = ''.join(data).split()
            print("Word 2 Vector File Loaded!")
            w2vmodel.vector_size
            vector = w2vmodel['easy']
            vector_list = [w2vmodel[word] for word in words if word in w2vmodel.vocab]
            words_filtered = [word for word in words if word in w2vmodel.vocab]
            word_vec_zip = zip(words_filtered, vector_list)
            word_vec_dict = dict(word_vec_zip)
            dfw2v = pd.DataFrame.from_dict(word_vec_dict, orient='index')
            print("\n--------------------------------------------\n")
            print(dfw2v.head(5))
            print("\n--------------------------------------------\n")        
            numpy_matrix = dfw2v.values[0:Number_OF_Words]
            veclist = numpy_matrix.tolist()
            X_train_Vector.append(veclist)
            print("Document Vectors Loaded:" + str(count))
            count = count + 1
        print( "length of Training Vectors" + str(len(X_train_Vector)))
        
        X = numpy.array(X_train_Vector)
        print( "length of Training Vectors" + str(len(X_train_Vector)))
        
        xTrain, xTest, yTrain, yTest = train_test_split(X, y_train, test_size = 0.2)
        return xTrain, xTest, yTrain, yTest
        
    
    
