# -*- coding: utf-8 -*-
"""
Created on Thu May 23 04:59:21 2019

@author: Fikri Rozan Imadudin

Menggunakan modul nltk stowords, WordNetLemmatizer, PorterStemmer
"""
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
import pandas as pd
from string import punctuation
import string
import re
from joblib import Parallel, delayed


nj= -1 #-1 menggunakan semua cpu

def clean_kalimat(x):
    x = x.lower()
    remove_punc = str.maketrans('', '', punctuation)
    x = x.translate(remove_punc)
    remove_digits = str.maketrans('', '', string.digits)
    x = x.translate(remove_digits)
    x = re.sub(r'\b(\w{1,3})\b','',x)
    remove_whitespace = str.maketrans('','', string.whitespace[1:])
    x = x.translate(remove_whitespace)
    x = x.replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').replace(' '*1, ' ').strip()
    return x

#cleaner lebih cepat jika diparalelkan
def cleaner(X):
    X = pd.Series(Parallel(n_jobs=nj)(X.apply(lambda x: delayed(clean_kalimat)(x))))
    return X

#Tidak bagus jika diparalel
def tokenisasi(X):
    import re
    WORD = re.compile(r'\w+')
    X = pd.Series(X.apply(lambda x: WORD.findall(x)))
    return X

#tokenisasi nltk cepat jika di paralalel
#def tokenisasi(X):
#    X = pd.Series(Parallel(n_jobs=nj)(X.apply(lambda x: delayed(word_tokenize)(x))))
#   return X

def stopwordsremove(X):
    # Menghapus kata Umum di dalam corpus
    stopWords = set(stopwords.words('english'))
    X = X.apply(lambda x: [item for item in x if item not in stopWords])
    return X

def lemmatizing(X):
    # Menyederhanakan kata menggunakan Lemmatizing
    wordnet = WordNetLemmatizer()
    X = X.apply(lambda x: [wordnet.lemmatize(y) for y in x])
    return X

def stemming(X):
    # Menyederhanakan kata menggunakan Stemming
    ps = PorterStemmer()
    X = X.apply(lambda x: [ps.stem(y) for y in x])
    return X

def preprocessing(X):
    '''
    Preprocessing
    Mengolah data mentah yang dimasukan dengan tahap sebagai berikut:
    1. Menghilangkan tanda baca
    2. Menghilangkan angka
    3. Menghilangkan kata yang kurang dari 3 huruf
    4. Membuat huruf Kapital menjadi huruf kecil
    5. Tokenisasi
    6. Menghilangkat Kalimat Umum/StopWords
    7. Menyederhanakan kalimat menggunakan Lemmatizing
    8. Menyederhanakan kalimat menggunakan Stemming
    
    Parameters
    ----------
    X : berupa corpus
    Returns
    -------
    X : berupa corpus yang telah tertokenisasi
    '''
    X = cleaner(X)
    #  Menjadikan corpus token untuk setiap kata di dalam dokumen
    X = tokenisasi(X)
    print("T")
    # Menghapus kata Umum di dalam corpus
    X = stopwordsremove(X)
    print("S")
    # Menyederhanakan kata menggunakan Lemmatizing
    X = lemmatizing(X)
    print("L")
    # Menyederhanakan kata menggunakan Stemming
    X = stemming(X)
    print("S")
    print("Preprocessing Selesai")
    return X 
    
def word_count(data):
    c = Counter()
    for line in data:
        c.update(line.strip().split(' '))
    return c
    
def document_term_freq(X):
    '''
    Document Term Frequency
    Dokumen term frequency berfungsi untuk mengetahui frekuensi dari kata di setiap dokumen
    
    Parameters
    ----------
    X : berupa corpus yang telah di Preprocessing berbentuk token
    Returns
    -------
    term_freq : berupa dictionary yang berisi frequensi kata di setiap dokumen
    
    References
    ----------
    [1] Park, H., Kwon, S., & Kwon, H. C. (2010, June). Complete gini-index text (git) feature-selection algorithm for text classification. 
    In The 2nd International Conference on Software Engineering and Data Mining (pp. 366-371). IEEE.
    
    '''
        
    x = X.tolist()
    counters = [ word_count(X) for X in x]
    DF = []
    for i in range(len(counters)):
        DF.append(dict(counters[i]))
    return DF

def label_ke_numerik(y):
    y=pd.factorize(y)
    y=y[0]
    return y





        