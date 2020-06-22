# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:35:40 2020

@author: mrpaolo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stopwords_list = stopwords.words('english')
import re, os
from nltk.tokenize import word_tokenize
import gensim
import string
import spacy
import time
from tqdm import tqdm
from slang import abbreviations
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer


tweets = pd.read_csv('../DATA_/tweets_engineered_4Sentiments_1.csv')

'''
Sparse matrix is a matrix with mostly zero values, storing only non-zero values
TBD: working with sparse matrices for NLP tasks
'''

# =================================================== REMOVE PUNCTUATION ============================================================
# Words with punctuations and special characters
line = '# RockyFire Update = > California Hwy . 20 closed in both directions due to Lake County fire - # CAfire # wildfires'

translate_table = dict((ord(char), None) for char in string.punctuation)   
# Drop all columns except leaned converted text
print(tweets.columns)
print(tweets.info())

tweets.drop('converted_text', axis=1, inplace=True)
tweets.drop('abbreviations_in_text', axis=1, inplace=True)
# ----------------------------------------------------------------------------------
tweets['converted_text'] = tweets['converted_text'].str.replace(r'#','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'<|>','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'=','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'_','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'@','') 
tweets['converted_text'] = tweets['converted_text'].str.replace(r'\|','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'(\. :)|(\.:)','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r';','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'`','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'\'\'','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'--','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'\\','')
tweets['converted_text'] = tweets['converted_text'].str.replace(r'(\./)','')
tweets['converted_text'].loc[362]
# Remove b-s at the beginning of the string text
tweets['converted_text'].loc[697]
tweets['converted_text'] = tweets['converted_text'].str.replace(r'^b','')

# Check if there's any comrades in text    
bratishka = tweets['clean_text'].str.contains('Comrade', case=False)
print(" Contains comrades: {0:f}".format(np.sum(bratishka)))
# Replacing all mentions with bratishka stuff
# Bratishka is Capital Letter because it should be mapped to proper noun
tweets['converted_text'] = tweets['converted_text'].str.replace(r'Bratishka','Comrade')
print(tweets['converted_text'].iloc[322])


tweets.to_csv('../DATA_/tweets_engineered_4Sentiments_1.csv', index = False)

# ========================================================== GET SENTIMENTS =========================================================
def get_polarity(txt):
    text_blob = TextBlob(txt)
    print(txt,'\nPolarity: ',text_blob.sentiment[0],'\n----------------------------------------')
    return text_blob.sentiment[0]

def get_subjectivity(txt):
    text_blob = TextBlob(txt)
    print(txt,'\nsubjectivity: ',text_blob.sentiment[1],'\n----------------------------------------')
    return text_blob.sentiment[1]

tweets['polarity']=np.nan
tweets['subjectivity']=np.nan

tweets['polarity'] = tweets['converted_text'].apply(get_polarity)
tweets['subjectivity'] = tweets['converted_text'].apply(get_subjectivity)

print(tweets.info())

# ------------------------------------------------------------------------------------------

# ========================================= TEXT SEQUENCE CREATION ============================================
# TBD: reduce vocab size to most common 10000/7000/5000 words and check again
vocab_size = 5000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(tweets['converted_text'])

# The .document_count gives us the number of documents used to build the vocabulary
# equal to the number of samples 
print(tokenizer.document_count)
print(tokenizer.num_words)
# number of unique words and symbols in the corpus
print(len(list(tokenizer.word_index))) #15010

sequences = tokenizer.texts_to_sequences(tweets['converted_text'])
#Let’s calculate the longest sequence length
maxlen = max([len(seq) for seq in sequences]) # 35
# pad all sequences to 35
X = pad_sequences(sequences, maxlen=maxlen)
print(X.shape)
y = tweets['target'].values
type(y)
print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

'''
TIP: Maybe Output_dim in embeddings layer and number of units in LSTM are related ???? 
'''
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=16, input_length=maxlen))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

h = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2) 

# Model is overfitting on validation data checking on test
loss, acc = model.evaluate(X_test, y_test, batch_size=32)
print(acc)# 0.7767
# 0.7867 --> with reduced vocabulary size, thus decreasing vocabulary, increases precision 

# printing the acc
dfhistory = pd.DataFrame(h.history)
dfhistory[['acc', 'val_acc']].plot(ylim=(-0.05, 1.05));

dfhistory[['loss', 'val_loss']].plot(ylim=(-0.05, 1.05));

model.summary()
'''
TIP: Model is overfitting on both training and testing data
less parameters  --> more accurate predictions
'''
# ================================== NLP WITH GLOVE ================================================
# https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert
# -------------------------------------- Creating glove twitter csv -------------------------------------
def create_corpus_new(df):
    corpus=[]
    for tweet in tqdm(tweets['converted_text']):
        words=[word.lower() for word in word_tokenize(tweet)]
        corpus.append(words)
    return corpus 

corpus=create_corpus_new(tweets)

glove_path = '/home/mrpaolo/Desktop/DATA SCIENCE/libs/glove/glove.twitter.27B.200d.txt'

embedding_dict={}
with open(glove_path,'r') as f:
    for line in tqdm(f):
        values=line.split()
        word = values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()
# 1193514it [02:33, 7764.16it/s]

MAX_LEN=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)
tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')

word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))

num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))
print(embedding_matrix.shape) # (15900, 100)

for word,i in tqdm(word_index.items()):
    if i < num_words:
        emb_vec=embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i]=emb_vec      


# ================================ Baseline Model with GloVe results ===========================

model=Sequential()


