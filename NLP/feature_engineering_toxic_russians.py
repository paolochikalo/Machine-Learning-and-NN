
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:30:58 2020

@author: mrpaolo
"""

import spacy
from string import punctuation
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS

import pandas as pd

import nltk
from nltk.stem.snowball import SnowballStemmer 
from nltk.corpus import stopwords

import stanza
from spacy_stanza import StanzaLanguage

#nltk.download("stopwords")
#stanza.download('ru')  # will take a while
#russian_stopwords = stopwords.words("russian")
russian_stopwords = spacy.lang.ru.stop_words.STOP_WORDS

# ================================================== EXAMPLE ===================================================
text = "Не ветер, а какой-то ураган!"
nlp = Russian()
doc = nlp(text)
russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
doc = nlp(text)
print([token.text for token in doc])
# =============================================================================================================

toxic_rus = pd.read_csv("./__DATA/NLP_Datasets/toxic_russian.csv")

toxic_rus.head()
toxic_rus.info()

# Removing punctuation
toxic_rus['comment'] =  toxic_rus['comment'].str.replace(r'[.,!?<>-]', '')
toxic_rus['comment'] =  toxic_rus['comment'].str.replace("\n"," ")
toxic_rus['comment'] =  toxic_rus['comment'].str.replace("\t"," ")
toxic_rus['comment'] =  toxic_rus['comment'].str.replace("(","")
toxic_rus['comment'] =  toxic_rus['comment'].str.replace(")","")

toxic_rus['comment'].head()

nlp = Russian()
russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')

def tokenize(txt):
    lda_tokens = []
    tokens = nlp(txt)
    for token in tokens:
        #if token not in string.punctuation:
        if token.is_stop == False:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
    return lda_tokens

toxic_rus['tokens'] = toxic_rus['comment'].apply(tokenize)

toxic_rus['tokens'].head()

toxic_rus.to_csv("toxic_rus_featured.csv", index=False, sep=',')

# ============================================== Stemming AND Lemmatizing TEXT ==================================================

snlp = stanza.Pipeline(lang="ru", processors= 'tokenize , pos , lemma', use_gpu=True, pos_batch_size=500)
nlp = StanzaLanguage(snlp)

text = "Хохлы это отдушина затюканого россиянина мол вон а у хохлов еще хуже Если бы хохлов не было кисель их бы придумал."

pushkin = '''У лукоморья дуб зелёный;
Златая цепь на дубе том:
И днём и ночью кот учёный
Всё ходит по цепи кругом;
Идёт направо - песнь заводит,
Налево - сказку говорит.'''
doc = nlp(text)
sdoc = snlp(text)

pushkin_doc = nlp(pushkin)

stemmer = SnowballStemmer("russian")

stemmed = []
for word in text.split(" "):
    print(word, "-->", stemmer.stem(word))
    stemmed.append(stemmer.stem(word)) 
print(" ".join(stemmed))
    
lemmatized = []
for token in pushkin_doc:
    #print(token, "-->", token.lemma_, "-->", token.ent_type_, "-->", token.pos_, "-->", token.dep_)
    print(token, "-->", token.pos_)
    lemmatized.append(token.lemma_)
print(" ".join(lemmatized))
    
    
for sent in sdoc.sentences:
    for word in sent.words:
        print(word.text, ":", word.lemma, "-->", word.upos, "-->", word.deprel)

# =================================================== Buildiung BOW model ===============================================================

# ------------------------------------- TEST -----------------------------------------
text = "Хохлы это отдушина затюканого россиянина мол вон а у хохлов еще хуже Если бы хохлов не было кисель их бы придумал."
text_2 = "Страницу обнови, дебил. Это тоже не оскорбление, а доказанный факт - не-дебил про себя во множественном числе писать не будет. Или мы в тебя верим - это ты и твои воображаемые друзья?"
from sklearn.feature_extraction.text import CountVectorizer    

vectorizer = CountVectorizer()

bow_matrix = vectorizer.fit_transform(pd.Series(text_2))

print(bow_matrix.shape)
print(vectorizer.get_feature_names())
# -------------------------------------------------------------------------------------

toxic_rus['comment'].head()

'''

BOW --> sparse matrice where words are kept as features (columns) rows are actual quotes from 
the document stored as indeces of corresponding string 

'''

# 1. BOW CountVectorizer without stemming or lemmatizing
BOW_1 = vectorizer.fit_transform(toxic_rus['comment'])

print(BOW_1.shape) # (14412, 68423)

# 1.1 BOW CountVectorizer without stemming or lemmatizing stopwords removed

def remove_stopwords(comment):
    no_stop_words = []
    for word in comment.split(" "):
        if word not in russian_stopwords:
            no_stop_words.append(word) 
    return " ".join(no_stop_words)

print(pd.Series(text_2).apply(remove_stopwords))

toxic_rus['comment_no_stop'] = toxic_rus['comment'].apply(remove_stopwords)

BOW_1_1 = vectorizer.fit_transform(toxic_rus['comment_no_stop'])

print(BOW_1_1.shape) # (14412, 68401)

# 2. BOW CountVectorizer with stemming

def stem_series(comment):
    stemmed = []
    for word in comment.split(" "):
        #print(word, "-->", stemmer.stem(word))
        stemmed.append(stemmer.stem(word)) 
    return " ".join(stemmed)

print(pd.Series(text_2).apply(stem_series))

toxic_rus['stemmed'] = toxic_rus["comment"].apply(stem_series)

print(toxic_rus['stemmed'].head())
print(toxic_rus['comment'].head())

BOW_2 = vectorizer.fit_transform(toxic_rus['stemmed'])
print(BOW_2.shape) # (14412, 52754)

# 3. BOW CountVectorizer with lemmatizing --> TOO SLOW

snlp = stanza.Pipeline(lang="ru")
nlp = StanzaLanguage(snlp)

doc = nlp(text_2)
sdoc = snlp(text)

def lemmatize_series(comment):
    
    doc = nlp(comment)
    
    lemmatized = []
    for token in doc:
#        print(token, "-->", token.lemma_, "-->", token.ent_type_, "-->", token.pos_, "-->", token.dep_)
        lemmatized.append(token.lemma_)
    return " ".join(lemmatized)

toxic_rus['lemmatized'] = toxic_rus["comment"].apply(lemmatize_series)
print(toxic_rus['lemmatized'].head())

BOW_3 = vectorizer.fit_transform(toxic_rus['lemmatized'])
print(BOW_3.shape) # 

# ============================================= PICKLE/UNPICKLE DATA ===============================================
import pickle

with open("./__DATA/NLP_Datasets/toxic_main.pickle", 'wb') as toxic:
    pickle.dump(toxic_rus, toxic)
    
with open("./__DATA/NLP_Datasets/BOW_1.pickle", 'rb') as bow:
    bow_no_lemmas = pickle.load(bow)

pushkin_doc
type(BOW_1)
type(bow_no_lemmas)

# ===============================================================================================================

bow_no_lemmas = pd.read_pickle("./__DATA/NLP_Datasets/BOW_1.pickle")

# Mapping feature indices with feature names

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(BOW_1.toarray())


# Map the column names to vocabulary 
bow_df.columns = vectorizer.get_feature_names()

# Print hwo many times specific word is met in the corpus
print(bow_df["хохол"].sum(axis=0))



# ============================================= TOPIC IDENTIFICATION =============================================== 

# 1. first select all nouns and adjectives from the comments

# NOUN --> token.pos_ == NOUN
# ADJECTIVE --> token.pos_ == ADJ

NOUNS = []
ADJECTIVES = []

for token in pushkin_doc:
    
    if token.pos_ == "NOUN":
        NOUNS.append(str(token))
    elif token.pos_ == "ADJ":
        ADJECTIVES.append(str(token))

print(pushkin_doc)        
print("NOUNS:", " ".join(NOUNS))
print("ADJECTIVES:", " ".join(ADJECTIVES))

my_toxic = pd.DataFrame(toxic_rus["comment"].iloc[0:100])

my_toxic.head()

snlp = stanza.Pipeline(lang="ru")
nlp = StanzaLanguage(snlp)

def is_noun(text):
    doc = nlp(text)    
    NOUNS = []
    for token in doc:
        if token.pos_ == "NOUN":
            NOUNS.append(str(token).lower())
    NOUNS = set(NOUNS)
    print(doc,"NOUNS:", NOUNS)
    return " ".join(NOUNS)
    
def is_adjective(text):
    doc = nlp(text)    
    ADJECTIVES = []
    for token in doc:
        if token.pos_ == "ADJ":
            ADJECTIVES.append(str(token).lower())
    ADJECTIVES = set(ADJECTIVES)
    #print(doc,"ADJECTIVES:", ADJECTIVES)
    return " ".join(ADJECTIVES)

import time

start = time.time()
print(start)
my_toxic["nouns"] = my_toxic["comment"].apply(is_noun)

end = time.time()
print(end - start)

print(my_toxic.dtypes)
print(my_toxic.memory_usage())


my_toxic["comment"].head()


