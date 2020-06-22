180р # -*- coding: utf-8 -*-
"""
https://www.kaggle.com/c/nlp-getting-started/overview
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stopwords_list = stopwords.words('english')
import re
from nltk.tokenize import word_tokenize
import gensim
import string
import spacy
import time
from slang import abbreviations
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from keras.optimizers import Adam

"""
    For your information, there is research that indicates that fake news articles 
    tend to have longer titles! Therefore, even extremely basic features 
    such as character counts can prove to be very useful in certain applications.
    
    ALWAYS: check the text in Excel for NLP tasks 
    ALWAYS: SUBSTITUTIONS SHOULD  CONTAIN RIGHT AND LEFT WHITESPACES
    ALWAYS: CONVERT ALL OF THE CLEANUP AND PREPROCESSING STEPS INTO PROCEDURES (IN PERSPECTIVE INTO PIPELINES)
    
QUESTIONS:
    1. Add meta features: punctuation_count, unique_word_count, stop_word_count ...
    2. How using number of hashtags influences Real/Fake? --> what if person wants to hype --> 
    hashtags are present in smaller amount of samples 
    3. Compare it to known (Real / Fake ) distribution like russian trolls tweets
    4. disaster words like: fire, disaster, tragedy, hurricane. etc influences Real/Fake??? -->
        try to use it in one feature (i.e num of disaster words in a tweet) or multiple features
    5. Number of URL's and Real/Fake??? 
    6. Substitute for @mention --> Bratishka should be mapped to proper noun (PROPN) 
"""

# train_tweets = pd.read_csv('../DATA_/train.csv')
# test_tweets = pd.read_csv('test.csv')
# tweets=train_tweets.copy()
#tweets = pd.read_csv('../DATA_/tweets_engineered.csv')

tweets = pd.read_csv('../DATA_/tweets_engineered_1.csv')

print('There are {} rows and {} cols in train'.format(train_tweets.shape[0],train_tweets.shape[1]))
print('There are {} rows and {} cols in test'.format(test_tweets.shape[0],test_tweets.shape[1]))
print(tweets.info())
print(test_tweets.describe())

# Counting unique values in the dataframe:
# nunique --> distinct values
# count --> only non-null values
print(train_tweets.agg(['nunique', 'count', 'size']))
# Count null values
count_null = lambda df: df.isnull().sum()
train_tweets.apply(count_null)

# top 20 locations
print(train_tweets['location'].value_counts().sort_values(ascending=False).head(20))
# Distribution of the target variables
x = train_tweets.target.value_counts()
sns.barplot(x.index, x)
plt.gca().set_ylabel('samples')
plt.show()



###------------------------------------------
# Add text len column
tweets['text_len'] =  tweets['text'].str.len()
tweets['clean_text'] = tweets['text']

# Distribution of test len
tweets['text_len'].hist()
plt.title("Text Len Distribution")
plt.show()
'''
TBD: Distribution is skewed to the right --> what does it mean???
'''



# =============================== Replacing all new lines tabs & symbols===================================
tweets['clean_text'] = tweets['clean_text'].str.replace("\n"," ")
tweets['clean_text'] = tweets['clean_text'].str.replace("\t"," ")
# Replace & with and 
tweets['clean_text'] = tweets['clean_text'].str.replace("&amp;"," and ")

# ========================================== EXTRACT AND REMOVE URL's ========================================================
#TBD: MAKE FUNCTION
from urlextract import URLExtract
extractor = URLExtract()
# Adding more stop chars in case of parentesses near URL's
stop_chars = list(extractor.get_stop_chars_right())
stop_chars.append(')')
extractor.set_stop_chars_right(set(stop_chars))
     
tweets['url_count'] = 0
for i,t in enumerate(tweets['clean_text']):
    if extractor.has_urls(t):
        urls = extractor.find_urls(t)
        # Add URL count feature       
        tweets['url_count'].iloc[i] = len(urls)
        print(i,len(urls))
        # Remove url's from text        
        line = t
        for item in urls:
            line = re.sub(item,'', line)
        tweets['clean_text'].iloc[i] = line

print(tweets.info())
print(tweets.describe())
print(tweets['clean_text'].iloc[1060])

# ============================================= HASHTAGS AND MENTIONS COUNT =============================================
# counting hashtags and mentions
def hashtags_count(words):
    hashtags = [word for word in words if word.startswith('#')]
    return len(hashtags)

tweets['hashtags_count'] = tweets['clean_text'].str.split().apply(hashtags_count)
tweets['hashtags_count'].hist(bins=20)
plt.title("Hashtags count distribution")
plt.show()

ment_re = re.compile('([@][\w_-]+)')

def count_mentions(text):
    # Return number of mentions
    mentions = ment_re.findall(text)
    return len(mentions)

def find_mentions(text):
    # Return mentions or empty string
    mentions = ment_re.findall(text)
    if len(mentions) > 0:    
        return mentions
    else:
        return np.nan

# Create a feature mention_count and display distribution @DataCamp
tweets['mentions_count'] = tweets['clean_text'].apply(count_mentions)
tweets['mentions_count'].hist(bins=20)
plt.title('Mention count distribution')
plt.show()

tweets[['clean_text','mentions_count']].iloc[614]

tweets['mentions'] = tweets['clean_text'].apply(find_mentions)
print(tweets.info())

print("Mentions Missing: {0:f}".format(tweets['mentions_count'].loc[tweets['mentions_count']==0].count()/
      tweets['mentions_count'].shape[0]))# 74%

print("Avg text len with #:",tweets[tweets.num_hashtags > 0]['text_len'].mean())
print("Avg text len with @:",tweets[tweets.mention_count > 0]['text_len'].mean())
print("Avg text len:",tweets['text_len'].mean())
'''
@ doesn't influence text len. # DO!
'''
print(tweets.info())
# TBD: Correlation plot of # and @

# Check if there's any bratishkas in text    
bratishka = tweets['clean_text'].str.contains('bratishka', case=False)
print(" Contains bratishka: {0:f}".format(np.sum(bratishka)))
# Replacing all mentions with bratishka stuff
# Bratishka is Capital Letter because it should be mapped to proper noun
tweets['clean_text'] = tweets['clean_text'].str.replace(r'([@][\w_-]+)',' Bratishka ')
print(tweets['clean_text'].iloc[322])

# ======================================= HTML tags cleaning ==============================================
# Still there are emojiis and other stuff
from bs4 import BeautifulSoup

# SEARCH FOR ALL encoded SYMBOLS
amp=tweets['clean_text'][tweets['clean_text'].str.contains('&', case=False)]
print(" Contains &: {}".format(len(amp)))

decode_html = lambda txt: BeautifulSoup(txt,'lxml').text
tweets['clean_text'] = tweets['clean_text'].apply(decode_html)

# ======================================== CLEANING ASCII CODES ======================================
# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
# -----------------------------------------------------------------------------------------------------------------------------
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))
# -----------------------------------------------------------------------------------------------------------------------------


# encode garbage as ascii symbols
tweets = pd.read_csv('../DATA_/tweets_engineered_1.csv')

sample = tweets.iloc[[152, 123, 38, 52, 57]].copy()
sample['clean_text']

ascii_re = re.compile('(&#\d+;)')
find_ascii = lambda txt: ascii_re.findall(txt)

for i,t in enumerate(tweets['clean_text']):
    line = str(t.encode('ascii', 'xmlcharrefreplace'))
    garbage = find_ascii(line) 
    if len(garbage) > 0:
        #line = line.replase(r'b\'','')
        line = re.sub(r'b\'','', line)
        print(line)
        for item in garbage:
            line = re.sub(item,' ', line)
            print(i ,' replaced: ', item)
        print(line)
        tweets['clean_text'].iloc[i] = line


tweets['clean_text'] = tweets['clean_text'].str.replace(r'b\'','')
tweets['clean_text'] = tweets['clean_text'].str.replace(r'(&#\d+;)',' ')
print(tweets['clean_text'].iloc[57])

# ========================================== SLANG ABBREVIATIONS ==============================================
# TBD:

def convert_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

def get_abbrev(text):    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() in abbreviations.keys()]
    text = ' '.join(tokens)
    return text

def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens=[convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text

tweets['converted_text'] = tweets['clean_text'].apply(lambda x: convert_abbrev_in_text(x))
tweets['abbreviations_in_text'] = tweets['clean_text'].apply(lambda x: get_abbrev(x))


# TBD test dataset cleanup
#test['converted_text'] = test['clean_text'].apply(lambda x: convert_abbrev_in_text(x))
#test['abbreviations_in_text'] = test['clean_text'].apply(lambda x: get_abbrev(x))

# ================================= REMOVE PUNCTUATION and ADD SOME COLUMNS  =======================================
tweets['bald_text'] = np.nan
print(tweets.info())
# Remove hastags and mentions before counting punctuation
tweets['bald_text'] = tweets['clean_text'].str.replace(r'#', '')
print(tweets['bald_text'].iloc[5], tweets['punctuation_count'].iloc[5])
tweets['bald_text'] = tweets['bald_text'].str.replace(r'@', '')
tweets['punctuation_count']=np.nan      
tweets['punctuation_count'] = tweets['bald_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
translate_table = dict((ord(char), None) for char in string.punctuation)   
remove_punctuation = lambda line: line.translate(translate_table)

tweets['bald_text'] = tweets['clean_text'].apply(remove_punctuation)
#remove all b's at the beginning of the line
tweets['bald_text'] = tweets['bald_text'].str.replace(r'(^b )', '')

# add mean word length column
tweets['mean_word_length'] = tweets['bald_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# =========================================== STOP WORDS and UNIQUE WORDS ==================================================
# WARNING: COUNTING STOP WORDS AND UNIQUE WORDS IS DONE AFTER SLANG CLEANUP
from spacy.lang.en.stop_words import STOP_WORDS

#Use STOP_WORDS.update(['@','#']) to add more stopwords

#nlp = spacy.load('en_core_web_sm') 
#def remove_stopwords_spacy(txt):    
#    doc = nlp(txt)
#    tokens = [token.text for token in doc]
#    filtered = []
#    for word in tokens:
#        lexeme = nlp.vocab[word]
#        if not lexeme.is_stop:
#            filtered.append(word)
#    text = ' '.join(filtered)
#    return text
#    
#tweets['bald_text'] = tweets['bald_text'].apply(remove_stopwords_spacy)
#tweets.to_csv('../DATA_/tweets_engineered_4Sentiments.csv', index = False)

# number of stop words in each line
tweets['stop_words_count'] = tweets['bald_text'].apply(
        lambda x: len([w for w in str(x).lower().split() if w in STOP_WORDS]))
#number of non-repetitive words in each line
tweets['unique_words_count'] = tweets['bald_text'].apply(lambda x: len(set(str(x).split())))

# -----------------------------------------------------------------------------------------
tweets.to_csv('../DATA_/tweets_engineered_2.csv', index = False)



# ================================== COUNTING DISASTERS ======================================
#TBD: stemming/lemmatizing before
fire=tweets['clean_text'].str.contains('fire', case=False)
print(" Contains fire: {0:f}".format(np.sum(fire)/tweets.shape[0]))
disaster = tweets['clean_text'].str.contains('disaster', case=False)
print(" Contains disaster: {0:f}".format(np.sum(disaster)/tweets.shape[0]))
earthquake = tweets['clean_text'].str.contains('earthquake', case=False)
print(" Contains earthquake: {0:f}".format(np.sum(earthquake)/tweets.shape[0]))
tragedy = tweets['clean_text'].str.contains('tragedy', case=False)
print(" Contains tragedy: {0:f}".format(np.sum(tragedy)/tweets.shape[0]))
hurricane = tweets['clean_text'].str.contains('hurricane', case=False)
print(" Contains hurricane: {0:f}".format(np.sum(hurricane)/tweets.shape[0]))
flood = tweets['clean_text'].str.contains('flood', case=False)
print(" Contains flood: {0:f}".format(np.sum(flood)/tweets.shape[0]))
flooding = tweets['clean_text'].str.contains('flooding', case=False)
print(" Contains flooding: {0:f}".format(np.sum(flooding)/tweets.shape[0]))
hazard = tweets['clean_text'].str.contains('hazard', case=False)
print(" Contains hazard: {0:f}".format(np.sum(hazard)/tweets.shape[0]))


# ============================================ TOKENS ========================================================
nlp = spacy.load('en_core_web_sm')
tweets['tokens'] = np.nan
tweets['an_tokens'] = np.nan
print(tweets.info())
for i,t in enumerate(tweets['clean_text']):
    doc = nlp(t)
    #Create Tokens - remove non-alphabetic characters
    #tokens = [token.text for token in doc if token.text.isalpha()]
    #print(i, tokens)
    #tweets['tokens'].iloc[i] = np.array(tokens)
    an_tokens = [token.text for token in doc]
    print(i,an_tokens)
    tweets['an_tokens'].iloc[i] = np.array(an_tokens)
print(tweets.info())
print(tweets['tokens'].head())


# ============================================ POS tagging ============================================
nlp = spacy.load('en_core_web_sm')

# -------------------------------- PROPER NOUNS(IGNORE) ----------------------
#Returns number of proper nouns
def proper_nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]    
    # Return number of proper nouns
    return pos.count('PROPN')
    #return pos

tweets['num_propn'] = tweets['clean_text'].apply(proper_nouns)

print(tweets[tweets['num_propn'] == 0].count()) #TBD: Good stat

print((tweets['num_propn'] == 0).sum())#1332

# Compute mean of proper nouns
real_propn = tweets[tweets['target'] == 1]['num_propn'].mean() # REAL
fake_propn = tweets[tweets['target'] == 0]['num_propn'].mean() # FAKE
print("Mean no. of proper nouns in real and fake tweets are %.2f and %.2f respectively"%
      (real_propn, fake_propn))# 3.21--> real / 2.70 --> fake
real_propn/fake_propn

tweets['num_propn'].hist(bins=30)
plt.title("PROP NOUN Distribution")
plt.show()
# TBD: Figure out what kind of distribution is that

real_propn_med = tweets[tweets['target'] == 1]['num_propn'].median() # REAL
fake_propn_med = tweets[tweets['target'] == 0]['num_propn'].median() # FAKE
print("Mean no. of proper nouns in real and fake tweets are %.2f and %.2f respectively"%
      (real_propn_med, fake_propn_med))# 3--> real / 2 --> fake

'''
Proper nouns will have a strong correlation with mentions
and a lot of missing values
'''
# ------------------------------- NOUNS --------------------------------------
    
def nouns(text, model=nlp):
  	# Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]    
    # Return number of other nouns
    return pos.count('NOUN')

print("NOUNS: {0} TARGET: {1}".format(nouns(tweets['clean_text'].iloc[614].strip(), nlp), 
      tweets['target'].iloc[614]))

# Noun usage in fake tweets

tweets['num_noun'] = tweets['clean_text'].apply(nouns)
print((tweets['num_noun'] == 0).sum()) #589

# Compute mean of other nouns
real_noun = tweets[tweets['target'] == 1]['num_noun'].mean()#REAL
fake_noun = tweets[tweets['target'] == 0]['num_noun'].mean()#FAKE
real_noun/fake_noun#>23.4%
print("Mean no. of other nouns in real and fake tweets are %.2f and %.2f respectively"%
      (real_noun, fake_noun))# 3.73 real/3.02 fake

#MEDIAN
real_noun_med = tweets[tweets['target'] == 1]['num_noun'].median()#REAL
fake_noun_med = tweets[tweets['target'] == 0]['num_noun'].median()#FAKE

print("Median no. of other nouns in real and fake tweets are %.2f and %.2f respectively"%
      (real_noun_med, fake_noun_med))# 4 real/3 fake
real_noun_med/fake_noun_med #0.25
#TBD: Impuprint(bow_matrix.shape)tation strategy --> substitute missing noun values with median
'''
In general number of nouns and proper nouns are ~20% more in real tweets
TBD:
    1. Find All annotations for real and fake tweets;
    2. Build normalized distributions of all annotations where difference between real/Fake is
        substantial;
    3. Compare distributions , maybe consider Bayes trying to predict real and fake tweets
    4 Repeat everything for test data
'''

print(tweets.info())

# ==================================== READABILITY SCORES ===========================================
# TBD: Relaunch after learning to count sentenses
# THIS IS A PROBLEM
# def sent_count(text, abbr=Abbreviations(), prepped=False):
#    """Count number of sentences."""
#    if not prepped:
#        text = punct_clean(text, abbr)
#    return text.count('.') + text.count('!') + text.count('?')

'''
There are 4 most common Readability scores for english:
    -- Flesch reading ease
    -- Gunning fog index
    -- Simple Measure of Gobbledygook (SMOG)
    -- Dale-Chall score
'''
'''
Forest fire near La Ronge Sask. Canada {'flesch_score': 103.04428571428575, 
'fleschkincaid_score': 0.6257142857142863, 'gunningfog_score': 2.8000000000000003, 
'smog_score': 3.1291, 'dalechall_score': 13.006557142857142}
'''
from spacy_readability import Readability

nlp = spacy.load('en_core_web_sm')
read = Readability()
nlp.add_pipe(read, last=True)

doc = nlp("Forest fire near La Ronge Sask Canada")

print(doc._.flesch_kincaid_grade_level)
print(doc._.flesch_kincaid_reading_ease)
print(doc._.dale_chall)
print(doc._.smog)
print(doc._.coleman_liau_index)
print(doc._.automated_readability_index)
print(doc._.forcast)

# TBD: Create separate feature for each readability score, then build correlation matrix heat map and 
# remove scores that too much of correlation with each other 

# ============================================== LEMMAS ====================================================
#TBD Lemmatizing tokenized text before creating bag of words to reduce number of features
# number of string that had mentions in text
users=tweets['bald_text'][tweets['bald_text'].str.contains('Bratishka', case=False)]
print(" Contains &: {}".format(len(users)))# 2009
# --------------------- Lemmatizing ----------------------------------
nlp = spacy.load('en_core_web_sm')
tweets['lemmas'] = np.nan

def lemmatize(txt):
    doc = nlp(txt)
    lemmas = [token.lemma_ for token in doc]
    #print(txt,'\n',lemmas,'---------------------------')
    return ' '.join(lemmas)
    
tweets['bald_text'] = tweets['bald_text'].apply(lemmatize)
print(tweets.info())

# ============================================ BAG OF WORDS ==============================================
# ============================================= Count VECTORIZER =============================================
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix

# TBD: change logistic regression threshold to Real/Fake proportion

# limiting bag of words to only to 2-grams that occure now less then at 5 tweets
vectorizer = CountVectorizer(ngram_range=(2,2), min_df=5, stop_words=ENGLISH_STOP_WORDS).fit(tweets['bald_text'])
X_txt = vectorizer.transform(tweets['bald_text'])
bow_matrix_2 = pd.DataFrame(X_txt.toarray(), columns=vectorizer.get_feature_names())
print(bow_matrix_2.shape) 
'''
(7613, 14839) all tweets
(7613, 1302) for 10 tweets
(7613, 2447) for 5 tweets
(7613, 969) 2-grams for 5 tweets
------------ after lemmatizing -----------------

'''
X_train, X_test, y_train, y_test = train_test_split(bow_matrix_2, tweets['target'], test_size = 0.2, 
                                                    random_state=42, stratify=tweets['target'])

print(len(X_train), len(y_train))
log_reg_bow_matrix_2 = LogisticRegression().fit(X_train, y_train)
y_pred = log_reg_bow_matrix_2.predict(X_test)
print('Accuracy of logistic regression with 2-grams min 5 docs: ', accuracy_score(y_test, y_pred)) # 0.7209

# limiting bag of words to only to unigrams that occure now less then at 5 tweets
vectorizer = CountVectorizer(ngram_range=(1,1), min_df=5, stop_words=ENGLISH_STOP_WORDS).fit(tweets['bald_text'])
X_txt = vectorizer.transform(tweets['bald_text'])
bow_matrix_1 = pd.DataFrame(X_txt.toarray(), columns=vectorizer.get_feature_names())
print(bow_matrix_1.shape)# (7613, 2458)

X_train, X_test, y_train, y_test = train_test_split(bow_matrix_1, tweets['target'], test_size = 0.2, 
                                                    random_state=42, stratify=tweets['target'])

# HAND-MADE Hyperparamerers tuning
for C_value in np.arange(0.05, 1, 0.05):
    log_reg_bow_matrix_1 = LogisticRegression(C=C_value, solver='lbfgs').fit(X_train, y_train)
    y_pred = log_reg_bow_matrix_1.predict(X_test)
    print('Accuracy of LogReg with C:{0} TRAIN: {1}'.format(C_value, log_reg_bow_matrix_1.score(X_train, y_train))) # 0.8891
    print('Accuracy of LogReg with C:{0} TEST: {1}'.format(C_value, accuracy_score(y_test, y_pred)))
    print('----------------------------------------------------------------------')

'''
best regularization strength is between 0.4 and 0.6 --> Acc: 0.82 
'''

# CONFUSION MATRIX
print('Confusion matrix test set: \n', confusion_matrix(y_test, y_pred)/len(y_test))


# using unigrams and 3-grams that occure not less then at 5 tweets
vectorizer = CountVectorizer(ngram_range=(1,3), min_df=5, stop_words=ENGLISH_STOP_WORDS).fit(tweets['bald_text'])
X_txt = vectorizer.transform(tweets['bald_text'])
bow_matrix_3 = pd.DataFrame(X_txt.toarray(), columns=vectorizer.get_feature_names())
print(bow_matrix_3.shape) 

X_train, X_test, y_train, y_test = train_test_split(bow_matrix_3, tweets['target'], test_size = 0.2, 
                                                    random_state=42, stratify=tweets['target'])

log_reg_bow_matrix_3 = LogisticRegression().fit(X_train, y_train)
y_pred = log_reg_bow_matrix_3.predict(X_test)
print('Accuracy of logistic regression with unigrams and 3-grams min 5 docs: ', 
      accuracy_score(y_test, y_pred))# 0.8155
'''
Using multigrams doesn't add any value to the model predictions
'''

# ============================================= Tfidf VECTORIZER =============================================
'''
Similar to Bag Of Words models (Count Vectorizers) but contains Log-Ratios between each term frequency in text
 and in all document instead of just count of words, thus Tfidf is having lower score for most common words  
PARAMETERS:
    max_df -- is used for removing data values that appear too frequently, 
    also known as "corpus-specific stop words".
    For example:
        max_df = 0.50 means "It ignores terms that appear in more than 50% of the documents".
        max_df = 25 means "It ignores terms that appear in more than 25 documents".

    min_df --  is used for removing terms that appear too infrequently.
        For example:
            min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
            min_df = 5 means "ignore terms that appear in less than 5 documents".
'''

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

vect = TfidfVectorizer(ngram_range=(2,2), min_df=5, stop_words=ENGLISH_STOP_WORDS).fit(tweets['bald_text'])
X_txt = vect.transform(tweets['bald_text'])
X=pd.DataFrame(X_txt.toarray(), columns=vect.get_feature_names())
print(X.shape)
'''
(7613, 57497) with 1 and 2 grams 
(7613, 42612) with 2 grams only 
(7613, 41017) with 3 grams

(7613, 969) -- 2 grams ignoring less than 5 documents
'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------- Creating Tfidf vectorizer with 2 grams ignoring less than 5 documents
X_fit = TfidfVectorizer(ngram_range=(2,2), min_df=5, stop_words=ENGLISH_STOP_WORDS).fit(tweets['bald_text'])
X_vect = X_fit.transform(X)
X_tfidf_2grams_5docs = pd.DataFrame(X_vect.toarray(), columns=X_fit.get_feature_names())
print(X_tfidf_2grams_5docs.shape)
print(len(X_tfidf_2grams_5docs))

# Check target variable balance
print(tweets['target'].value_counts()/len(tweets['target']))# 57% agains 43% better to use stratified sampling

X_train, X_test, y_train, y_test = train_test_split(X_tfidf_2grams_5docs, tweets['target'], test_size = 0.2, 
                                                    random_state=42, stratify=tweets['target'])

print(len(X_train), len(y_train))

log_reg_2grams_5docs = LogisticRegression().fit(X_train, y_train)
y_pred = log_reg_2grams_5docs.predict(X_test)
print('Accuracy of logistic regression with 2-grams min 5 docs: ', accuracy_score(y_test, y_pred)) # 0.72619

# ---------------------- Creating Tfidf vectorizer with 2 and 3-grams ignoring less than 5 documents
X_fit = TfidfVectorizer(ngram_range=(2,3), min_df=5, stop_words=ENGLISH_STOP_WORDS).fit(tweets['bald_text'])
X_vect = X_fit.transform(tweets['bald_text'])
X_tfidf_2_3grams_5docs = pd.DataFrame(X_vect.toarray(), columns=X_fit.get_feature_names())
print(X_tfidf_2_3grams_5docs.shape)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf_2_3grams_5docs, tweets['target'], test_size = 0.2, 
                                                    random_state=42, stratify=tweets['target'])

log_reg_2_3grams_5docs = LogisticRegression().fit(X_train, y_train)
y_pred = log_reg_2_3grams_5docs.predict(X_test)
print('Accuracy of logistic regression with 2-grams min 5 docs: ', accuracy_score(y_test, y_pred)) # 0.7242
'''
Multigrams doesn't add any predictive power to the model
'''




# ================================================ Embeddings =====================================================
'''
On Kaggle they use embeddings with 'GloVe' and 'Fast Text' vocabularies no spell checking is performed
'''
DISASTER_TWEETS = tweets['target'] == 1
print(DISASTER_TWEETS)
print(~DISASTER_TWEETS)


