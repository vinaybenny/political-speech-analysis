import os
import json
import re, string, unicodedata
import nltk
import contractions
import inflect
import numpy as np
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from pandas import DataFrame, Series, read_json

DATA_DIRECTORY = "./data/data_scraped_translated"

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas

if __name__== "__main__":

    filedata = []
    directory = DATA_DIRECTORY

    # Load all files into memory as a list
    for filename in os.listdir(directory):
        with open(directory + '/' + filename, 'r',encoding='utf-8') as json_file:            
            data = json.load(json_file)
            filedata.append(data)
    
    # Convert the list into a pandas dataframe
    filedata = DataFrame(filedata)
    filedata = filedata.drop('content', axis = 1)

    # Vectorise required data pre-processing functions
    v_replace_contractions = np.vectorize(replace_contractions)
    
    # Apply data pre-processing & lemmatisation
    filedata['translated_data'] = Series(v_replace_contractions(filedata['translated_data']))
    filedata['cleaned_data'] = filedata['translated_data'].apply(word_tokenize)
    filedata['cleaned_data'] = filedata['cleaned_data'].apply(remove_non_ascii)    
    filedata['cleaned_data'] = filedata['cleaned_data'].apply(to_lowercase)
    filedata['cleaned_data'] = filedata['cleaned_data'].apply(remove_punctuation)
    filedata['cleaned_data'] = filedata['cleaned_data'].apply(replace_numbers)
    filedata['cleaned_data'] = Series(remove_stopwords(filedata['cleaned_data']))    
    filedata['cleaned_data'] = filedata['cleaned_data'].apply(lemmatize_verbs)    
