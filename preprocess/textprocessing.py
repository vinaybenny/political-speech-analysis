import os
import json
import re, string, unicodedata
import nltk
import contractions
import inflect
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pandas import DataFrame, Series, read_json, to_datetime

import gensim
from gensim.models import CoherenceModel, LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary

import pyLDAvis.gensim
from pprint import pprint
import operator

import spacy, en_core_web_sm
import matplotlib.pyplot as plt

DATA_DIRECTORY = "./data/data_scraped_translated"
#DATA_DIRECTORY = "./data/test"

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_special_occurences(text):
    """Replace special groups of characters in string of text using regular expressions"""
    # Remove new lines or tabs
    text = re.sub(r'\s+', ' ', text)    
    # Remove URLs
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    # Remove twitter pic links
    text = re.sub(r'pic. twitter. co(m|\.[a-z]*)\/(\/|\S+)', '', text)
    return text

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
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

    # Set up NLTK's stop word corpus.
    stop_words = stopwords.words('english')

    # List extra words to be discarded from tokens
    stop_words.extend(['ji', 'twitter'\
        , 'come', 'today', 'work', 'year', 'brother_sister', 'time', 'go', 'give', 'tell', 'say', 'want', 'get', 'day', 'thing', 'brother'\
        , 'way', 'see', 'crore', 'bring', 'take', 'man', 'lot', 'hand'\
        , 'country', 'people', 'government', 'india'\
            ])

    # Set up spacy's stop word list
    nlp = en_core_web_sm.load()

    # Create a combination of spacy and nltk stop words
    nlp.Defaults.stop_words |= {word for word in stop_words}

    for word in words:
        if not(nlp.vocab[word].is_stop):
            new_words.append(word)
    return new_words

def make_ngrams(docs):
    bigram = gensim.models.Phrases(docs, min_count=3, threshold=5)
    trigram = gensim.models.Phrases(bigram[docs], min_count=3, threshold=5)

    bigram_phraser = gensim.models.phrases.Phraser(bigram)
    trigram_phraser = gensim.models.phrases.Phraser(trigram)

    #bigram_words = [bigram_phraser[doc] for doc in docs]
    trigram_words = [trigram_phraser[bigram_phraser[doc]] for doc in docs]   
    return DataFrame({'test_set': trigram_words})   

def lemmatize(words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = en_core_web_sm.load()
    doc = nlp(" ".join(words)) 
    words_out = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return words_out

def word_tokenize(text):
    return gensim.utils.simple_preprocess(str(text), deacc=True, min_len=2, max_len=20)

def normalize(words):
    words = replace_contractions(words)
    words = remove_special_occurences(words)
    words = word_tokenize(words)
    
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)    
    words = lemmatize(words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])  
    return words


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("Number of Topics is {}".format(num_topics))
        model = LdaModel(corpus=corpus, num_topics=num_topics, alpha='auto', eta='auto', id2word=dictionary, chunksize=2000, passes=50)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        print("Coherence score is {}".format(coherencemodel.get_coherence()))
        print('Perplexity: ', model.log_perplexity(corpus))
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

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
    filedata['speechdate'] = to_datetime(filedata['speechdate'], infer_datetime_format=True)
    filedata = filedata.sort_values(by = ['speechdate'])
    filedata = filedata.reset_index(drop=True)
    
    # Apply data pre-processing & lemmatisation    
    filedata['cleaned_data'] = filedata['translated_data'].apply(normalize)
    filedata['cleaned_data'] = make_ngrams(filedata['cleaned_data'])
    filedata['lemmatised_data'] = filedata['cleaned_data'].apply(remove_stopwords)

    #TODO: Perform a spell-correction on the words
    #TODO: Remove words that occur only once (after spell correction)

    # Try overall topic detection first
    dictionary = Dictionary(filedata['lemmatised_data'])
    corpus = [dictionary.doc2bow(text) for text in filedata['lemmatised_data']]
    lm_list, coherence_values = compute_coherence_values(dictionary, corpus, filedata['lemmatised_data'], limit = 30, step = 2)
    # lm_list[np.argmax(coherence_values)]


    # Try batchwise topic detection
    batchsize = 10
    best_models=[]
    for batchnum in range(0, len(filedata['lemmatised_data']) - batchsize):
        dictionary = Dictionary(filedata[batchnum:(batchnum + batchsize)]['lemmatised_data'])
        corpus = [dictionary.doc2bow(text) for text in filedata[batchnum:(batchnum + batchsize)]['lemmatised_data']]
        model_list, coherence_measures = compute_coherence_values(dictionary, corpus, filedata[batchnum:(batchnum + batchsize)]['lemmatised_data'], limit = 10, step = 2)
        best_models.append(model_list[np.argmax(coherence_measures)])

    
    


# import pickle
# with open('.\\data\\best_models.pkl', 'wb') as output:
#     pickle.dump(best_models, output, pickle.HIGHEST_PROTOCOL)

# with open('.\\data\\best_models.pkl', 'rb') as input:
#     best_models = pickle.load(input)

# limit=30
# start=2
# step=2
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()

# pprint(lm_list[8].print_topics(num_words=20))
# # doc_lda = lm_list[11][corpus]
# # model = LdaModel(corpus=corpus, num_topics=12, id2word=dictionary, chunksize=2000, passes=1)

# # Check most frequent words
# import itertools
# from collections import defaultdict

# total_count = defaultdict(int)
# for word_id, word_count in itertools.chain.from_iterable(corpus):
#     total_count[dictionary[word_id]] += word_count
# # Top ten words
# sorted(total_count.items(), key=lambda x: x[1], reverse=False)[:100]

# once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]

# frequency = defaultdict(int)
# for text in filedata['cleaned_data']:
#      for token in text:
#          frequency[token] += 1

# texts = [
#      [token for token in text if frequency[token] == 1]
#      for text in filedata['cleaned_data']
#  ]

# print(texts)