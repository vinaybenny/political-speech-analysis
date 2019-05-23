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
import itertools
from collections import defaultdict
import pickle

import gensim
from gensim.models import CoherenceModel, LdaModel, ldaseqmodel
from gensim.models.wrappers import LdaMallet, DtmModel
from gensim.corpora import Dictionary

import pyLDAvis.gensim
from pprint import pprint
import operator

import spacy, en_core_web_sm
import matplotlib.pyplot as plt

import logging


DTM_BINARY = "./lib/dtm-win64.exe"
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
#        , 'come', 'today', 'work', 'year', 'brother_sister', 'time', 'go', 'give', 'tell', 'say', 'want', 'get', 'day', 'thing', 'brother'\
#        , 'way', 'see', 'crore', 'bring', 'take', 'man', 'lot', 'hand'\
#        , 'country', 'people', 'government', 'india'\
#        , 'do_not', 'will_not', 'not_only'
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

def paragraph_tokenize(text):
    '''Returns a list of paragraphs for each text'''
    sentence_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    sentence_spans = list(sentence_tokenizer.span_tokenize(text))
    breaks = []
    for i in range(len(sentence_spans) - 1):
        sentence_divider = text[sentence_spans[i][1]: \
            sentence_spans[i+1][0]]
        if '\n' in sentence_divider:
            breaks.append(i)
    paragraph_spans = []
    paragraphs = []
    start = 0
    for break_idx in breaks:
        paragraph_spans.append((start, sentence_spans[break_idx][1]))
        start = sentence_spans[break_idx+1][0]
    paragraph_spans.append((start, sentence_spans[-1][1]))
    
    # Get paragraphs from spans
    for idx in paragraph_spans:
        paragraphs.append(text[idx[0]:idx[1]])

    return paragraphs

def normalize(words):
    words = replace_contractions(words)
    words = remove_special_occurences(words)
    words = word_tokenize(words)    
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)    
    words = lemmatize(words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])  
    return words


def tune_lda_model(dictionary, corpus, texts, limit, start=2, step=3):
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
        model = LdaModel(corpus=corpus, num_topics=num_topics, alpha='auto', eta='auto', id2word=dictionary, chunksize=500, passes=50 \
                         , iterations = 150, random_state = 12345, minimum_probability = 0.05, decay=0.5)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        print("Coherence score is {}".format(coherencemodel.get_coherence()))
        print('Perplexity: ', model.log_perplexity(corpus))
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

if __name__== "__main__":

    filedata = []
    directory = DATA_DIRECTORY
#    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
    filedata['month_year'] = to_datetime(filedata['speechdate']).dt.to_period('M')
    
    paragraph_data = DataFrame(data = None, columns = ['fileindex', 'paragraphs'])
    for idx, row in filedata.iterrows():
        paragraphs = paragraph_tokenize(row['translated_data'])
        paragraph_data = paragraph_data.append(DataFrame(data = {'fileindex': [idx] * len(paragraphs), 'paragraphs':paragraphs}), ignore_index=True)
    
    ############### Apply data pre-processing & lemmatisation ##############       
    
    filedata['cleaned_data'] = filedata['translated_data'].apply(normalize)
    filedata['cleaned_data'] = make_ngrams(filedata['cleaned_data'])
    filedata['cleaned_data'] = filedata['cleaned_data'].apply(remove_stopwords)    
    paragraph_data['paragraphs'] = paragraph_data['paragraphs'].apply(normalize)
    paragraph_data = paragraph_data[paragraph_data['paragraphs'].map(lambda d: len(d)) > 15]
    paragraph_data = paragraph_data.reset_index(drop=True)
    paragraph_data['paragraphs'] = make_ngrams(paragraph_data['paragraphs'])
    paragraph_data['paragraphs'] = paragraph_data['paragraphs'].apply(remove_stopwords)
    

    # Compactify the paragraph dataframe to remove rows with blank cleaned data.
    paragraph_data = paragraph_data[paragraph_data['paragraphs'].astype(bool)]    
    full_data = paragraph_data.merge(filedata, how='inner', left_on='fileindex', right_index = True)
    full_data = full_data.reset_index(drop=True)

    #TODO: Perform a spell-correction on the words
    #TODO: Remove words that occur only once (after spell correction)

    ############## 1: Try overall topic detection for the files ##############
    
    dictionary_files = Dictionary(filedata['cleaned_data'])
    corpus_files = [dictionary_files.doc2bow(text) for text in filedata['cleaned_data']]
    lm_list_files, coherence_files = tune_lda_model(dictionary_files, corpus_files, filedata['cleaned_data'], limit = 30, step = 2)
    best_model_files = lm_list_files[np.argmax(coherence_files)]
    
    ############## 2: Try overall topic detection for the paragraphs ##############
    
    dictionary_para = Dictionary(full_data['paragraphs'])
    dictionary_para.filter_extremes(no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None)    
    corpus_para = [dictionary_para.doc2bow(text) for text in full_data['paragraphs']]    
    # Remove all words that appear only once in the whole document set.
    total_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus_para):
        total_count[word_id] += word_count
#    sorted(total_count.items(), key=lambda x: x[1], reverse=True)[:100]
    dictionary_para.filter_tokens([k for k,v in total_count.items() if float(v) <= 1])
    dictionary_para.compactify()
    corpus_para = [dictionary_para.doc2bow(text) for text in full_data['paragraphs']]    
    
    lm_list_para, coherence_para = tune_lda_model(dictionary_para, corpus_para, full_data['paragraphs'], limit = 30, step = 2)
    best_model_para = lm_list_para[np.argmax(coherence_para)]


    ############## 3:  Try a dynamic topic model for files ##############
    dm_files = DtmModel(DTM_BINARY, corpus=corpus_files, id2word=dictionary_files, time_slices=filedata.groupby('month_year')['month_year'].count().tolist(), num_topics=16)
    
    ############## 4:  Try a dynamic topic model for paragraphs ##############
    dm_para = DtmModel(DTM_BINARY, corpus=corpus_para, id2word=dictionary_para, time_slices=full_data.groupby('month_year')['month_year'].count().tolist(), num_topics=16)    
    ldaseq_para = ldaseqmodel.LdaSeqModel(corpus=corpus_para, id2word=dictionary_para, time_slice=full_data.groupby('month_year')['month_year'].count().tolist(), num_topics=16)
    # Save the dynamic model
    dm_para.save(os.path.join('.\\data\\', 'dtm.gensim'))    
    topics = dm_para.show_topic(topicid=7, time=5, num_words=15)
      
    # View Coherence for specific time slices
    dm_para_time = dm_para.dtm_coherence(time=3)
    coherence_dm_para_time = CoherenceModel(topics=dm_para_time, corpus=corpus_para, dictionary=dictionary_para, coherence='u_mass')    
    print ("U_mass topic coherence")
    print ("Wrapper coherence is ", coherence_dm_para_time.get_coherence())
    
    


limit=30
start=2
step=2
x = range(start, limit, step)
plt.plot(x, coherence_para)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

pprint(best_model_para.print_topics(num_words=20))
# doc_lda = lm_list[11][corpus]





frequency = defaultdict(int)
for text in paragraph_data['paragraphs']:
     for token in text:
         frequency[token] += 1

texts = [
     [token for token in text if frequency[token] == 1]
     for text in paragraph_data['paragraphs']
 ]

print(texts)

nlp = en_core_web_sm.load()
doc = nlp(filedata['translated_data'][0])
d = []
for idno, sentence in enumerate(doc.sents):
    d.append({"id": idno, "sentence":str(sentence)})
    print('Sentence {}:'.format(idno + 1), sentence) 
df = pd.DataFrame(d)
df.set_index('id', inplace=True)
print(df) 