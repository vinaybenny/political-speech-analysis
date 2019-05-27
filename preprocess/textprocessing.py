# This code borrows helper functions from GitHub repository : llefebure/un-general-debates, by Luke Lefebure

import os
import json
import re, unicodedata
import nltk
import contractions
import inflect
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pandas import DataFrame, Series, read_json, to_datetime
import itertools
from collections import defaultdict

import gensim
from gensim.models import CoherenceModel, LdaModel, ldaseqmodel, LdaMulticore
from gensim.models.wrappers import LdaMallet, DtmModel
from gensim.corpora import Dictionary

import pyLDAvis.gensim
from pprint import pprint

import spacy, en_core_web_sm
import matplotlib.pyplot as plt


DTM_BINARY = "./lib/dtm/dtm-win64.exe"
DATA_DIRECTORY = "./data/data_scraped_translated"
#DATA_DIRECTORY = "./data/test"
#MALLET_BINARY = "./lib/mallet/bin/mallet.bat"

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


def lemmatize_remove_stopwords(docs, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    stop_words = stopwords.words('english')
    stop_words.extend(['ji', 'twitter'\
        ,'brother_sister', 'brother', 'sister', 'do_not', 'should_not', 'will_not', 'can_not', 'today', 'year', 'time'\
        ,'come'
#        , 'work', 'go', 'give', 'tell', 'say', 'want', 'get', 'day', 'thing'\
#        , 'way', 'see', 'crore', 'bring', 'take', 'man', 'lot', 'hand'\
#        , 'country', 'people', 'government', 'india'\
#        , 'do_not', 'will_not', 'not_only'
            ])
    nlp = en_core_web_sm.load()
    nlp.Defaults.stop_words |= {word for word in stop_words}   
    cleaned_docs =[]
    for doc in nlp.pipe([x.lower() for x in docs.values], batch_size=50):
        doc = [token.lemma_ for token in doc if ( not(token.is_stop) and not(token.is_punct) and token.pos_ in allowed_postags ) ]
        cleaned_docs.append(doc)
    return cleaned_docs


def make_ngrams(docs):
    bigram = gensim.models.Phrases(docs, min_count=3, threshold=5)
    trigram = gensim.models.Phrases(bigram[docs], min_count=3, threshold=5)

    bigram_phraser = gensim.models.phrases.Phraser(bigram)
    trigram_phraser = gensim.models.phrases.Phraser(trigram)

    #bigram_words = [bigram_phraser[doc] for doc in docs]
    trigram_words = [trigram_phraser[bigram_phraser[doc]] for doc in docs]   
    return DataFrame({'test_set': trigram_words})   

def paragraph_tokenize(text):
    '''
    Returns a list of paragraphs for each text
    Borrowed from Github repository : llefebure/un-general-debates
    '''
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
#    words = remove_non_ascii(words)
#    words = remove_punctuation(words)
#    words = replace_numbers(words)
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
        model = LdaModel(corpus=corpus, num_topics=num_topics, alpha='symmetric', eta='auto', id2word=dictionary, chunksize=500, passes=50 \
                         ,iterations = 150, random_state = 12345, minimum_probability = 0.05, decay=0.5)
        
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        print("Coherence score is {}".format(coherencemodel.get_coherence()))
        print('Perplexity: ', model.log_perplexity(corpus))
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def tune_dtm_model(dictionary, corpus, texts, times, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print("Number of Topics is {}".format(num_topics))
        model = DtmModel(DTM_BINARY, corpus=corpus, id2word=dictionary, time_slices=times, num_topics=num_topics\
                          ,lda_sequence_min_iter=50, lda_sequence_max_iter=150, lda_max_em_iter=150, rng_seed = 12345\
                          ,alpha=0.05, top_chain_var=0.05)
        model_list.append(model)        
        # FInd Coherence for each time slice
        coherence_n = []
        for time_slice in range(len(time_slices)):
            topics_n_t = model.dtm_coherence(time=time_slice)
            coherence_model_t = CoherenceModel(topics=topics_n_t, corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v')   
            print ("Coherence for time slice %d is " % time_slice, coherence_model_t.get_coherence())
            coherence_n.append(coherence_model_t.get_coherence())        
        
        print("Average Coherence score is {}".format(np.mean(coherence_n)))
        coherence_values.append(np.mean(coherence_n))
    return model_list, coherence_values

def term_distribution(model, term, topic):
    """Extracts the probability over each time slice of a term/topic pair."""
    word_index = model.id2word.token2id[term]
    topic_slice = np.exp(model.lambda_[topic])
    topic_slice = topic_slice / topic_slice.sum(axis=0)
    return topic_slice[word_index]

    
def plot_terms(model, topic, terms, title=None, name=None, hide_y=True):
    """Creates a plot of term probabilities over time in a given topic."""
    fig, ax = plt.subplots()
    plt.style.use('fivethirtyeight')
    for term in terms:
        ax.plot(
            range(6), term_distribution(model, term, topic),
            label=term)
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if hide_y:
        ax.set_yticklabels([])
    ax.set_ylabel('Probability')
    if title:
        ax.set_title(title)
    if name:
        fig.savefig(
            name, dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
    return fig, ax

def top_term_table(model, topic, slices, topn=10):
    """Returns a dataframe with the top n terms in the topic for each of
    the given time slices."""
    data = {}
    for time_slice in slices:
        time = np.where(model.time_slice_labels == time_slice)[0][0]
        data[time_slice] = [
            term for p, term
            in model.show_topic(topic, time=time, topn=topn)
        ]
    return DataFrame(data)

def term_distribution_ldaseq(model, term, topic):
    """Extracts the probability over each time slice of a term/topic pair."""
    word_index = model.id2word.token2id[term]
    topic_slice = np.exp(model.topic_chains[topic].e_log_prob)
    topic_slice = topic_slice / topic_slice.sum(axis=0)
    return topic_slice[word_index]

def plot_terms_lda(model, topic, terms, title=None, name=None, hide_y=True):
    """Creates a plot of term probabilities over time in a given topic."""
    fig, ax = plt.subplots()
    plt.style.use('fivethirtyeight')
    for term in terms:
        ax.plot(
            range(6), term_distribution_ldaseq(model, term, topic),
            label=term)
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if hide_y:
        ax.set_yticklabels([])
    ax.set_ylabel('Probability')
    if title:
        ax.set_title(title)
    if name:
        fig.savefig(
            name, dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
    return fig, ax


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
    filedata['cleaned_data'] = lemmatize_remove_stopwords(filedata['cleaned_data'], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    filedata['cleaned_data'] = make_ngrams(filedata['cleaned_data'])
    paragraph_data['paragraphs'] = paragraph_data['paragraphs'].apply(normalize)
    paragraph_data['paragraphs'] = lemmatize_remove_stopwords(paragraph_data['paragraphs'], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    paragraph_data = paragraph_data[paragraph_data['paragraphs'].map(lambda d: len(d)) > 10]
    paragraph_data = paragraph_data.reset_index(drop=True)
    paragraph_data['paragraphs'] = make_ngrams(paragraph_data['paragraphs'])
    

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
    time_slices = full_data.groupby('month_year')['month_year'].count().tolist()
    dm_list_para, dm_coherence_para = tune_dtm_model(dictionary_para, corpus_para, full_data['paragraphs'], time_slices, start = 14, limit = 20, step = 2)
    best_dtm_para = dm_list_para[np.argmax(dm_coherence_para)]
    # Save the dynamic model
    best_dtm_para.save(os.path.join('.\\data\\', 'dtm.gensim'))
    
    top_term_table(best_dtm_para, topic = 0, slices = range(1,7), topn = 10)
    
    
    
#    ldaseq_para = ldaseqmodel.LdaSeqModel(corpus=corpus_para, id2word=dictionary_para, time_slice=time_slices, num_topics=14, random_state=12345\
#                                          ,initialize='ldamodel',lda_model = best_model_para )    
#    ldaseq_para.save(os.path.join('.\\models\\', 'ldaseq.gensim'))

    
    



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
best_model_para.show_topics(formatted=False,num_words=10)
best_model_para.top_topics(corpus = corpus_para, texts = full_data['paragraphs'], dictionary=dictionary_para, coherence='c_v')


best_dtm_para.show_topics(num_topics=-1, times=1, num_words=100, formatted=True)
plot_terms(best_dtm_para, topic = 1, terms = ['congress', 'terrorism'], title=None, name=None, hide_y=True)
coherence_model_t = CoherenceModel(topics=best_dtm_para.dtm_coherence(time = 1), corpus=corpus_para, texts=full_data['paragraphs'], dictionary=dictionary_para, coherence='c_v')
coherence_model_t.get_coherence_per_topic()



frequency = defaultdict(int)
for text in paragraph_data['paragraphs']:
     for token in text:
         frequency[token] += 1

texts = [
     [token for token in text if frequency[token] == 1]
     for text in paragraph_data['paragraphs']
 ]

temp = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus_para):
    temp[dictionary_para[word_id]] += word_count
sorted(temp.items(), key=lambda x: x[1], reverse=True)[:100]