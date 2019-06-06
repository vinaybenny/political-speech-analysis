# -*- coding: utf-8 -*-
"""
@created: 23 April 2019
@author: vinay.benny
@description: For all scraped files present in .data/data_scraped folder, this program extracts those files that
    are in non-latin character languages (or has some content that is in non-latin), and translates into English. Uses 
    the python package "googletrans" (totally unethical). This script is intended to be a short-term stopgap until
    this homemade proof-of-concept is completed. The script deliberately uses a sentence-by-sentence translation to
    prevent causing unecessary load on google translate, and each successive translate query is delayed randomly with 
    an average duration specified by constant "GTRANS_QUERY_DELAY". Still doesn't excuse the use of this backdoor, but
    oh well. The code is to be run from command line, and needs specification of the source folder for the input files. 
    The script has only been tested on content that is in Hindi language (or mix of Hindi & English) so far. If there
    are any files that are completely in English, no google translate queries are made for those files, and are simply
    skipped over. Invoke the script as follows:
        python ./preprocess/lang_translator.py -f <foldername>
    where foldername is the folder in "./data_scraped/" with all the files to be translated. Use this script after 
    running the crawler "nm_speech_spider" in the "./extract_speech" folder.
    
    This code uses progress-bar function sourced from user "Greenstick" on StackOverflow - full credit.
"""
import json
import os
import time
import re
from googletrans import Translator
import numpy.random
import unicodedata as ud
from argparse import ArgumentParser

latin_letters= {}
GTRANS_QUERY_DELAY = 3
DATA_DIRECTORY = "./data/data_scraped"

def is_latin(uchr):
    try: return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

def only_roman_chars(unistr):
    return all(is_latin(uchr) for uchr in unistr if uchr.isalpha()) # isalpha suggested by John Machin

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

if __name__== "__main__":   

    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="src_directory", help="Source directory for input files", metavar="FILE")    
    args = parser.parse_args()   
    rejected_files = []
    directory = DATA_DIRECTORY + '/' + args.src_directory

    for filename in os.listdir(directory):
        print("Starting translation for file %s" % filename)
        with open(directory + '/' + filename, encoding='utf-8') as json_file:  
            data = json.load(json_file)
            delimiters = ".", "।"
            regexPattern = '|'.join(map(re.escape, delimiters))
            sentences = re.split(regexPattern, data['content'])
            translated_list=[]
            sentence_count = len(sentences)
            iter_counter=1

            # Check if the file content is completely in English. If so, we can skip any translations.
            if only_roman_chars(data['content']):
                translated_list.append(data['content'])

            else:
                for sentence in sentences:
                    #print(sentence)
                    # Check is the full sentence is in English. If so, no translation is required.
                    if only_roman_chars(sentence):
                        translated_sentence = sentence
                    else:
                        # Check whether the number of characters in the sentence is longer than 12k. If so, abort translation of current file.
                        if len(sentence) > 12000:
                            rejected_files.append(filename)
                            print("File %s has been rejected from translation queue owing to sentences with more than 12k characters." % filename)
                            continue
                        # Apply Google translate to sentence
                        translator = Translator()
                        translated_sentence = translator.translate(sentence).text

                        # Add a bit of (capped) randomness to the time delay between successive google translate queries.
                        wait_delay= round(numpy.random.normal(GTRANS_QUERY_DELAY, 2),1)
                        if wait_delay < 1.9:
                            wait_delay = 1.9
                        elif wait_delay > 7.3:
                            wait_delay = 7.3
                        else:
                            pass
                        time.sleep(wait_delay)
                    
                    #print(translated_sentence)
                    translated_list.append(translated_sentence)
                    printProgressBar(iter_counter, sentence_count, prefix = 'Progress:', suffix = 'Complete', length = 50)
                    #print("%s / %s" % (iter_counter, sentence_count))                            
                    iter_counter += 1

        translated_data = ". ".join(translated_list)
        data['translated_data'] = translated_data
        #print(translated_data)
        
        tgtpath = DATA_DIRECTORY + '_translated/' + args.src_directory + '/' 
        if not os.path.exists(tgtpath):
            os.makedirs(tgtpath)
        
        with open(tgtpath + filename, 'w', encoding='utf-8') as datafile:
            json.dump(data, datafile, ensure_ascii=False)

        #os.remove(directory + '/' + filename)
        
        # Print rejected files into a log.
        with open(DATA_DIRECTORY + '_translated/rejectedfiles.txt', 'w', encoding='utf-8') as f:
            for item in rejected_files:
                f.write("%s\n" % item)