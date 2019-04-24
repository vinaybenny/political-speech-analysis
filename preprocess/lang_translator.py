import json
import os
import time
from googletrans import Translator
import numpy.random
import unicodedata as ud

latin_letters= {}
GTRANS_QUERY_DELAY = 3
DATA_DIRECTORY = "../extract_speech/data"

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

    rejected_files = []
    directory = DATA_DIRECTORY

    for filename in os.listdir(directory):
        print("Starting translation for file %s" % filename)
        with open(directory + '/' + filename, encoding='utf-8') as json_file:  
            data = json.load(json_file)        
            sentences = data['content'].split('।')
            translated_list=[]
            sentence_count = len(sentences)
            iter_counter=1

            for sentence in sentences:
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
                
                #print(sentence)
                #print(translated_sentence)
                translated_list.append(translated_sentence)
                printProgressBar(iter_counter, sentence_count, prefix = 'Progress:', suffix = 'Complete', length = 50)
                #print("%s / %s" % (iter_counter, sentence_count))                            
                iter_counter += 1

        translated_data = ". ".join(translated_list)
        data['translated_data'] = translated_data
        #print(translated_data)

        with open(directory + '_translated/' + filename, 'w', encoding='utf-8') as datafile:
            json.dump(data, datafile, ensure_ascii=False)

        os.remove(directory + '/' + filename)