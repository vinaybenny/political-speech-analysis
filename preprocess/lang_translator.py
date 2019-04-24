import json
import os
import time
from googletrans import Translator

GTRANS_QUERY_DELAY = 2
DATA_DIRECTORY = "./extract_speech/data"

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

if __name__== "__main__":

    rejected_files = []
    directory = DATA_DIRECTORY

    for filename in os.listdir(directory):        
        with open(directory + '/' + filename, encoding='utf-8') as json_file:  
            data = json.load(json_file)        
            sentences = data['content'].split('।')
            translated_list=[]
            sentence_count = len(sentences)
            iter_counter=1

            for sentence in sentences:
                # Check is the full sentence is in English. If so, no translation is required.
                if isEnglish(sentence):
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
                    time.sleep(GTRANS_QUERY_DELAY)
                
                #print(sentence)
                #print(translated_sentence)
                translated_list.append(translated_sentence)
                print("%s / %s" % (iter_counter, sentence_count))                            
                iter_counter += 1

        translated_data = ". ".join(translated_list)
        data['translated_data'] = translated_data
        print(translated_data)

        with open(directory + '_translated/' + filename, 'w', encoding='utf-8') as datafile:
            json.dump(data, datafile, ensure_ascii=False)

        os.remove(directory + '/' + filename)