import json
import os
import time
from googletrans import Translator

directory = "./extract_speech/data"

for filename in os.listdir(directory):
    with open(directory + '/' + filename, encoding='utf-8') as json_file:  
        data = json.load(json_file)        
        sentences = data['content'].split('।')
        translated_list=[]
        i=1

        for sentence in sentences:
            translator = Translator()
            translated_sentence = translator.translate(sentence).text
            #print(sentence)
            #print(translated_sentence)
            translated_list.append(translated_sentence)
            time.sleep(2)
            i += 1
            print(i)
        
    translated_data = ". ".join(translated_list)
    data['translated_data'] = translated_data

    with open(directory + '_translated/' + filename, 'w', encoding='utf-8') as datafile:
        json.dump(data, datafile, ensure_ascii=False)

    #os.remove(directory + '/' + filename)