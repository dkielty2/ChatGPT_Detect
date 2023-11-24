import os
import time
import datetime

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from collections import Counter

import openai 

# API_keys.py is a file that holds an api key
# we can simply import the variable that holds the api key and use it
from API_keys import open_ai_key
openai.api_key = open_ai_key

from retry.api import retry_call

#choose where to start and end making GPT essays in file_list: 
#start = 4000
start = 4173
end = 8000

#initialize global variable i at the start
i = start


class chatAI:
    """
    Object class to use ChatGPT API
    based off of: https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
    """
    def __init__(self,model="gpt-3.5-turbo"):
        self.model = model
        self.log = [ {"role": "system", 
                      "content": "You are a intelligent assistant."} ]
    def chat(self,prompt):
        self.log.append( {"role": "user",
                          "content": prompt}) 
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.log)
        reply = chat.choices[0].message.content # get the reply
        #log it
        self.log.append({"role": "system",
                         "content": reply}) 
        return reply

def gen_essays():
    
    global i
    
    for essay_prompt in prompts[i:end]:
        gpt = chatAI() # create an object that uses the API
        essay = gpt.chat(prompt = essay_prompt)

        file_name = 'GPT_' + file_list[i]
        file_path = os.path.join(GPT_dir, file_name)

        with open(file_path, "w") as text_file:
            text_file.write(essay)

        i += 1
        if i%5 == 0:
            print(f"Generated the {i}th ChatGPT essay.")
            
if __name__=='__main__':
    #make list of human essays file names

    data_path = '../data/'
    folder_path = data_path + 'train'

    file_list = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    file_list.sort() # os.listdir() returns the file list in random(ish) order. Sort to standardize.
    
    
    #make new folder for ChatGPT essays
    folder_name = 'GPT_essays' 
    GPT_dir = data_path + folder_name

    if os.path.exists(GPT_dir):
        print('GPT_essay directory exists.')
    else:
        os.mkdir(GPT_dir)
        print('GPT_essay directory has been created.')
    
    # load prompts
    file_name = 'ChatGPT_prompts.txt'

    with open(file_name, 'r', encoding='utf-8') as file:
        prompts_str = file.read()

    prompts = prompts_str.split('\n')
    
    # loop to auto reconnect after timeout
    #global i
    while i<=end:
        try:
            retry_call(gen_essays())
        except APIError:# we got a timeout error
            print('Timed out at ', datetime.datetime.now())
            time.sleep(10)# pause for 10 sec
            continue # start back to the top of loop
    
    print('Done.')