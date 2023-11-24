import numpy as np
import n2w

import string
import contractions #library pertaining to contractions (things like "don't" and "you're")
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words') #nltk's collection of words

from nltk.corpus import stopwords,  words 
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.metrics.distance import jaccard_distance #distance we'll use to find the nearest correct word
from nltk.util import ngrams

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

import gensim.downloader as api
w2v = api.load('word2vec-google-news-300')



words_into_list = words.words() #this is a text file with 1 word per line
#words_into_list = words_into_list+[str(i) for i in range(0,1000000)] #add numbers to the list
words_lower = [word.lower() for word in words_into_list] #we will make all words lowercase
word_set = set(words_lower)
word_arr = np.array(list(word_set))

def token_word(essay):
    tokens = nltk.word_tokenize(essay.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens
def token_sent(essay):
    tokens = nltk.sent_tokenize(essay.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens

# The following function calculates burstiness of an essay
def burstiness(essay):
    sentences = token_sent(essay)
    num_words   = len(token_word(essay))  #Total number of words in text
    num_sents   = len(sentences)  #Total number of sentences in text
    avg_freq = num_words/num_sents #Average number of words per sentence 
    variance = sum((len(sentence.split()) - avg_freq) ** 2 for sentence in sentences) / len(sentences)
    return variance

def metrics(essay, dict_out=False):
    """
    input:
        text essay as string
    output:
        a bunch of metrics for that essay
    """
    sia_scores = sia.polarity_scores(essay)
    
    if dict_out:
        metrics = sia_scores
        metrics['burst'] = burstiness(essay)
    else:
        metrics = np.array([burstiness(essay), sia_scores['pos'] , sia_scores['neg'], sia_scores['neu'], sia_scores['compound']])
    
    return metrics


def preprocess_spelling(essay):
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+') #This tokenises strings that consist of characters and number, i.e. it removes other symbols
    text_no_contr = contractions.fix(essay)  #Expands all contractions in essays. For example, converts "you're" to "you are".
    words_no_punct= tokenizer.tokenize(text_no_contr.lower())  #Removes all non-letter and non-number symbols. Also makes everything lowercase.
    return words_no_punct


def corrected(essay):
    errors=[]
    clean_essay = preprocess_spelling(essay)
    correct_essay_words = clean_essay
    for word_index in range(0,len(clean_essay)): #loop through words in each essay
        word = clean_essay[word_index]
        if word not in word_set: #if a word is not contained in our word_set, correct it using jaccard
            temp = [(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))),w) 
            for w in word_set if w[0]==word[0]] 
            correct_word = sorted(temp, key = lambda val:val[0])[0][1] #corrected word
            correct_essay_words[word_index] = correct_word
    correct_essay = " ".join(correct_essay_words)
    return correct_essay


def preprocess(essay, word_tokenize=False):
    """
    Do a lot of the pre-processing of the text
    input:
        essay (string)
    output:
        correct_essay_words (list of strings) cleaned/corrected essay words in order
    """
    
    clean_essay = preprocess_spelling(essay)
    correct_essay_words = clean_essay
    
    # use numpy to avoid the nested loops
    
    # find the indices where the words arent in the word set
    missing_word_inds = np.where(~np.isin(np.array(clean_essay),word_arr))[0]
    
    for word_index in missing_word_inds:
        word = clean_essay[word_index]

        
        try: 
            # first check if the word is in word2vec
            w2v.get_vector(word)
        except KeyError:# it's not in w2v
            # check if it's a number first
            num = n2w.convert(word)
            if num=='Input not a valid number':# it's not a number
                
                # find the closest word in the list of words
                temp = [(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))),w) for w in word_set if w[0]==word[0]]
                try:
                    correct_word = sorted(temp, key = lambda val:val[0])[0][1] #corrected word
                    correct_essay_words[word_index] = correct_word
                except IndexError: # couldn't find a "closest word"
                    pass# so we just leave the word back as is
                
            else:
                correct_essay_words[word_index] = num
            
    if word_tokenize:
        return correct_essay_words
    else:
        correct_essay = " ".join(correct_essay_words)
        return correct_essay