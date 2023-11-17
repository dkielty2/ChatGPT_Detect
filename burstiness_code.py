# Burstiness
# This is a script file containing a function that calculates burstiness of an essay.


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.tokenize import sent_tokenize


# We preprocess and tokenise by words with all words in lower case, and we remove punctuation and stop words.
def token_word(essay):
    tokens = nltk.word_tokenize(essay.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens


# We preprocess and tokenise by sentences with all the words in lowercase and with punctuation removed.
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






