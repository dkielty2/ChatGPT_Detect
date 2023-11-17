# Spell check
# This file contains a function that takes an essay and returns the number of spelling mistakes.
#

import contractions #library pertaining to contractions (things like "don't" and "you're")
import nltk
from nltk.tokenize import RegexpTokenizer #we'll use this to remove non-number and non-letter symbols


# The following code preprocesses an essay. Corresponding to each essay we now have a list of 
# words that are lowercase, all contractions are expanded, and they do not contain no non-number and 
# non-letter symbols.

def preprocess_spelling(essay):
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+') #This tokenises strings that consist of characters and number, i.e. it removes other symbols
    text_no_contr = contractions.fix(essay)  #Expands all contractions in essays. For example, converts "you're" to "you are".
    words_no_punct= tokenizer.tokenize(text_no_contr.lower())  #Removes all non-letter and non-number symbols. Also makes everything lowercase.
    return words_no_punct



# The following code loads the list of most common english words and adds numbers to it as I assume 
# we don't want to count the use of numbers as a spelling mistake.

my_words = open("word_list.txt", "r") 
text1 = my_words.read()  #this is a text file with 1 word per line
words_into_list = text1.split("\n")  #split the text file at every line
words_into_list = words_into_list+[str(i) for i in range(0,1000000)] #add numbers to the list
word_set = set(words_into_list)



# The following function takes an essay and returns the number of spelling mistakes in it.

def mispellings(essay):
    errors=0
    clean_essay = preprocess_spelling(essay)
    for word in clean_essay: #loop through words in each essay
        if word not in word_set: #if a word is not contained in our word_set, add +1 to errors
            errors=errors+1
    return errors










