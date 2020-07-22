from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import numpy as np 
import pandas as pd 
import nltk as nltk
with open("positive.review.txt", "r") as f:
    contents = f.read()
    soup = BeautifulSoup(contents, 'lxml')
    positive_review = soup.find_all('review_text')
# Create a placeholder for model

stopwords = set(w.rstrip() for w in open('stopwords.txt'))
wordnet_lemmatizer = WordNetLemmatizer()
positive_tokenized = []
orig_reviews = []
word_map = {}
dict_prob  = {}

print (positive_review[:2])
print("*******************")

def my_tokenizer(s):
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    return tokens

def get_ngrams(text, n ):
    n_grams = ngrams(nltk.word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]

def get_total_count(tokenname):
    count = 0
    length = 0
    for review in positive_review:
      review = review.text.lower()
      tokens = get_ngrams(review,3)
      length += len(tokens)
      if(tokenname == tokens[1]):
          print ("Orig Retrieved %s %s " % (tokenname, tokens[1]))
          count+=1
    print (count / length)
    return count / length
    
for review in positive_review[:2]:
    #print("Review, %s!" % review)
    review = review.text.lower()
    tokens = get_ngrams(review,3)
    tokens_filtered = tokens.split()
    for token in tokens_filtered:
        print (token)

    ''''  
    ngramtokens = my_tokenizer(tokens)
    positive_tokenized.append(review.text.lower())
    for ngramtokens in tokens:
        first_word = ngramtokens[0]
        second_word = ngramtokens[2]
        print("First Second %s %s" %(first_word , second_word))
        if ngramtokens not in dict_prob:
            dict_prob[tokens[1]] = 1/get_total_count(ngramtokens[1])
        else:
            dict_prob[tokens[1]] += 1/get_total_count(ngramtokens[1])'''
        

def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_map) + 1) # last element is for the label
    for t in tokens:
        i = dict_prob[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
