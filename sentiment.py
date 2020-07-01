from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import numpy as np 
import pandas as pd 
import nltk as nl
with open("positive.review.txt", "r") as f:
    contents = f.read()
    soup = BeautifulSoup(contents, 'lxml')
    positive_review = soup.find_all('review_text')

with open("negative.review.txt", "r") as f:
    contents = f.read()
    soup = BeautifulSoup(contents, 'lxml')
    negative_review = soup.find_all('review_text')

#nl.download('punkt')
np.random.shuffle(positive_review)
positive_review = positive_review[:len(negative_review)]
print(type(positive_review))
word_index = {}
print ("Length" , len(positive_review) , len(negative_review))
positive_review_list = [word for word in positive_review if str(word) != 'nan']

words_map = {}
words = nl.word_tokenize(str(positive_review))
print (type(words))
for word in words:
    if word in words_map.keys():
        words_map[word] += 1
    else:
        words_map[word] = 1

print(words_map)