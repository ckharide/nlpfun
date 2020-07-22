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

nl.download('punkt')
nl.download('averaged_perceptron_tagger')
nl.download('maxent_ne_chunker')
nl.download('words')
np.random.shuffle(positive_review)
positive_review = positive_review[:len(negative_review)]
print(type(positive_review))
word_index = {}
print ("Length" , len(positive_review) , len(negative_review))
#positive_review_list = [word for word in positive_review if str(word) != 'nan']

print("First Reivew is " , positive_review[0])

words_map = {}
for positive_review_token in positive_review[0]:
  words = nl.word_tokenize(str(positive_review_token))
  for word in words:
        if word in words_map.keys():
            words_map[word] += 1
        else:
            words_map[word] = 1


for negative_review_token in negative_review[0]:
  words = nl.word_tokenize(str(negative_review_token))
  for word in words:
        if word in words_map.keys():
            words_map[word] += 1
        else:
            words_map[word] = 1

def token_vector(tokens, label):
    tokenarray = np.zeros(1 + len(words_map))
    for token in tokens:
        index = words_map[token]
        tokenarray[index] +=1
    tokenarray  = tokenarray/tokenarray.sum()
    tokenarray[-1] = label

N =  len(positive_review) + len(negative_review)
#data = np.zeros(N, len(words_map) + 1)

print(N)

s = "Issac Newton was born in the 17th Century"
s1 = "My son loves playing with legos"
tags = nl.pos_tag(s1.split())
print(nl.ne_chunk(tags))

nl.ne_chunk(tags).draw()


print(nl.pos_tag(("Akshay is a good boy but sometimes he shows peevish behaviour").split()))