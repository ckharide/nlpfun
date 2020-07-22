import nltk as nltk
from nltk.book import *
from urllib import request

print(len(text1))
print(text1.count("world"))

print(len(set(text1)))

'''for w in text1:
    print("Word  ==> %s  Count ==> %s" % (w,len(w)))'''

fist = FreqDist(text1)
vocubalary = fist.keys()
#print(vocubalary)
print(text1)

fdist = FreqDist(w for w in text1)
max = fdist.max()
print(max)
print(fdist.max())

print (fdist.freq(max))

url = "http://www.gutenberg.org/files/2554/2554-0.txt"
resp = request.urlopen(url)
raw = resp.read().decode('utf8')
tokens = nltk.word_tokenize(raw)
print ("Tokens List %s " %len(tokens))
text = nltk.Text(tokens)
print(text[0:10])
words = [ w.lower() for w in tokens]
vocab= []
vocab = sorted(set(words))
print(type(words))
print(vocab[1000:1010])
