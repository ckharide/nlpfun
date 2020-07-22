import nltk 
from nltk import WordNetLemmatizer

nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()
titles = [line.rstrip() for line in open('all-books.txt')]
stopswords = [word.rstrip() for word in open('stopwords.txt')]


def my_tokenizer(str):
    str = str.lower()
    tokens = nltk.word_tokenize(str)
    print("Original Tokens " , len(tokens))
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    #tokens = [t for t in tokens if t not in stopswords]
    print("Tokens Length  ==> " , len(tokens))
    return tokens


for title in titles[:5]:
    print("Token title " , title)
    tokenfortitle = my_tokenizer(title)
    for myword in tokenfortitle:
        print(myword,end=" ")


