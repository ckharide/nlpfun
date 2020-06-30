
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np


df = pd.read_csv('spam.csv', engine = 'python')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],axis =1)
mydata = df[["v1", "v2"]]
mydata['labels'] = mydata['v1'].map({'ham' : 0, 'spam' : 1})
mydata['v2'] = mydata['v2'].str.lower()
mydata = mydata.drop(['v1'], axis=1)
mydata = mydata.drop(mydata.index[0])
#print(mydata.head(2))

#print(mydata.shape)


#Using TFID Vectorizer. 
#vectorizer = CountVectorizer(decode_error='ignore')
vectorizer = TfidfVectorizer(decode_error='ignore')
X_train_counts = vectorizer.fit_transform(mydata['v2'])

X_train, X_test, y_train, y_test = train_test_split(X_train_counts, mydata['labels'], random_state = 0)

#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train, y_train)
#print(X_train.shape)



predicted = clf.predict(X_test)
print(np.mean(predicted == y_test))