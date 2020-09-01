
from sklearn.naive_bayes import MultinomialNB

import pandas as pd 
import numpy as np 
data = pd.read_csv('spambase.data.csv')
Y = data.iloc[:, -1]
X = data.iloc[:, :48]

# print the first 100 rows . 
X_train = X[:-100]
Y_train = Y[:-100]

# last 100 rows. 
X_test = X[-200:]
Y_test = Y[-200:]

model = MultinomialNB().fit(X_train, Y_train)
Y_predict = model.predict(X_test)
print(np.mean(Y_predict == Y_test))
print("MUltiNomial BS classification score " , model.score(X_test,Y_test))

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier().fit(X_train, Y_train)
print("AdaBoost Classifier BS classification score " , model.score(X_test,Y_test))
