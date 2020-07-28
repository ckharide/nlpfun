import numpy as np 

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
print (data.keys())
print (data.feature_names)
print (data.target)
print (data.target_names)

from sklearn.model_selection import train_test_split

X_train , X_test, Y_train , Y_test = train_test_split(data.data, data.target, test_size=0.4)
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
print("Model Train accuracy ", rfc.score(X_train,Y_train))
print("Model test accuracy   ", rfc.score(X_test,Y_test))

print(rfc.predict(X_test))
N = len(Y_test)
predictions = rfc.predict(X_test)
print(np.sum(predictions == Y_test)/N)