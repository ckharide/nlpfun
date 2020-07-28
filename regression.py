import numpy as nlp 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('airfoil_self_noise.dat',sep='\t',header=None)
print (df.head())
#print (df.info())

print(type(df))

print(df.shape)

data = df[[0,1,2,3,4]].values
target = df[5].values

print(data.shape)

mydata = df.iloc[:, [0,1,2,3,4]].values
print(mydata.shape)

X_train , X_test , Y_train , Y_test = train_test_split(data,target,test_size=0.33)
lf = LinearRegression()
lf.fit(X_train, Y_train)
print(lf.score(X_test,Y_test))

preds = lf.predict(X_test)
print(preds)