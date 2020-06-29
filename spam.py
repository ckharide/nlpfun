
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 

data = pd.read_csv('spam.csv', skiprows=1, engine='python')
print(data.head)

#Using TFID Vectorizer. 
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(data.iloc[:, 1])

print(X_train.shape)
