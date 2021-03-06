#Machine Learning Project 2
#   Authors:
#   Sankarshan Araujo
#   Logistic regression and SVM for IMDB reviews

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("IMDBDataset.csv")

print("\nattributes in the dataset: "+ str(len(data.columns)))

print("\ndata size: " + str(len(data.index)))

X = data['review'].values
y = data['sentiment'].values

# Encoding strings for classifier
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(X)

print('-'*80)
print(f'Shape of X is {X.shape}\nShape of y is {y.shape}')
X_train, X_test, y_train, y_test = train_test_split(le, y, test_size = 0.2, random_state = 0)
print('-'*80)
print(f"Length of X_train: {len(X_train)}\nLength of X_test: {len(X_test)}")
print(f"Length of y_train: {len(y_train)}\nLength of y_test: {len(y_test)}")

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
print(classifier)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)
print("{:.0%}".format(accuracy_score(y_test,y_pred)))
