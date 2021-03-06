#Machine Learning Project 2
#   Authors:
#   Sankarshan Araujo
#   Logistic regression and SVM for IMDB reviews

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data.csv")
df = pd.DataFrame(data = data)

X = df[["Yards"]]
y = df["Success"]
plt.plot(X, y, 'ro')
plt.ylabel('success')
plt.xlabel('yards')
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LogisticRegression()
lr.fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

yards = 52
pass_prediction = lr.predict([[yards]])
pass_probability = lr.predict_proba([[yards]])
print("pass: {}".format(pass_prediction[0]))
print("fail/pass probability: {}".format(pass_probability[0]))


