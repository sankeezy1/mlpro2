#Machine Learning Project 2
#   Authors:
#   Sankarshan Araujo
#   Logistic regression, SVM, and kNN for all NFL field goals in 2003

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

#read data into dataframe
data = pd.read_csv("data.csv")
df = pd.DataFrame(data = data)

# #set columns 'Yards' and 'Success' to X and y
X = df[["Yards"]]
y = df["Success"]
# plt.plot(X, y)
# plt.ylabel('goal success')
# plt.xlabel('yards')
# plt.show()

#input user test size
inputSize = input("Enter an appropriate test size for training the model\n\n")
inputSize = float(inputSize)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = inputSize)

#logistic regression model
#max iterations set to highest possible
lr = LogisticRegression(max_iter = 100000)
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

#linear SVC model
#regularization parameter 1,000 and max iterations at max possible
lsvc = LinearSVC(C = 1000, max_iter = 100000)
lsvc.fit(X_train, y_train)
print("Linear SVM Training set score: {:.2f}%".format(100 * lsvc.score(X_train, y_train)))
print("Linear SVM Test set score: {:.2f}%".format(100 * lsvc.score(X_test, y_test)))
lsvc.predict(X_test)
print(lsvc.coef_)
print(lsvc.intercept_)
