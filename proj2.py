#Machine Learning Project 2
#   Authors:
#   Sankarshan Araujo
#   Logistic regression and SVM

# 3/6/21 KNN model of different wheat seeds
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import cm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d   # must keep
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
data = pd.read_csv('data.csv')

X = data[['Yards',]]
y = data['Success']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.25)
df = pd.DataFrame(data = data)
#kNN set up
knn = KNeighborsClassifier(n_neighbors = 7, weights = 'distance')
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
actual = [0,0,0]
predict = [0,0,0]
preds=knn.predict(X_test)
# confusion matrix
matrix = confusion_matrix(y_test, preds)
print('Confusion matrix : \n',matrix)

matrix = classification_report(y_test, preds)
print('Classification report : \n',matrix)

#Showing accuracy according to n_neighnors
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance', metric = 'minkowski', p = 2)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

#Plotting
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])
#plt.show()

#Showing the training set proportion
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance', metric = 'minkowski', p = 2)
plt.figure()
for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
#Showing plots
plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
#plt.show()



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

y_pred = lr.predict(X_test)
# confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix : \n',matrix)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_test, y_pred)
print('Classification report : \n',matrix)

#linear SVC model
#regularization parameter 1,000 and max iterations at max possible
lsvc = LinearSVC(C = 1000, max_iter = 100000)
lsvc.fit(X_train, y_train)
print("Linear SVM Training set score: {:.2f}%".format(100 * lsvc.score(X_train, y_train)))
print("Linear SVM Test set score: {:.2f}%".format(100 * lsvc.score(X_test, y_test)))
lsvc.predict(X_test)
print(lsvc.coef_)
print(lsvc.intercept_)

y_pred = lsvc.predict(X_test)
# confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix : \n',matrix)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_test, y_pred)
print('Classification report : \n',matrix)
