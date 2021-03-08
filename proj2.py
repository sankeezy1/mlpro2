#Machine Learning Project 2
#   Authors:
#   Sankarshan Araujo
#   Logistic regression and SVM

# 3/6/21 KNN model of different wheat seeds
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Get data from csv
data = pd.read_csv('data.csv')

# Put data in dataframe
df = pd.DataFrame(data = data)

X = df[["Yards"]]
y = df["Success"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

# kNN set up
knn = KNeighborsClassifier(n_neighbors = 7, weights = 'distance')
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
preds = knn.predict(X_test)

# Display kNN scores
print("kNN training set score: {:.2f}%".format(knn.score(X_train, y_train)))
print("kNN test set score: {:.2f}%".format(knn.score(X_test, y_test)))

# kNN confusion matrix
matrix = confusion_matrix(y_test, preds)
print('kNN Confusion matrix : \n',matrix)

# kNN classification report
report = classification_report(y_test, preds)
print('kNN Classification report : \n', report)

# Showing accuracy according to n_neighbors from 0 to 20
k_range = range(1, 100)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance', metric = 'minkowski', p = 2)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# Plotting kNN values
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Showing the training set proportion
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance', metric = 'minkowski', p = 2)
plt.figure()
for s in t:
    scores = []
    for i in range(1, 100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

# Showing plots
plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')

# Logistic regression model
# Max iterations set to highest possible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
lr = LogisticRegression(max_iter = 10000)
lr.fit(X_train, y_train)

# Display LR scores
print("Logistic training set score: {:.2f}%".format(lr.score(X_train, y_train)))
print("Logistic test set score: {:.2f}%".format(lr.score(X_test, y_test)))

# LR prediction plot
x_range = 100
Xs = [i for i in range(x_range)]
Ys = [lr.predict_proba([[value]])[0][1] for value in range(x_range)]

plt.figure()
plt.scatter(X, y)
plt.xlabel('Yards')
plt.ylabel('Probability')
plt.plot(Xs, Ys, color='red')

# LR confusion matrix
y_pred = lr.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
print('Logistic confusion matrix : \n',matrix)

# LR classification report for precision, recall f1-score and accuracy
report = classification_report(y_test, y_pred)
print('Logistic classification report : \n',report)

# Linear SVC model
# Regularization parameter 1,000 and max iterations at max possible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lsvc = LinearSVC(C = 100, max_iter = 10000, dual = False)
lsvc.fit(X_train, y_train)

# Display SVC scores
print("Linear SVC Training set score: {:.2f}%".format(100 * lsvc.score(X_train, y_train)))
print("Linear SVC Test set score: {:.2f}%".format(100 * lsvc.score(X_test, y_test)))



# SVC confusion matrix
y_pred = lsvc.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
print('linear SVC confusion matrix : \n',matrix)

# SVC classification report for precision, recall f1-score and accuracy
report = classification_report(y_test, y_pred)
print('Linear SVC classification report : \n',report)

# Print plots
# plt.show()
