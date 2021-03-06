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
dataset = pd.DataFrame(data = data)
# print(X)


