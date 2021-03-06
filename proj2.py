#Machine Learning Project 2
#   Authors:
#   Sankarshan Araujo
#   Logistic regression and SVM for IMDB reviews

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re

count = CountVectorizer()
docs = pd.read_csv("IMDBDataset.csv")
bag = count.fit_transform(docs)
print(count.get_feature_names())
print(count.vocabulary_)
print(bag.toarray())
np.set_printoptions(precision=2)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())

