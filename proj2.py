#Machine Learning Project 2
#   Authors:
#   Sankarshan Araujo
#   Logistic regression and SVM for IMDB reviews

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
dataset = pd.DataFrame(data = data)

plt.plot(dataset["Yards"], dataset["Goal Success"], 'ro')
plt.ylabel('Goal Success')
plt.xlabel('Yards')
plt.show()

