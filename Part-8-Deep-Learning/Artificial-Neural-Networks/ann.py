# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
dataset['Geography'] = labelencoder_X_1.fit_transform(dataset['Geography'])
labelencoder_X_2 = LabelEncoder()
dataset['Gender'] = labelencoder_X_2.fit_transform(dataset['Gender'])
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
