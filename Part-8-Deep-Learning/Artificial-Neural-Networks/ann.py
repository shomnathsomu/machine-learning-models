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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
