# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
