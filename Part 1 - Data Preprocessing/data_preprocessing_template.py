# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  #take all rows and all columns except the last one
y = dataset.iloc[:, 3].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)  # ie, 20% of the data for test. other good choices are 25%, 22% etc. a 50% is very bad. Y???

# Feature Scaling
# since the values across the columns are not in the same range (age from 27 to 50 and salary from 52k to 83k), we need to make them be within same range (-1 to +1) so as to avloid errors in our models
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train) 
#X_test = sc_X.transform(X_test)       # we only transform the x_test
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)  #we do not need to xform y_test