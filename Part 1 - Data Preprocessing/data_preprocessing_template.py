# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  #take all rows and all columns except the last one
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# categorical data such as country and Purchased need to be encoded into some numeric form
# their present alphanumeric forms will be problematic later when fit into a machine learning model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])     # This encodes the countries with codes 0,1,2. Doing this will however deceive the computer that the encoded values are ranked. That is 2>1>0. and tempting to say Spain>Germany>France. One way to solve this is to use oneHotEncoder (which is in bits form. 3bits for 3 data)
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)  # ie, 20% of the data for test. other good choices are 25%, 22% etc. a 50% is very bad. Y???

# Feature Scaling
# since the values across the columns are not in the same range (age from 27 to 50 and salary from 52k to 83k), we need to make them be within same range (-1 to +1) so as to avloid errors in our models
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)       # we only transform the x_test
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)  #we do not need to xform y_test