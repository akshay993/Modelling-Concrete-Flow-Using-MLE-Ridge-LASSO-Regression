import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#Importing Data from the Slump_Test Data File
df=pd.read_csv("../Data/slump_test.data")

#Removing the 'No' column from the data frame
df=df.drop('No',1)

#Creating Linear Regression Object
lm=LinearRegression()

