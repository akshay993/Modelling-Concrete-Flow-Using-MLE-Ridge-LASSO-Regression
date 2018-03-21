import glmnet_python
from glmnet import glmnet
import numpy as np
import pandas as pd

#Importing Data from the Slump_Test Data File
df=pd.read_csv("../Data/slump_test.data")

#Removing the 'No' column from the data frame
df=df.drop('No',1)

#Randomly Selecting 85 Observations
for i in range(0,85):
    
    
    
    
