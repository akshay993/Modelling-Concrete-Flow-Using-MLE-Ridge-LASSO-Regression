import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

#Importing Data from the Slump_Test Data File
df=pd.read_csv("../Data/slump_test.data")

#Removing the 'No' column from the data frame
df=df.drop('No',1)

#Setting Up Exploratory Variables (x) and Response Variable (y)
x=df.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']]
y=df.loc[:,['FLOW(cm)']]



for i in range(0,10):
        
    #Creating Training and Testing Set (85 Training Set and 18 Testing Set)
    xTrain, xTest, yTrain, yTest=train_test_split(x,y,test_size=18)
        
    #Creating Linear Regression Model
    lreg=LinearRegression()
    
    #Training the Model
    cvfit=lreg.fit(xTrain,yTrain)
    
    #Predicting on the testing data
    yPred = lreg.predict(xTest)
    
    #Calculating Coefficients/Weights
    #coeff = DataFrame(xTrain.columns)
    #coeff['Coefficient Estimate'] = Series(lreg.coef_)

    #Calculating the Mean Squared Error
    mse = np.mean((yPred - yTest)**2)
    
    #Calculating R-Squared
    rSqd=lreg.score(xTest,yTest)
    
    if i==0:
        mse_list=mse
    else:
        mse_list=np.append(mse_list,[mse])
        
                
mean_mse=np.mean(mse_list)



