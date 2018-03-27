import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


#Importing Data from the Slump_Test Data File
df=pd.read_csv("../Data/slump_test.data")

#Removing the 'No' column from the data frame
df=df.drop('No',1)

#Setting Up Exploratory Variables (x) and Response Variable (y)
x=df.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']]
y=df.loc[:,['FLOW(cm)']]


 
#Creating Training and Testing Set (85 Training Set and 18 Testing Set)

alpha=1

for i in range(0,20,2):
    
    xTrain, xTest, yTrain, yTest=train_test_split(x,y,test_size=18)
    #Creating Ridge Regression model
    ridgeReg = Ridge(alpha, normalize=True)
    
    #Training the Model
    ridgeReg.fit(xTrain, yTrain)
    
    #Predicting on the Testing set
    yPred = ridgeReg.predict(xTest)
    
    #Calculating Mean Squared Error
    mse = np.mean((yPred - yTest)**2)
    
    #Calculating R-Squared
    rSqd=ridgeReg.score(xTest,yTest) 
    
    
    if i==0:
        r_list=rSqd
        
    else:
        r_list=np.append(r_list,[rSqd])
        
        
