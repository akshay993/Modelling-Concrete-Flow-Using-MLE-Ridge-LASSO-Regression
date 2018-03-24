import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

#Importing Data from the Slump_Test Data File
df=pd.read_csv("../Data/slump_test.data")

#Removing the 'No' column from the data frame
df=df.drop('No',1)

#Setting Up Exploratory Variables (x) and Response Variable (y)
x=df.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']]
y=df.loc[:,['FLOW(cm)']]

#Creating Training and Testing Set (85 Training Set and 18 Testing Set)
xTrain, xTest, yTrain, yTest=train_test_split(x,y,test_size=18)

#Creating Lasso Regression model
lassoReg = Lasso(alpha=0.3, normalize=True)

#Training the Model
lassoReg.fit(xTrain,yTrain)

#Predicting on the Testing set
yPred = lassoReg.predict(xTest)

# calculating Mean Squared Error
mse = np.mean((yPred - yTest)**2)

#Calculating R-Squared
rSqrd=lassoReg.score(xTest,yTest)

