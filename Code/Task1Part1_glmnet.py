#Linear Regression
import glmnet_python
from glmnet import glmnet
from glmnet import glmnet; from glmnetPlot import glmnetPlot 
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


#Importing Data from the Slump_Test Data File
df=pd.read_csv("../Data/slump_test.data")

#Removing the 'No' column from the data frame
df=df.drop('No',1)

#Setting Up Exploratory Variables (x) and Response Variable (y)
x=df.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']]
y=df.loc[:,['FLOW(cm)']]

#Creating Training and Testing Set (85 Training Set and 18 Testing Set)
xTrain, xTest, yTrain, yTest=train_test_split(x,y,test_size=18)

#xTrain=xTrain.values
#yTrain=yTrain.values


#Calling glmnet
fit = cvglmnet(x = xTrain.values, y = yTrain.values, ptype='mse')
