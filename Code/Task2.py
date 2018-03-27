import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from math import log, exp


#Importing Data from the Slump_Test Data File
df=pd.read_csv("../Data/slump_test.data")

#Removing the 'No' column from the data frame
df=df.drop('No',1)

#Setting Up Exploratory Variables (x) and Response Variable (y)
x=df.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']]
y=df.loc[:,['FLOW(cm)']]

#Creating Training and Testing Set (85 Training Set and 18 Testing Set)
xTrain, xTest, yTrain, yTest=train_test_split(x,y,test_size=18)
xTrain=np.asarray(xTrain)
yTrain=np.asarray(yTrain)

#Choosing initial alpha value
a=0.5

#Declaring lists which will be used in the code
coefs=[]
alphas=[]

#Procedure for creating Regularization Path for Ridge Regression

#Iterating 100 Times increasing alpha each time by 0.5
#The reason we are taking high value of alpha here is because ultimaltely, we will be taking the log of alpha
for i in range(0,100):
    
    #Creating ridge regression model and training it with the test data
    ridgeReg = Ridge(alpha=a, normalize=True)
    ridgeReg.fit(xTrain, yTrain)
    #yPred = ridgeReg.predict(xTest)
    
    #storing the coefficients of the model created in a list
    coefs.append(ridgeReg.coef_[0])
    alphas.append(a)
    
    #incrementing alpha for next iteration
    a=a+0.5
    

#Conversion into array
aalphas=np.asarray(alphas)

#Converting each alpha value into log alpha
for i in range(0,100):
 #   aalphas[i]=-np.log10(aalphas[i])
     aalphas[i]=log(aalphas[i]) 
#for i in range(0,20):
#    coefs[i] = coefs[i][0]
   
    
#Plotting the Regularization path (between coefficients and alpha values) for Ridge Regression
ax = plt.gca()

ax.plot(aalphas, coefs)
#ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis

plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('Regularization Path for Ridge Regression')
plt.axis('tight')
plt.show()


#Now, Procedure for creating Regularization path for LASSO Regression

yTrain1=yTrain
yTrain2=[]
for i in range(len(yTrain1)):
    yTrain2.append(yTrain1[i][0])
yTrain = np.array(yTrain2)    

#Here, we extract alphas and coefficients using linear_model.lars_path method
#LARS is used for computing Regularization Path for LASSO 
#Read About it on scikit-learn.org
alphas, _, coefs = linear_model.lars_path(xTrain, yTrain, method='lasso', verbose=True)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

#Now, we plot the graph between the coefficients and the L1 Norm
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.xlabel('L1 Norm')
plt.ylabel('Coefficients')
plt.title('Regularization Path for LASSO Regression')
plt.axis('tight')
plt.show()