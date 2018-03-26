import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
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

a=0.5

list_model_ridge=[]
coefs=[]
alphas=[]

for i in range(0,100):
    
    ridgeReg = Ridge(alpha=a, normalize=True)
    ridgeReg.fit(xTrain, yTrain)
    yPred = ridgeReg.predict(xTest)
    
    list_model_ridge.append(ridgeReg)    
    coefs.append(ridgeReg.coef_[0])
    alphas.append(a)
    
    a=a+0.5
    

aalphas=np.asarray(alphas)

for i in range(0,100):
 #   aalphas[i]=-np.log10(aalphas[i])
     aalphas[i]=log(aalphas[i]) 
#for i in range(0,20):
#    coefs[i] = coefs[i][0]
    
ax = plt.gca()

ax.plot(aalphas, coefs)
#ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis

plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()




a=0.05
yPred=0

list_model_lasso=[]
coefs=[]
alphas=[]

for i in range(0,100):
    
    lassoReg = Lasso(alpha=a, normalize=True)
    lassoReg.fit(xTrain, yTrain)
    yPred = lassoReg.predict(xTest)
    
    list_model_lasso.append(lassoReg)    
    coefs.append(lassoReg.coef_[0])
    alphas.append(a)
    
    a=a+0.05
    

yTrain1=yTrain
yTrain2=[]
for i in range(len(yTrain1)):
    yTrain2.append(yTrain1[i][0])
yTrain = np.array(yTrain2)    


alphas, _, coefs = linear_model.lars_path(xTrain, yTrain, method='lasso', verbose=True)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()