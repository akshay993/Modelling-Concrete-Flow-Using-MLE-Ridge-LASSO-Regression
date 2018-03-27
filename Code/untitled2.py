from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets



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
#yTrain=np.asarray(yTrain)
yTrain1=yTrain.values
yTrain2=[]
for i in range(len(yTrain1)):
    yTrain2.append(yTrain1[i][0])
yTrain = np.array(yTrain2)    
    
eps = 5e-3

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(xTrain, yTrain, eps, fit_intercept=False)

print("Computing regularization path using the positive lasso...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    xTrain, yTrain, eps,  fit_intercept=False)


plt.figure(1)
ax = plt.gca()

colors = cycle(['b', 'r', 'g', 'c', 'k'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)

for coef_l, c in zip(coefs_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
    
    