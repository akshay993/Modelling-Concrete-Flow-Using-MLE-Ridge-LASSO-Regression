import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt


#Importing Data from the Slump_Test Data File
df=pd.read_csv("../Data/slump_test.data")

#Removing the 'No' column from the data frame
df=df.drop('No',1)

#Data containing the 8 parameters (7 features and 1 response)
data=df.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.','FLOW(cm)']]


#Declaring variable/list which will be used in the code
mean_mse=0
mean_rsqd=0
list_mse=[]
list_rsqd=[]
list_ypred=[]
list_model_unreq=[]


mean_mse_ridge=0
mean_rsqd_ridge=0
list_mse_ridge=[]
list_rsqd_ridge=[]
list_ypred_ridge=[]
list_model_ridge=[]


mean_mse_lasso=0
mean_rsqd_lasso=0
list_mse_lasso=[]
list_rsqd_lasso=[]
list_ypred_lasso=[]
list_model_lasso=[]


#Running 10 iterations  
for i in range(0,10):
    
    #Creating training and test data   
    Train, Test=train_test_split(data,test_size=18, random_state=i)
    
    #Now creating 5 Fold with the 'Train' data
    kf = KFold(n_splits=5)
    
    min_mse=9999    
    best_fit=None
    min_mse_ridge=9999
    best_fit_ridge=None
    min_mse_lasso=9999
    best_fit_lasso=None
        
    #Performing k fold cross vlaidation
    for train_indices, test_indices in kf.split(Train):
        train_data = np.array(Train)[train_indices]
        test_data = np.array(Train)[test_indices]
        
        #Converting numpy array to dataframe
        traindata=pd.DataFrame(train_data.reshape(68,8),columns=['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.','FLOW(cm)'])
        testdata=pd.DataFrame(test_data.reshape(17,8),columns=['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.','FLOW(cm)'])
      
        xTrain=traindata.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']]
        yTrain=traindata.loc[:,['FLOW(cm)']]
        
        xTest=testdata.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']]
        yTest=testdata.loc[:,['FLOW(cm)']]
            
        
        #Unregularized Regression 
        lreg=LinearRegression() 
        lreg.fit(xTrain,yTrain)
        yPred = lreg.predict(xTest)
        mse = np.mean((yPred - yTest)**2)
        rSqd=lreg.score(xTest,yTest)
        
       
        #Storing minimum mse value and corresponding model 
        if mse.values[0]<min_mse:
            min_mse=mse.values[0]
            best_fit=lreg
        else:
            min_mse=min_mse
            
        
        #Ridge Regression
        ridgeReg = Ridge(alpha=0.15, normalize=True)
        ridgeReg.fit(xTrain, yTrain)
        yPred_ridge = ridgeReg.predict(xTest)
        mse_ridge = np.mean((yPred_ridge - yTest)**2)
        rSqd_ridge=ridgeReg.score(xTest,yTest) 
        
        #Storing minimum mse value and corresponding model
        if mse_ridge.values[0]<min_mse_ridge:
            min_mse_ridge=mse_ridge.values[0]
            best_fit_ridge=ridgeReg
        else:
            min_mse_ridge=min_mse_ridge
            
            
        #Lasso Regression
        lassoReg = Lasso(alpha=0.15, normalize=True)
        lassoReg.fit(xTrain,yTrain)
        yPred_lasso = lassoReg.predict(xTest)
        yTest1=yTest["FLOW(cm)"]
        mse_lasso = np.mean((yPred_lasso - yTest1)**2)
        rSqrd_lasso=lassoReg.score(xTest,yTest)
        
        #Storing minimum mse value and corresponding model
        if mse_lasso<min_mse_lasso:
            min_mse_lasso=mse_lasso
            best_fit_lasso=lassoReg
        else:
            min_mse_lasso=min_mse_lasso
        
            
    
    x_Test=Test.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']]
    y_Test=Test.loc[:,['FLOW(cm)']]
    
    #Predicting Y from the Linear Model created from 5-Fold and storing mse, rSqrd, yPred and model in lists
    y_Pred=best_fit.predict(x_Test)
    mse_new=np.mean((y_Pred - y_Test)**2)
    rSqd_new=best_fit.score(x_Test,y_Test)    
    list_mse.append(mse_new.values[0])
    list_rsqd.append(rSqd_new)
    list_ypred.append(y_Pred)
    list_model_unreq.append(best_fit)

    
    #Predicting Y from the Ridge Model created from 5-Fold and storing mse, rSqrd, yPred and model in lists
    y_Pred_ridge=best_fit_ridge.predict(x_Test)
    mse_new_ridge=np.mean((y_Pred_ridge - y_Test)**2)
    rSqd_new_ridge=best_fit_ridge.score(x_Test,y_Test)
    list_mse_ridge.append(mse_new_ridge.values[0])
    list_rsqd_ridge.append(rSqd_new_ridge)
    list_ypred_ridge.append(y_Pred_ridge)
    list_model_ridge.append(best_fit_ridge)

    
    #Predicting Y from the Lasso Model created from 5-Fold and storing mse, rSqrd, yPred and model in lists    
    y_Pred_lasso=best_fit_lasso.predict(x_Test)
    y_Test1=y_Test["FLOW(cm)"]
    mse_new_lasso=np.mean((y_Pred_lasso - y_Test1)**2)
    rSqd_new_lasso=best_fit_lasso.score(x_Test,y_Test)
    list_mse_lasso.append(mse_new_lasso)
    list_rsqd_lasso.append(rSqd_new_lasso)
    list_ypred_lasso.append(y_Pred_lasso)
    list_model_lasso.append(best_fit_lasso)

    

#Mean value of MSE for MLE, Ridge and Lasso    
mean_mse=sum(list_mse)/float(len(list_mse))
mean_mse_ridge=sum(list_mse_ridge)/float(len(list_mse_ridge))
mean_mse_lasso=sum(list_mse_lasso)/float(len(list_mse_lasso))
            
#Mean value of RSqd for MLE, Ridge and Lasso
mean_rsqd=sum(list_rsqd)/float(len(list_rsqd))
mean_rsqd_ridge=sum(list_rsqd_ridge)/float(len(list_rsqd_ridge))
mean_rsqd_lasso=sum(list_rsqd_lasso)/float(len(list_rsqd_lasso))
    
             
#Choosing Best Y prediction(one which has least MSE) for unreg 
index_1=list_mse.index(min(list_mse))
best_ypred=list_ypred[index_1]

#Choosing Best Y Prediction(one which has least MSE) for Ridge
index_2=list_mse_ridge.index(min(list_mse_ridge))
best_ypred_ridge=list_ypred_ridge[index_2]


#Choosing Best Y prediction(one which has least MSE) for Lasso
index_3=list_mse_lasso.index(min(list_mse_lasso))
best_ypred_lasso=list_ypred_lasso[index_3]

#Choosing Best Model for unreg , ridge and lasso (one which has last MSE)
best_model_unreg=list_model_unreq[index_1]
best_model_ridge=list_model_ridge[index_2]
best_model_lasso=list_model_lasso[index_3]


#Storing coefficients of Best Model of unreg, ridge and lasso
coef_unreg=best_model_unreg.coef_
coef_ridge=best_model_ridge.coef_
coef_lasso=best_model_lasso.coef_

#Drawing plot between Y Predicticed vs Y Test for MLE, Ridge and Lasso
plt.scatter(y_Test.values.tolist(), best_ypred, marker='v', label='Unregularized')
plt.scatter(y_Test.values.tolist(),best_ypred_ridge, marker='x', label='Ridge')
plt.scatter(y_Test.values.tolist(),best_ypred_lasso, marker='s', label='Lasso')
plt.xticks(np.arange(10,90,step=10))
plt.xlabel("Y Test")
plt.ylabel("Y Predicted")
plt.legend()



        
        