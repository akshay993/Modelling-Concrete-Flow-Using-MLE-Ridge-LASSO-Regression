import glmnet_python
from glmnet import glmnet
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#Importing Data from the Slump_Test Data File
df=pd.read_csv("../Data/slump_test.data")

#Removing the 'No' column from the data frame
df=df.drop('No',1)

data=df.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.','FLOW(cm)']]

for i in range(0,10):
        
    Train, Test=train_test_split(data,test_size=18)
    
    #Now creating 5 Fold with the 'Train' data
    kf = KFold(n_splits=5)
        
    for train_indices, test_indices in kf.split(Train):
        train_data = np.array(Train)[train_indices]
        test_data = np.array(Train)[test_indices]
        
        #Converting into numpy array to dataframe
        traindata=pd.DataFrame(train_data.reshape(68,8),columns=['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.','FLOW(cm)'])
        testdata=pd.DataFrame(test_data.reshape(17,8),columns=['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.','FLOW(cm)'])
      
        xTrain=traindata.loc[:,['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']]
        yTrain=traindata.loc[:,['FLOW(cm)']]
        
        