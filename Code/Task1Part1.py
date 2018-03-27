import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def linearregressionfunction(xTrain,xTest,yTrain,yTest):
        
    #Creating Training and Testing Set (85 Training Set and 18 Testing Set)
    #xTrain, xTest, yTrain, yTest=train_test_split(x,y,test_size=18)
        
    #Creating Linear Regression Model
    lreg=LinearRegression()
    
    #Training the Model
    lreg.fit(xTrain,yTrain)
    
    #Predicting on the testing data
    yPred = lreg.predict(xTest)
    
    #Calculating Coefficients/Weights
    #coeff = DataFrame(xTrain.columns)
    #coeff['Coefficient Estimate'] = Series(lreg.coef_)

    #Calculating the Mean Squared Error
    mse = np.mean((yPred - yTest)**2)
    
    #Calculating R-Squared
    rSqd=lreg.score(xTest,yTest)
        
    lobj=np.append(mse,rSqd)
    lobj=np.append(lobj,lreg)
    
    return lobj


def ridgeregressionfunction(xTrain,xTest,yTrain,yTest):
    
    #Creating Ridge Regression model
    ridgeReg = Ridge(alpha=1)
    
    #Training the Model
    ridgeReg.fit(xTrain, yTrain)
    
    #Predicting on the Testing set
    yPred = ridgeReg.predict(xTest)
    
    #Calculating Mean Squared Error
    mse = np.mean((yPred - yTest)**2)
    
    #Calculating R-Squared
    rSqd=ridgeReg.score(xTest,yTest) 
    
    robj=np.append(mse,rSqd)
    
    return robj


def lassoregressionfunction(xTrain,xTest,yTrain,yTest):
    
    #Creating Lasso Regression model
    lassoReg = Lasso(alpha=1)
    
    #Training the Model
    lassoReg.fit(xTrain,yTrain)
    
    #Predicting on the Testing set
    yPred = lassoReg.predict(xTest)
    
    yTest1=yTest["FLOW(cm)"]
    
    #Calculating Mean Squared Error
    mse = np.mean((yPred - yTest1)**2)
    
    #Calculating R-Squared
    rSqrd=lassoReg.score(xTest,yTest)

    lassobj=np.append(mse,rSqrd)
    
    return lassobj






 
    





    
        
                




