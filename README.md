# Modelling-Concrete-Flow-Using-MLE-Ridge-Lasso-Regression
Modeling slump flow of concrete using MLE, Ridge and LASSO regression

This is done as a Programming Assignment for CSE 574 -  Introduction to Machine Learning Class.

Description of Data set:
1.The data set contains 103 observations.
2.Explanatory Variables are:
  'Cement','Slag','Fly ash','Water','SP','Coarse Aggr.' and 'Fine Aggr.'. Response Variable is: 'FLOW(cm)'.

Task 1:
1.Implemented code to randomly select 85 observations for 5-fold cross validation (5 x 17; 4 x 17 training set; 17 validate set) and 18 observations for a test set.
2.Performed the Unregularized, Regularized L2(Ridge) and Regularized L1(LASSO) regressions saving the best model as determined by cross validation.
3.Iterated this process 10 times.
4.Ultimately, got the Best model for Unregularized, Regularized L2(Ridge) and Regularized L1(LASSO) regressions respectively (the model having least Mean Square Error).
5.Plotted graph for Y-Prediction vs Y-Test for the Unregularized, Regularized L2(Ridge) and Regularized L1(LASSO) regressions.


Task 2:
Plot Regularization Paths for Ridge and LASSO.


Files: 
1.All the code is in the 'Code' Folder.
2.Task0.py contains implementation of Task 0 and 1
3.Task2.py contains implementation of  Regularization paths for Ridge and LASSO.
4.'Data' folder contains the data used. 
5.'slump_test.data' is the data file and 'slump_test.names' is the file containing the description for the data.
6.Report folder contains the report


