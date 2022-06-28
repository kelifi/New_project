# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 18:51:36 2022

@author: hkhelifi
"""

import lightgbm as lgb
#from sklearn.metrics import mean_absolute_percentage_error
#from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
from tools import get_time
from sklearn.ensemble import RandomForestRegressor
import math
#from sklearn import linear_model


def tester_selection(data , target, column):
    
    time_start1 = datetime.now()  
    
    #splitting the data 
    train , test= split_data(data,column)
    X_train, y_train= train.drop(columns=[target]),list(train[target].values)
    X_test, y_test= test.drop(columns=[target]),list(test[target].values)
    
    '''
    rf=RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_MAPE= mape(rf_pred,y_test)
    rf_squared= mean_squared_error(rf_pred,y_test)
    print("MAPE: ",rf_MAPE)
    print("mean squared error: ",rf_squared)
    final_time1=datetime.now()
    Time1=get_time(time_start1,final_time1)
    print('the duration of this procedure is equal to ',Time1 )
    '''
    
    
    
    time_start2 = datetime.now()  

    gbm = lgb.LGBMRegressor()
    #lr = linear_model.Lasso(alpha=0.1)
    gbm.fit(X_train, y_train)
    #lr.fit(X_train, y_train)
    y_pred = gbm.predict(X_test)
    #lr_pred = lr.predict(X_test)
    LG_MAPE= mape(y_pred,y_test)
    LG_squared= mean_squared_error(y_pred,y_test)
    
    
    
    print("MAPE: ",LG_MAPE)
    print("mean squared error: ",LG_squared)
    
    final_time2=datetime.now()
    Time2=get_time(time_start2,final_time2)
    print('the duration of this procedure is equal to ',Time2 )
    


#Defining MAPE function
def mape(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

#Defining a function to split a time series data
def split_data(data, column , threshold=np.nan):
    
    data[column]=data[column].astype('int')
    if not math.isnan(threshold) :
        assert threshold >= min(data[column]) and threshold <= max(data[column]), "The date is not in the data, please check your thresholds."
        #Sort the data by time
        data = data.sort_values(by=column)
        #Divide the data into two groups: a training set and a test set
        train_set=data.loc[data[column] <= threshold]
        test_set=data.loc[data[column] > threshold]
        return train_set,test_set

    else:
        threshold=int(.80 *(max(data[column])-min(data[column]))) + min(data[column])
        #Sort the data by time
        data = data.sort_values(by=column)
        #Divide the data into two groups: a training set and a test set
        train_set=data.loc[data[column] <= threshold]
        test_set=data.loc[data[column] > threshold]
        train_set[column]=data[column].astype('category')
        test_set[column]=data[column].astype('category')
        return train_set,test_set
    
    data[column]=data[column].astype('category')
    











