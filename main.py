# -*- coding: utf-8 -*-
"""
Created on Sun May 22 07:38:38 2022

@author: hkhelifi
"""
# Import Libraries


import scipy.stats as stats
from scipy.stats import normaltest
import pandas as pd
from simple_imputation.clustring_and_lightgbm import (
    intersection
    
)
from deletion.delete_columns import delete_columns
from recommendation.impute_by_recommended import impute_by_recommended
from tools import create_lists, encoding
from outliers.outliers import outlier_detect_IQR,outlier_detect_mean_std,windsorization
from outliers.rare_values import GroupingRareValues
import yaml
from feature_selection.lgbm_importance import random_bar
import warnings
from feature_selection.mrmr import FCD,FCQ,RFCQ,mRMR
from recommendation.utils import check_p_val 
warnings.filterwarnings("ignore")


with open(
    "C:/Users/hkhelifi/OneDrive - Infor/Desktop/PFE_Git/science/2022/docs/The_configurable_parameters.yml",
    "r",
) as stream:
    config_parameters = yaml.safe_load(stream)


def main(data,target, parameters=config_parameters):
    """Imputes a series, dataframe  with
    the best imputation method for the given data.

    :param data: The data that should be imputed.
    :type data: pandas.Series or pandas.DataFrame
    :param parameters: A file containing the configurable parameters
    :type parameters: YAML file
    :rtype: pandas.Series, pandas.DataFrame, or None
    :raises: TypeError, ValueError
    """

    # Check that data is a  Dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The data has to be a DataFrame.")
    
    print('Loading the parameters from the yaml file' )
    print("-----------------------------")    
    schema = parameters["schema"]
    hierarchies = parameters["hierarchies"]
    thresholds = parameters["thresholds"]

    (
        keys,
        valid_column,
        del_columns,
        miss_num_columns,
        miss_categ_columns,
    ) = create_lists(data, schema, thresholds["Miss_threshold"])
    assert len(keys) != 0, "List is empty : we should have at least one key."
    nRow, nCol = data.shape  # shape of your data
    
    
    
    
    print("Substitute the nan values using the intersection function")
    print("-----------------------------")
    for key in keys:
        df_i = data[hierarchies[key]]
        (
            keys1,
            valid_column,
            del_columns,
            miss_num_columns,
            miss_categ_columns,
        ) = create_lists(df_i, schema, thresholds["Miss_threshold"])
        for column in (miss_num_columns+miss_categ_columns):
            intersection(df_i, column, key)
            
    (
        keys,
        valid_column,
        del_columns,
        miss_num_columns2,
        miss_categ_columns2,
    ) = create_lists(data, schema, thresholds["Miss_threshold"])
    
    total = data.isna().sum().sort_values(ascending=False) / nRow
    
    d = {
        miss_column: total[miss_column]
        for miss_column in (miss_num_columns2 + miss_categ_columns2)
    }
    
    cols_categ=data.select_dtypes(include ='category')
    enc =GroupingRareValues(cols=list(cols_categ.columns),threshold=0.001).fit(data)
    data = enc.transform(data)

    if len(d) != 0 :
        
         
        print('Encoding categorical features' )
        print("-----------------------------") 

        encoding(data, schema)
        
        if len(del_columns) != 0 :
            print('Deleting column with high level of missing values :' , del_columns )
            print("-----------------------------")
            delete_columns(data, columns=del_columns, inplace=True)
            
        for column in d.keys():
            
            data, zz = impute_by_recommended(data, column)
            print('Substitute the nan values in {} column.' .format(column))
            print("-----------------------------")
            
    
    for column in data.columns:
        if schema[column][0] == "categorical":
            data[column] = data[column].astype("category")
            
    print('Processing of missing values completed' )
    print("-----------------------------") 
   
    print('Start the outliers detection process')
    print("-----------------------------") 
    
    
            
    cols_num=[
        s
        for s in data.columns
        if schema[s][0] == "numerical"]
    
    for i in list(cols_num):
        stat, p_val = normaltest(data[i])
        if check_p_val(p_val, alpha=0.05):    
            outlier_index=outlier_detect_mean_std(data,i)
            if outlier_index != 'we do not have outliers' :
                para=outlier_index[1]
                data = windsorization(data=data,col='longitude',para=para,strategy='both')
                
        else:
            
            outlier_index=outlier_detect_IQR(data,i)
            if outlier_index != 'we do not have outliers' :
                para=outlier_index[1]
                data = windsorization(data=data,col='longitude',para=para,strategy='both')

    
    print('Processing of outliers completed' )
    print("-----------------------------") 

    print('Start the feature selection process')
    print("-----------------------------") 
    
    for column in data.columns:
        if schema[column][0] == "categorical":
            data[column] = data[column].astype("category")

    selected_feature=set((random_bar(data,target))['cols'].values)
    selector = mRMR(score_func='FCD', k=10)
    selector.fit(data,target)
    selected_feature2=set(selector.transform(data,10))
     
    selector = mRMR(score_func='FCQ', k=10)
    selector.fit(data,target)
    selected_feature3=set(selector.transform(data,10))
     
    selected_feature=selected_feature.union(selected_feature2)
    selected_feature=list(selected_feature.union(selected_feature3))
            
   
    print('The feature selection process has been completed.')
    
    
    data = data.sort_index(ascending=True)    
            

    
    return data , list(selected_feature)
