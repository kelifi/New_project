import pandas as pd
from datetime import datetime
from datetime import time
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
import numpy as np
import sklearn.preprocessing

def download(path, delimiter=","):
    data = pd.read_csv(path, delimiter)
    return data

from data_exploration.explore import get_dtypes
def create_lists(data, schema, thresh=0.7):

    """Create a set of lists .

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with missing values (nans).

    dict_final : a dictionary specifies the type of each column

    thresh : the percentage of missigness in a column

    Returns
    -------
    4 lists
    valid_column: list of columns without missingness
    del_column: list of columns containing percentage of missigness > thresh (will be deleted)
    miss_num_columns: list of numerical columns containg missing values
    miss_categ_columns: list of categorical columns containg missing values
    keys: list of  key columns

    """
    keys = [s for s in data.columns if schema[s][1] == "key"]
    valid_column = [s for s in data.columns if data[s].isnull().sum().sum() == 0 and s not in keys]
    del_column = [s for s in data.columns if (data[s].isna().sum() / (data.shape[0])) > thresh]
    miss_num_columns = [
        s
        for s in data.columns
        if schema[s][0] == "numerical"
        and data[s].isnull().sum().sum() > 0
        and s not in del_column
    ]
    miss_categ_columns = [
        s
        for s in data.columns
        if schema[s][0] == "categorical"
        and data[s].isnull().sum().sum() > 0
        and s not in del_column
    ]

    return keys, valid_column, del_column, miss_num_columns, miss_categ_columns


def merge_data(df1, df2, ch):

    """
    Parameters
    ----------
    df1 : pandas.DataFrame

    df2 : pandas.DataFrame


    ch : key of the Merge


    """

    assert ch in df1 and ch in df2, "Column name does not exist within one of the file."

    df3 = pd.merge(df1, df2, on=ch)

    return df3


def add_time(date1, date2):

    current_time1 = date1.strftime("%H:%M:%S")
    current_time2 = date2.strftime("%H:%M:%S")
    t1 = datetime.strptime(current_time1, "%H:%M:%S")
    t2 = datetime.strptime(current_time2, "%H:%M:%S")
    time_zero = datetime.strptime("00:00:00", "%H:%M:%S")

    return (t1 - time_zero + t2).time()


def get_time(date1, date2):

    c = date2 - date1
    seconds = c.total_seconds()
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    my_time = time(int(hour), int(minutes), int(seconds))

    return my_time


def encoding(df, schema):

    #time_start = datetime.now()
    str_var_list, num_var_list, all_var_list=get_dtypes(df)            
    df_temp = df[str_var_list].astype("object").apply(LabelEncoder().fit_transform)
    df[str_var_list] = df_temp.where(~df.isna(), df)        

    for column in df.columns:
        if schema[column][0] == "categorical":
            df[column] = df[column].astype("category")

    #get_time(time_start, datetime.now())

    #final_time = datetime.now()
    #Time = get_time(time_start, final_time)

    #return df
    
    
def check_missing(data,output_path=None):
    """
    check the total number & percentage of missing values
    per variable of a pandas Dataframe
    """
    
    result = pd.concat([data.isnull().sum(),data.isnull().mean()],axis=1)
    result = result.rename(index=str,columns={0:'total missing',1:'proportion'})
    if output_path is not None:
        result.to_csv(output_path+'missing.csv')
        print('result saved at', output_path, 'missing.csv')
    return result    



def data_discretization(X, n_bins):
    """
    This function implements the data discretization function to discrete data into n_bins
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    n_bins: {int}
        number of bins to be discretized
    Output
    ------
    X_discretized: {numpy array}, shape (n_samples, n_features)
        output discretized data, where features are digitized to n_bins
    """

    # normalize each feature
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_normalized = min_max_scaler.fit_transform(X)

    # discretize X
    n_samples, n_features = X.shape
    X_discretized = np.zeros((n_samples, n_features))
    bins = np.linspace(0, 1, n_bins)
    for i in range(n_features):
        X_discretized[:, i] = np.digitize(X_normalized[:, i], bins)

    return X_discretized