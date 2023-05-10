#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import time

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline


# In[ ]:


def drop_columns(df):
    '''
    Drop columns from the dataframe

    Parameters
    ----------
    df : pandas dataframe
        The dataframe to drop columns from

        
    Returns
    -------
    df : pandas dataframe
        The dataframe with the columns dropped
    '''
    
    cols_to_drop = ['RFM_BUYER', 'ORDER_VALUE_RANGE','ID_CONDITION', 'ORDER_MARKETING_CHANNEL' , 
                   'Total_likes', 'Total_wishes', 'Total_MMAO_NB', 'Avg_commision_x' , 'Avg_commision_y', 
                   'NL_REACTIVITY_GROUP','NB_PURCHASED', 'RANK_BUYER', 'BUYER_SEGMENT', 'NB_SOLD','RANK_WITHIN_SEGMENT', 
                   'ID_SEGMENT', 'ID_RFM_BUYER','Frequnecy_like_12M', 'VOUCHER_REVENUE']
    
    #
    #'NB_products_liked'

    to_drop_maybe = ['NB_products_liked','NB_categories_liked', 'NB_products_commented', 'NB_categories_commented']

    df.drop(cols_to_drop, axis=1, inplace=True)

    return df


# In[ ]:


def date_time_converting(df):
    '''
    Convert the date columns to datetime and create new columns for the number of days 
    between the first order and the creation of the account

    Parameters
    ----------

    df : pandas dataframe
    The dataframe to convert the date columns to datetime and create new columns for the 
    number of days between the first order and the creation of the account

    Returns
    -------
    df : pandas dataframe
    The dataframe with the date columns converted to datetime and new columns for the

    '''
    # Convert the date columns to datetime
    df['DATE_CREATION'] = pd.to_datetime(df['DATE_CREATION'])
    df['DATE_FIRST_PURCHASE'] = pd.to_datetime(df['DATE_FIRST_PURCHASE'])
    df['DATE_LAST_LOGIN'] = pd.to_datetime(df['DATE_LAST_LOGIN'])
    df['DATE_LAST_PURCHASE'] = pd.to_datetime(df['DATE_LAST_PURCHASE'])

    # Create a new column for the number of days between the first order and the creation of the account
    df['days_bf_first_order'] = (df['DATE_FIRST_PURCHASE'] - df['DATE_CREATION']).dt.days

    # recency login compared to today
    df['days_since_last_login'] = (pd.to_datetime('2023/02/11') - df['DATE_LAST_LOGIN']).dt.days

    # recency order compared to today
    #df['days_since_last_order'] = (pd.to_datetime('2023/02/11') - df['DATE_LAST_PURCHASE']).dt.days

    # Drop the date columns
    df.drop(['DATE_CREATION', 'DATE_NEW_BUYER', 'LastLikeDate', 'LastCommentDate', 
             'DATE_FIRST_PURCHASE', 'DATE_LAST_LOGIN', 'DATE_LAST_PURCHASE'], axis=1, inplace=True)
    df.drop(['Total_nb_likes', 'Total_nb_wish'], axis=1, inplace=True)

    return df


# In[ ]:


def missing_values(df):
    # Dealing with the missing values

    # Replacing the missing values with 0 
    df['NB_products_liked'].fillna(0, inplace=True)
    df['NB_categories_liked'].fillna(0, inplace=True)
    #df['Frequnecy_like_12M'].fillna(0, inplace=True)
    df['NB_products_commented'].fillna(0, inplace=True)
    df['NB_categories_commented'].fillna(0, inplace=True)
    df['Frequnecy_comment_12M'].fillna(0, inplace=True)

    # Replacing the missing values with 999
    df['Recency_comment'].fillna(999, inplace=True)
    df['Recency_liked'].fillna(999, inplace=True)

    return df


# In[ ]:


import category_encoders as ce


def target_encoding(train_df, valid_df, cols_to_encode, target_col):
    """
    Apply target encoding to a set of columns and avoid data leakage.
    
    Args:
        train_df (pandas.DataFrame): The training data.
        valid_df (pandas.DataFrame): The validation/test data.
        cols_to_encode (list of str): The names of the columns to encode.
        target_col (str): The name of the target column.
    
    Returns:
        pandas.DataFrame: The training data with target-encoded columns,
                           and the validation/test data with corresponding target-encoded columns.
    """
    # Create a TargetEncoder object
    te = ce.TargetEncoder()
    
    # Fit the target encoder on the training data
    te.fit(train_df[cols_to_encode], target_col)
    
    # Transform the training and validation/test data separately
    train_encoded = te.transform(train_df[cols_to_encode])
    valid_encoded = te.transform(valid_df[cols_to_encode])
    
    # Replace the original columns with the encoded columns in the training and validation/test data
    train_df = train_df.drop(cols_to_encode, axis=1)
    train_df = pd.concat([train_df, train_encoded], axis=1)
    
    valid_df = valid_df.drop(cols_to_encode, axis=1)
    valid_df = pd.concat([valid_df, valid_encoded], axis=1)
    
    return train_df, valid_df

def encoding(df):
    ### ecoding the categorical variables 
    
    ##ordinal encoding
    # USER_SEGMENT
    user_seg_map = {'Hibernating' :0, 'Inactive 6-12M': 1, 'Dormant 6M': 2, 'About to Sleep': 3, 'At High Risk':4, 'At Risk' :5, 'Need Attention':6, 'New Customer': 7, 'Potential Engaged':8, 'Engaged':9, 'Highly Engaged':10}
    df['USER_SEGMENT'] = df['USER_SEGMENT'].map(user_seg_map)


    # if task == 'classification':
    #     class_te(df)
    # elif task == 'regression':
    #     regr_te(df)
    # else: 
    #     print('Target encoding was not choosed for this run')
    # #BUYER_SEGMENT 
    # buyer_seg_map = {'Low potential one timers':0,'High potential one timers' : 1,'Low quality repeaters':2, 'High potential repeaters':3, 'Top buyers - only':4, 'Top buyers - low value': 5, 'Top buyers - high value':6, 'Top buyers - VVIC':7}
    # # OR maybe map with levels 1 to 3 with one timers, repeaters and top buyers
    # df['BUYER_SEGMENT'] = df['BUYER_SEGMENT'].map(buyer_seg_map)
    #drop the columns 

    return df


# In[ ]:


def pre_processing(df, X_train, X_test, scaling:bool = False, scaler:str = None, outliers:bool = False, outliers_method:str = None):
    '''
    Preprocess the data to be used in the model

    Parameters
    ----------
    df : pandas dataframe
        The dataframe to preprocess

    X_train : pandas dataframe
        The train set

    X_test : pandas dataframe
        The test set

    Returns
    -------
    X_train : pandas dataframe
        The train set after preprocessing

    X_test : pandas dataframe
        The test set after preprocessing

    '''

    # Scaling the data
    if scaling:
        if scaler == 'MinMaxScaler':
            # MinMaxScaler
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        elif scaler == 'StandardScaler':
            # StandardScaler
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        else:
            print('Please choose a valid scaler')   
    
    return X_train, X_test

