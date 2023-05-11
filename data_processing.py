#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports 
import pandas as pd
import numpy as np
import os


# # Data Preprocessing

# In this document we will proceed to the import of the important of the raw datasets extracted from Vestiaire Collective's datalake. We will proceed data wrangling, data cleaning and data transformation to obtain a clean dataset ready to be used for the analysis. The final dataframe will be called "df_model" as it will be then used for the modelisation part.

# ## MMAO & TRANSACTION DATASETS

# In[3]:


def data_processing_mmao_transac(df_mmao, df_transaction):
    # Drop the columns with more than 15% null values
    df = df_transaction.dropna(thresh=0.85*len(df_transaction), axis=1)

    # Keeping only relevant features for our project
    df_tri = df[['ID_PRODUCT',
    'ID_ORDER',
    'ID_CATEGORY',
    'ID_SUBCATEGORY',
    'ID_CONDITION',
    'ID_BRAND',
    'UNIVERSE',
    'DEPOSIT_PRICE',
    'PRICE_SOLD_GMV',
    'NB_ITEMS',
    'DISCOUNT_AMOUNT_GMV',
    'DATE_SOLD',
    'DATE_PUBLISHED',
    'ID_PAYMENT_TYPE',
    'ID_BUYER',
    'RANK_BUYER',
    'RANK_WITHIN_SEGMENT',
    'ID_SEGMENT',
    'LOVED',
    'ORDER_VALUE_RANGE',
    'BUYER_FEE_GMV',
    'ORDER_MARKETING_CHANNEL',
    'MMAO_PRICE_DROP',
    'VOUCHER_REVENUE',
    'BUYER_TYPE',
    'FLAG_FRAUD',
    'ID_GENDER',
    'DATE_NEW_BUYER',
    'DATE_CREATION',
    'DATE_LAST_LOGIN',
    'INACTIVE',
    'NB_SOLD',
    'VALUE_SOLD',
    'NB_PUBLISHED',
    'VALUE_PUBLISHED',
    'NB_PURCHASED',
    'VALUE_PURCHASED',
    'NL_REACTIVITY_GROUP',
    'ID_RFM_BUYER',
    'RFM_BUYER',
    'USER_SEGMENT',
    'DATE_LAST_PURCHASE',
    'DATE_FIRST_PURCHASE',
    'BUYER_SEGMENT']]

    # Creation of the two target variables 
    ##1. Is the buyer a repeater?
    ### Convert buyer types to 0 and 1
    df_tri['REPEATER'] = df_tri['BUYER_TYPE'].map({'New_Buyer': 0, 'Repeat': 1, 'Repeat_90D': 1})

    ## 2. CLTV
    ### Add the value purchased by the buyer and the value sold by the buyer
    df_tri['CLTV'] = df_tri['VALUE_PURCHASED'] + df_tri['VALUE_SOLD']

    # Drop the columns that are not needed anymore
    df_transac_final = df_tri.drop(['BUYER_TYPE', 'VALUE_PURCHASED', 'VALUE_SOLD', 'VALUE_PUBLISHED'], axis=1)

    # Create a feature with the number of offers grouped by each buyer
    df_mmao['NB_OFFERS'] = df_mmao.groupby('ID_BUYER')['NB_TOTAL_OFFERS'].transform('sum')
    df_mmao['AVG_OFFERS'] = df_mmao.groupby('ID_BUYER')['NB_TOTAL_OFFERS'].transform('mean')

    # Drop the duplicates in mmao dataframe
    df_mmao_final = df_mmao[['ID_BUYER', 'NB_OFFERS', 'AVG_OFFERS']].drop_duplicates()

    # Merge the two dataframes

    df_tr_mmao = pd.merge(df_transac_final, df_mmao_final, on='ID_BUYER', how='left')

    # Fill null values due to the merge and drop the remaining values 

    df_tr_mmao['NB_OFFERS'].fillna(0, inplace=True)
    df_tr_mmao['AVG_OFFERS'].fillna(0, inplace=True)

    df_tr_mmao.dropna(inplace=True)

    # Drop unrelevant features for the project 
    df_tr_mmao.drop(['ID_PRODUCT','ID_ORDER','UNIVERSE','LOVED','FLAG_FRAUD'], axis=1, inplace=True)

    # Create feature number of days between date sold and date published, then drop the two columns

    df_tr_mmao['DATE_SOLD'] = pd.to_datetime(df_tr_mmao['DATE_SOLD'])
    df_tr_mmao['DATE_PUBLISHED'] = pd.to_datetime(df_tr_mmao['DATE_PUBLISHED'])
    df_tr_mmao['NB_DAYS_ONLINE'] = (df_tr_mmao['DATE_SOLD'] - df_tr_mmao['DATE_PUBLISHED']).dt.days
    df_tr_mmao.drop(['DATE_SOLD','DATE_PUBLISHED'], axis=1, inplace=True)

    df_tr_mmao_test = df_tr_mmao.copy()

    # Create aggregated features as we are grouping on ID_BUYER

    ## Categorical or data that will be grouped by mode value
    grouped = df_tr_mmao_test.groupby('ID_BUYER')
    agg_mode = grouped.agg({'ID_CATEGORY': pd.Series.mode, 'ID_SUBCATEGORY': pd.Series.mode, 'ID_BRAND': pd.Series.mode,'ID_PAYMENT_TYPE': pd.Series.mode,'ORDER_MARKETING_CHANNEL': pd.Series.mode})
    agg_mode= agg_mode.reset_index()
    df_tr_mmao_test = df_tr_mmao_test.merge(agg_mode, on='ID_BUYER', how='left')
    df_tr_mmao_test = df_tr_mmao_test.drop(['ID_CATEGORY_x', 'ID_SUBCATEGORY_x', 'ID_BRAND_x', 'ID_PAYMENT_TYPE_x', 'ORDER_MARKETING_CHANNEL_x'], axis=1)
    df_tr_mmao_test = df_tr_mmao_test.rename(columns={'ID_CATEGORY_y': 'ID_CATEGORY','ID_SUBCATEGORY_y': 'ID_SUBCATEGORY','ID_BRAND_y': 'ID_BRAND','ID_PAYMENT_TYPE_y': 'ID_PAYMENT_TYPE', 'ORDER_MARKETING_CHANNEL_y': 'ORDER_MARKETING_CHANNEL'})

    ## Numerical or data that will be grouped by mean value
    agg_mean = df_tr_mmao_test.groupby('ID_BUYER').agg({'DEPOSIT_PRICE': 'mean', 'PRICE_SOLD_GMV': 'mean', 'NB_ITEMS': 'mean','DISCOUNT_AMOUNT_GMV': 'mean','BUYER_FEE_GMV': 'mean','MMAO_PRICE_DROP': 'mean', 'VOUCHER_REVENUE': 'mean', 'ID_CONDITION': 'mean', 'ID_GENDER': 'mean'})
    agg_mean = agg_mean.reset_index()
    df_tr_mmao_test = df_tr_mmao_test.merge(agg_mean, on='ID_BUYER', how='left')
    df_tr_mmao_test = df_tr_mmao_test.drop(['DEPOSIT_PRICE_x', 'PRICE_SOLD_GMV_x', 'NB_ITEMS_x','DISCOUNT_AMOUNT_GMV_x','BUYER_FEE_GMV_x','MMAO_PRICE_DROP_x', 'VOUCHER_REVENUE_x','ID_CONDITION_x', 'ID_GENDER_x'], axis=1)
    df_tr_mmao_test = df_tr_mmao_test.rename(columns={'DEPOSIT_PRICE_y': 'DEPOSIT_PRICE','PRICE_SOLD_GMV_y': 'PRICE_SOLD_GMV','NB_ITEMS_y': 'NB_ITEMS','DISCOUNT_AMOUNT_GMV_y': 'DISCOUNT_AMOUNT_GMV','BUYER_FEE_GMV_y': 'BUYER_FEE_GMV','MMAO_PRICE_DROP_y': 'MMAO_PRICE_DROP', 'VOUCHER_REVENUE_y': 'VOUCHER_REVENUE','ID_CONDITION_y': 'ID_CONDITION', 'ID_GENDER_y':'ID_GENDER' })

    # Remap the gender values 
    df_tr_mmao_test.loc[df_tr_mmao_test['ID_GENDER'] == 0, 'ID_GENDER'] = 1
    df_tr_mmao_test.loc[df_tr_mmao_test['ID_GENDER'] == 1, 'ID_GENDER'] = 0
    df_tr_mmao_test.loc[df_tr_mmao_test['ID_GENDER'] == 2, 'ID_GENDER'] = 1
    df_tr_mmao_test.loc[df_tr_mmao_test['ID_GENDER'] == 3, 'ID_GENDER'] = 2

    df_final_v1 = df_tr_mmao_test.copy()

    # groupby ID_BUYER and if in the column REPEATER there is a 1, change all other values to 1 as well

    df_final_v1['REPEATER'] = df_final_v1.groupby('ID_BUYER')['REPEATER'].transform('max')

    # create a dataframe with unique ID_BUYERS by grouping by ID_BUYER 
    df_final_v1 = df_final_v1.groupby('ID_BUYER').first().reset_index()

    return df_final_v1


# ## LIKES

# In[5]:


def data_processing_likes(df_likes):

    df = df_likes.dropna(thresh=0.85*len(df_likes), axis=1)

    df2 = df[["ID_CLIENT",
            "ID_BUYER",
            "ID_LIKE",
            "DATE_LIKED",
            "DATE_PUBLISHED",
            "UPDATEDAT",
            "NBWISH",
            "NB_CONSULTATION",
            "MMAO_NB",
            "LIKES",
            "WISHES",
            "LOVED",
            "IS_ONLINE",
            "IS_ACTIVE",
            "IS_WITHDRAWN",
            "IS_REJECTED",
            "IS_RESERVED",
            "STATUS_AFTER_7_DAYS",
            "CURRENT_STATUS",
            "ID_PRODUCT",
            "ID_CATEGORY",
            "ID_SUBCATEGORY",
            "ID_BRAND",
            "ID_UNIVERSE",
            "ID_CONDITION",
            "SEGMENT",
            "ONED_SOLD_STATUS",
            "TWOD_SOLD_STATUS",
            "THREED_SOLD_STATUS",
            "SEVEND_SOLD_STATUS",
            "FIFTEEND_SOLD_STATUS",
            "THIRTYD_SOLD_STATUS",
            "NINETYD_SOLD_STATUS",
            "ID_LAST_ACTION",
            "IS_ITEM_WHITELISTED",
            "IS_NEWIN_LIKED_IN_7DAYS",
            "IS_NEWIN_WISHLISTED_IN_7DAYS",
            "IS_NEWIN_MMAO_IN_7DAYS",
            "IS_NEWIN_COMMENTED_IN_7DAYS",
            "IS_NEWIN_ATC_IN_7DAYS",
            "COMMISSION"
    ]]

    df2 = df2.drop(['IS_ONLINE', 'IS_ACTIVE', 'IS_REJECTED','IS_RESERVED', 'IS_ITEM_WHITELISTED', 'ID_LAST_ACTION', 'NB_CONSULTATION',
                    'ONED_SOLD_STATUS', 'TWOD_SOLD_STATUS', 'THREED_SOLD_STATUS', 
                    'SEVEND_SOLD_STATUS', 'FIFTEEND_SOLD_STATUS', 'THIRTYD_SOLD_STATUS',
                    'NINETYD_SOLD_STATUS', 'IS_WITHDRAWN', 'IS_NEWIN_LIKED_IN_7DAYS',
                    'IS_NEWIN_MMAO_IN_7DAYS', 'IS_NEWIN_WISHLISTED_IN_7DAYS', 'IS_NEWIN_COMMENTED_IN_7DAYS',
                    'IS_NEWIN_ATC_IN_7DAYS', 'ID_UNIVERSE', 'LOVED', 'SEGMENT', 
                    'NBWISH', 'DATE_PUBLISHED', 'UPDATEDAT', 
                    'ID_BRAND', 'ID_CONDITION', 'ID_SUBCATEGORY', 'STATUS_AFTER_7_DAYS','CURRENT_STATUS','ID_CLIENT'], axis=1)

    df2_test = df2.copy()

    # Create features with the total number of likes, wishes and consultations grouped by each buyer
    df2_test['Total_likes'] = df2_test.groupby('ID_BUYER')['LIKES'].transform('sum')
    df2_test['Total_wishes'] = df2_test.groupby('ID_BUYER')['WISHES'].transform('sum')
    df2_test['Total_MMAO_NB'] = df2_test.groupby('ID_BUYER')['MMAO_NB'].transform('sum')

    # Create features with the unique number of commented products and categories grouped by each buyer
    df2_test['NB_products_liked'] = df2_test.groupby('ID_BUYER')['ID_PRODUCT'].transform('nunique')
    df2_test['NB_categories_liked'] = df2_test.groupby('ID_BUYER')['ID_CATEGORY'].transform('nunique')

    # Create a feature with average commission grouped by each buyer
    df2_test['Avg_commision'] = df2_test.groupby('ID_BUYER')['COMMISSION'].transform('mean')

    df2_test['DATE_LIKED'] = pd.to_datetime(df2_test['DATE_LIKED'])

    #Feature for the last comment made grouped by ID_BUYER
    #recency is calculated based on difference of days from last comment ever made on platform
    df_recency = df2_test.groupby(by='ID_BUYER',
                            as_index=False)['DATE_LIKED'].max()
    df_recency.columns = ['ID_BUYER', 'LastLikeDate']
    df_recency['LastLikeDate'] = pd.to_datetime(df_recency['LastLikeDate'])
    recent_date = df_recency['LastLikeDate'].max()
    df_recency['Recency_liked'] = df_recency['LastLikeDate'].apply(
        lambda x: (recent_date - x).days)
    df_recency.head()


    filtered_df = df2_test[~(df2_test['DATE_LIKED'] < '2022-01-01')]

    frequency_df = filtered_df.groupby(
        by=['ID_BUYER'], as_index=False)['DATE_LIKED'].count()
    frequency_df.columns = ['ID_BUYER', 'Frequnecy_like_12M']
    frequency_df.head()

    df2_test2 = df2_test[['ID_BUYER', 'Total_likes', 'Total_wishes', 'Total_MMAO_NB', 'NB_products_liked', 'NB_categories_liked', 'Avg_commision']].drop_duplicates()

    df2_with_recency = df2_test2.merge(df_recency, on='ID_BUYER')

    df2_with_recency_and_frquency = df2_with_recency.merge(frequency_df, on='ID_BUYER',how='left')

    df2_with_recency_and_frquency['Frequnecy_like_12M'].fillna(0, inplace=True)

    return df2_with_recency_and_frquency


# ## COMMENTS 

# In[7]:


def data_preprocessing_comment(df_comment):
    # Drop the columns with more than 20% null values
    df = df_comment.dropna(thresh=0.80*len(df_comment), axis=1)


    df2 = df[["ID_CLIENT",
            "ID_BUYER",
            "ID_COMMENT",
            "DATE_COMMENT",
            "DATE_PUBLISHED",
            "UPDATEDAT",
            "NB_LIKES",
            "NBWISH",
            "NB_CONSULTATION",
            "LIKES",
            "WISHES",
            "LOVED",
            "IS_ONLINE",
            "IS_ACTIVE",
            "IS_WITHDRAWN",
            "IS_REJECTED",
            "IS_RESERVED",
            "STATUS_AFTER_7_DAYS",
            "CURRENT_STATUS",
            "ACCEPTED_BY",
            "CURATOR_TYPE",
            "ID_PRODUCT",
            "ID_CATEGORY",
            "ID_SUBCATEGORY",
            "ID_BRAND",
            "UNIVERSE",
            "ID_CONDITION",
            "SEGMENT",
            "ONED_SOLD_STATUS",
            "TWOD_SOLD_STATUS",
            "THREED_SOLD_STATUS",
            "SEVEND_SOLD_STATUS",
            "FIFTEEND_SOLD_STATUS",
            "THIRTYD_SOLD_STATUS",
            "NINETYD_SOLD_STATUS",
            "ID_LAST_ACTION",
            "IS_ITEM_WHITELISTED",
            "IS_NEWIN_LIKED_IN_7DAYS",
            "IS_NEWIN_WISHLISTED_IN_7DAYS",
            "IS_NEWIN_MMAO_IN_7DAYS",
            "IS_NEWIN_COMMENTED_IN_7DAYS",
            "IS_NEWIN_ATC_IN_7DAYS",
            "COMMISSION"
    ]]

    df2 = df2.drop(['CURATOR_TYPE', 'IS_ONLINE', 'IS_ACTIVE', 'IS_REJECTED','IS_RESERVED', 'IS_ITEM_WHITELISTED','ACCEPTED_BY', 'ID_LAST_ACTION', 'NB_CONSULTATION'], axis=1)
    df2 = df2.drop(['ONED_SOLD_STATUS', 'TWOD_SOLD_STATUS', 'THREED_SOLD_STATUS', 
                'SEVEND_SOLD_STATUS', 'FIFTEEND_SOLD_STATUS', 'THIRTYD_SOLD_STATUS',
                'NINETYD_SOLD_STATUS', 'IS_WITHDRAWN', 'IS_NEWIN_LIKED_IN_7DAYS',
                'IS_NEWIN_MMAO_IN_7DAYS', 'IS_NEWIN_WISHLISTED_IN_7DAYS', 'IS_NEWIN_COMMENTED_IN_7DAYS',
                'IS_NEWIN_ATC_IN_7DAYS', 'UNIVERSE', 'LOVED', 'SEGMENT', 
                'LIKES', 'WISHES', 'DATE_PUBLISHED', 'UPDATEDAT', 
                'ID_BRAND', 'ID_CONDITION', 'ID_SUBCATEGORY', 'STATUS_AFTER_7_DAYS','CURRENT_STATUS'], axis=1)
    
    df2 = df2.drop(['ID_CLIENT'], axis=1)


    df2_test = df2.copy()

    # Create features with the total number of likes, wishes and consultations grouped by each buyer

    df2_test['Total_nb_likes'] = df2_test.groupby('ID_BUYER')['NB_LIKES'].transform('sum')
    df2_test['Total_nb_wish'] = df2_test.groupby('ID_BUYER')['NBWISH'].transform('sum')

    # Create features with the unique number of commented products and categories grouped by each buyer
    df2_test['NB_products_commented'] = df2_test.groupby('ID_BUYER')['ID_PRODUCT'].transform('nunique')
    df2_test['NB_categories_commented'] = df2_test.groupby('ID_BUYER')['ID_CATEGORY'].transform('nunique')



    # Create a feature with average commission grouped by each buyer
    df2_test['Avg_commision'] = df2_test.groupby('ID_BUYER')['COMMISSION'].transform('mean')

    df2_test['DATE_COMMENT'] = pd.to_datetime(df2_test['DATE_COMMENT'])

    #Feature for the last comment made grouped by ID_BUYER
    #recency is calculated based on difference of days from last comment ever made on platform
    df_recency = df2_test.groupby(by='ID_BUYER',
                            as_index=False)['DATE_COMMENT'].max()
    df_recency.columns = ['ID_BUYER', 'LastCommentDate']
    df_recency['LastCommentDate'] = pd.to_datetime(df_recency['LastCommentDate'])
    recent_date = df_recency['LastCommentDate'].max()
    df_recency['Recency_comment'] = df_recency['LastCommentDate'].apply(
        lambda x: (recent_date - x).days)

    filtered_df = df2_test[~(df2_test['DATE_COMMENT'] < '2022-01-01')]


    frequency_df = filtered_df.groupby(
        by=['ID_BUYER'], as_index=False)['DATE_COMMENT'].count()
    frequency_df.columns = ['ID_BUYER', 'Frequnecy_comment_12M']

    df2_test2 = df2_test[['ID_BUYER', 'Total_nb_likes', 'Total_nb_wish', 'NB_products_commented', 'NB_categories_commented', 'Avg_commision']].drop_duplicates()

    df2_with_recency = df2_test2.merge(df_recency, on='ID_BUYER')

    df2_with_recency_and_frquency = df2_with_recency.merge(frequency_df, on='ID_BUYER',how='left')

    df2_with_recency_and_frquency['Frequnecy_comment_12M'].fillna(0, inplace=True)

    df2_with_recency_and_frquency.to_csv('df_comment_final.csv', index=False)

    return df2_with_recency_and_frquency


# ## Merging Dataframes

# In[8]:


# merging the three dataframes for the final model

def merge_dataframes(df_transac, df_likes_final, df_comments_final):
    # Merge the dataframes
    df_merge = pd.merge(df_transac, df_likes_final, on='ID_BUYER', how='left')
    df_merge = pd.merge(df_merge, df_comments_final, on='ID_BUYER', how='left')

    # Export as csv

    return df_merge

