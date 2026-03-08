## This file contains code to load and clean a .db file

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



# load data 
def connect_db(link,db_name):
    db = sqlite3.connect(link)  # link should be a string path of database file
    query = f'SELECT * FROM {db_name};'    # use SQL commands get whole dataset
    df = pd.read_sql_query(query, db)
    return df.dropna()  # remove entries with missing values

# function to view data
def view_data(df):
    print(df)


# function to clean data as per EDA
# one additional data preprocessing step is to replace CustomerTypes with number, so that all data types are numerical and can be used in machine learning models
def clean_data(df):
    df['CustomerType']= df['CustomerType'].replace({'': 'Unknown', 'nan':'Unknown', 'None':'Unknown', 'returning_Visitor': 'Returning_Visitor'})
    df['CustomerType']= df['CustomerType'].replace({'New_Visitor': 1, 'Returning_Visitor': 2, 'Unknown':3, 'Other': 4})
    df['CustomerType'] = df['CustomerType'].astype(float )
    df[['ProductPageTime', 'BounceRate']] = df[['ProductPageTime', 'BounceRate']].abs()
    df = df.drop(columns=["PurchaseCompleted", 'BounceRate'])
    return df


# function to only select the relevant entries in CustomerType
# this function is for future proofing, in case there are more inconsistent entries. may result in less data.
def prune_customer(df):
    valid = ['Unknown', 'New_Visitor', 'Returning_Visitor','Other']
    df = df[df['CustomerType'].isin(valid)]
    return df

# function to perform data transformations
def transform_data(df):
    scale = StandardScaler()

    # log transform and then scale ProductPageTime
    df['Time_scaled'] = np.log1p(df['ProductPageTime'])
    df['Time_scaled'] = scale.fit_transform(df[['Time_scaled']])



test = connect_db('data/online_shopping.db','online_shopping')
test = clean_data(test)
print(test.dtypes)