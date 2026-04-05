#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import joblib as jb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


# In[11]:


df = pd.read_csv("bank_transactions.csv")


# In[12]:


df.head()


# In[13]:


df.dtypes


# In[14]:


df.isnull().sum()


# In[31]:


df['CustomerDOB'].fillna(df['CustomerDOB'].mode()[0],inplace=True)


# In[32]:


df['CustGender'].fillna(df['CustGender'].mode()[0],inplace=True)


# In[33]:


df['CustLocation'].fillna(df['CustLocation'].mode()[0],inplace=True)


# In[34]:


df['CustAccountBalance'].fillna(df['CustAccountBalance'].mean())


# In[19]:


df.isnull().sum()


# In[20]:


df.dtypes


# In[21]:


x = df.drop(['TransactionID','CustomerID','CustomerDOB','TransactionDate','TransactionTime'],axis=1)


# In[22]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[23]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[24]:


numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


# In[25]:


categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


# In[26]:


preprocessor = ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols),
    ('cat',categorical_transformer,categorical_cols)
])


# In[27]:


model = Pipeline(steps=[
    ('pre',preprocessor),('reg',KMeans(n_clusters=3,random_state=42))
])


# In[35]:


model.fit(x)


# In[29]:


cluster = model.predict(x)
print(f'{cluster}')


# In[30]:


jb.dump(model,'KMeans.pkl')


# In[49]:


import streamlit as st
import pandas as pd
import joblib as jb

load = jb.load('KMeans.pkl')

st.title("Customer Segmentation prediction")

CustGender = st.selectbox("Gender",["Male","Female","Other"])
CustLocation = st.text_input("CustLocation")
CustAccountBalance = st.number_input("CustAccountBalance")
transaction_amount = st.number_input('TransactionAmount (INR)')

if st.button("Predict"):

    data = {
        "CustGender":[CustGender],
        "CustLocation":[CustLocation],
        "CustAccountBalance":[CustAccountBalance],
        "TransactionAmount (INR)":[transaction_amount]
    }

    df = pd.DataFrame(data)

    prediction = load.predict(df)
    
    if prediction[0] == 0:
        st.success("High Spender 💰")
    else :
     st.success("Low Spender 😐")

