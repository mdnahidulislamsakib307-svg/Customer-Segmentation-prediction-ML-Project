#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import joblib as jb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


# In[10]:


df = pd.read_csv("bank_transactions.csv")


# In[12]:


df.head()


# In[13]:


df.dtypes


# In[14]:


df.isnull().sum()


# In[16]:


df['CustomerDOB'].fillna(df['CustomerDOB'].mode()[0],inplace=True)


# In[18]:


df['CustGender'].fillna(df['CustGender'].mode()[0],inplace=True)


# In[20]:


df['CustLocation'].fillna(df['CustLocation'].mode()[0],inplace=True)


# In[34]:


df['CustAccountBalance'].fillna(df['CustAccountBalance'].mean())


# In[38]:


df.isnull().sum()


# In[39]:


df.dtypes


# In[40]:


x = df.drop(['TransactionID','CustomerID','CustomerDOB','TransactionDate','TransactionTime'],axis=1)


# In[42]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[43]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[45]:


numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


# In[46]:


categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


# In[47]:


preprocessor = ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols),
    ('cat',categorical_transformer,categorical_cols)
])


# In[52]:


model = Pipeline(steps=[
    ('pre',preprocessor),('reg',KMeans(n_clusters=3,random_state=42))
])


# In[57]:


model.fit(x)


# In[59]:


cluster = model.predict(x)
print(f'{cluster}')


# In[65]:


jb.dump(model,'KMeans.pkl')

