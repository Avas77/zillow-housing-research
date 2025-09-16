#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time, psutil, tracemalloc


# In[2]:


t0 = time.perf_counter()
rent_df = pd.read_csv("dataset/raw/zori-rental-all-homes.csv")
mortgage_df = pd.read_csv("dataset/raw/zhvi-total-monthly-payment.csv")
t1 = time.perf_counter()

print("load time", t1 - t0)
print("Shape of rental dataframe:", rent_df.shape)
print("Shape of Mortgage dataframe:", mortgage_df.shape)
print("mem usage (MB)(Rent):", rent_df.memory_usage(deep=True).sum() / 1024**2)
print("mem usage (MB)(Mortgage):", mortgage_df.memory_usage(deep=True).sum() / 1024**2)


# In[3]:


rent_df["RegionName"] = rent_df["RegionName"].astype('category')
rent_df["StateName"] = rent_df["StateName"].astype('category')

mortgage_df["RegionName"] = mortgage_df["RegionName"].astype('category')
mortgage_df["StateName"] = mortgage_df["StateName"].astype('category')


# In[4]:


rent_df = rent_df[rent_df["RegionType"] == "msa"]
mortgage_df = mortgage_df[mortgage_df["RegionType"] == "msa"]


# In[5]:


rent_df = rent_df.drop(columns=["RegionType"])
mortgage_df = mortgage_df.drop(columns=["RegionType"])


# In[6]:


def melt_df(df, value_name):
    id_vars = ['RegionID', 'RegionName', 'StateName', 'SizeRank'] 
    df_melted = df.melt(id_vars=id_vars, var_name="Date", value_name=value_name)
    df_melted["Date"] = pd.to_datetime(df_melted["Date"]) 
    return df_melted    


# In[7]:


t0 = time.perf_counter()
rent_long_df = melt_df(rent_df, "Rent_per_month")
mortgage_long_df = melt_df(mortgage_df, "Mortgage_per_month")
t1 = time.perf_counter()

print("melt time:", t1 - t0)


# In[8]:


t0 = time.perf_counter()
rent_parquet = pd.read_parquet("rent_data.parquet", engine="fastparquet")
mortgage_parquet = pd.read_parquet("mortgage_data.parquet", engine="fastparquet")
t1 = time.perf_counter()

print("load time", t1 - t0)
print("Shape of rental dataframe:", rent_parquet.shape)
print("Shape of Mortgage dataframe:", mortgage_parquet.shape)
print("mem usage (MB)(Rent):", rent_parquet.memory_usage(deep=True).sum() / 1024**2)
print("mem usage (MB)(Rent):", mortgage_parquet.memory_usage(deep=True).sum() / 1024**2)


# In[9]:


rent_parquet


# In[ ]:




