#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("data.csv")


# In[3]:


df.head()


# In[6]:


df.info()


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


df.corr()


# In[12]:


sns.heatmap(df.corr(), annot = True, cmap='coolwarm')


# In[13]:


df.hist()
plt.show()


# In[14]:


n = df.nunique()

print("Number of unique values in each column:\n", n)


# In[21]:


df_noex = df[df['Exercise']=='No']


# In[38]:


df_noex.shape


# In[37]:


df_noex.head()


# In[40]:


print(df_noex[df_noex['Heart_Disease'] == 'No'].shape[0])
print(df_noex[df_noex['Heart_Disease'] == 'Yes'].shape[0])


# In[45]:


columns = df_noex.columns


# In[50]:


columns = df_noex.columns
for col in columns:
    print(col, 'No', df_noex[df_noex[f'{col}'] == 'No'].shape[0])
    print(col, 'Yes', df_noex[df_noex[f'{col}'] == 'Yes'].shape[0])


# In[54]:


df_noex2 = df_noex.groupby(['Age_Category','Heart_Disease']).size()
df_noex2 = df_noex2.unstack()
df_noex2.plot(kind='bar')


# In[56]:


columns_object = df.select_dtypes(include=['object']).columns.tolist()
print(columns_object)


# In[57]:


for col in columns_object:
    df2 = df.groupby([f'{col}','Heart_Disease']).size()
    df2 = df2.unstack()
    df2.plot(kind='bar')


# In[60]:


for col in columns_object:
    df2 = df.groupby([f'{col}','Heart_Disease'])
     


# In[ ]:




