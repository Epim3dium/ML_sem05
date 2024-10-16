#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib.pyplot as plt 
import numpy as np
import random
mean= [3, 3]
cov = [[1, 0], [0, 1]]
a = np.random.multivariate_normal(mean, cov, 500).T
mean = [-3, -3]
cov = [[2, 0], [0, 5]]
b = np.random.multivariate_normal(mean, cov, 500).T
c = np.concatenate((a, b) , axis = 1) 
c=c.T
np.random.shuffle (c)
c=c.T
x = c[0] 
y=c[1]
plt.plot(x, y, 'x') 
plt.axis('equal') 
plt.show()


# In[27]:


print(c)


# In[28]:


import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN


# In[29]:


df = pd.DataFrame(c.T, columns=["x", "y"])
print(df)


# In[48]:


# DBSCAN algorithm implementation
dbscan = DBSCAN(eps=0.8, min_samples=5)  
df['cluster'] = dbscan.fit_predict(df[['x', 'y']])

# Cluster centers calculation
centers = df.groupby('cluster')[['x', 'y']].mean()
print(centers)


# Clusters visualization
plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='rainbow', alpha=0.5, label='Data points')
plt.scatter(centers['x'], centers['y'], c='black', marker='x', s=100, label='Clusters centers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[35]:


# Core points calculation
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True


# In[37]:


# Distinguishing the core points 
df['is_core_point'] = core_samples_mask

# Clusters visualization with core points
plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='rainbow', alpha=0.5, label='Data points')
plt.scatter(df[df['is_core_point']]['x'], df[df['is_core_point']]['y'], c='black', marker='o', label='Core points', edgecolor='white')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[51]:


# Defining different eps values for which we want to see the clustering
eps_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# Creating plots for each eps value
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()  # Flattening the grid of plots into one dimension

for i, eps in enumerate(eps_values):
    # DBSCAN for the given eps value
    dbscan = DBSCAN(eps=eps, min_samples=8)
    df['cluster'] = dbscan.fit_predict(df[['x', 'y']])
    
    # Plotting on the appropriate axis
    axes[i].scatter(df['x'], df['y'], c=df['cluster'], cmap='rainbow', alpha=0.6)
    axes[i].set_title(f'eps = {eps}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].set_xlim([-10, 10])
    axes[i].set_ylim([-10, 10])

plt.tight_layout()
plt.show()


# In[ ]:




