#!/usr/bin/env python
# coding: utf-8

# In[68]:


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


# In[69]:


print(c)


# In[70]:


import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN


# In[71]:


df = pd.DataFrame(c.T, columns=["x", "y"])
print(df)


# In[84]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Counting the distance between the points 
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Counting datapoints neighbours 
def region_query(df, point_idx, eps):
    neighbors = []
    x1, y1 = df.iloc[point_idx]['x'], df.iloc[point_idx]['y']  
    for idx, point in df.iterrows():
        x2, y2 = point['x'], point['y']  
        if distance(x1, y1, x2, y2) <= eps:
            neighbors.append(idx)
    return neighbors

# Clusters expanditure 
def expand_cluster(df, point_idx, neighbors, cluster_id, eps, min_samples, labels):
    labels[point_idx] = cluster_id
    
    for neighbor in neighbors:
        if labels[neighbor] == -1:  
            labels[neighbor] = cluster_id
        
        if labels[neighbor] == 0:  
            labels[neighbor] = cluster_id
            current_neighbors = region_query(df, neighbor, eps)
            
            if len(current_neighbors) >= min_samples:
                expand_cluster(df, neighbor, current_neighbors, cluster_id, eps, min_samples, labels)

# Funkcja DBSCAN 
def dbscan_custom(df, eps, min_samples):
    labels = np.zeros(len(df))  # 0 = not counted yet 
    cluster_id = 0
    
    for point_idx in range(len(df)):
        if labels[point_idx] != 0:
            continue
        
        neighbors = region_query(df, point_idx, eps)
        
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Oznacz jako szum
        else:
            cluster_id += 1
            expand_cluster(df, point_idx, neighbors, cluster_id, eps, min_samples, labels)
    
    return labels

# Generowanie danych
df = pd.DataFrame(c.T, columns=["x", "y"])
# Wartości eps i min_samples
eps_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
min_samples = 8

# Tworzenie wykresów dla każdej wartości eps
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, eps in enumerate(eps_values):
    # Wykonanie DBSCAN z daną wartością eps
    labels = dbscan_custom(df[['x', 'y']], eps, min_samples)
    
    # Rysowanie wyników
    axes[i].scatter(df['x'], df['y'], c=labels, cmap='rainbow', alpha=0.6)
    axes[i].set_title(f'eps = {eps}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].set_xlim([-10, 10])
    axes[i].set_ylim([-10, 10])

# Wyświetlenie wykresów
plt.tight_layout()
plt.show()


# In[ ]:




