#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[14]:


dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values


# In[15]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = "ward"))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidian Distances")
plt.show()


# In[22]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(x)


# In[23]:


plt.scatter(x[y_hc==0, 0], x[y_hc==0, 1], s=100, c="red", label="Cluster 1")
plt.scatter(x[y_hc==1, 0], x[y_hc==1, 1], s=100, c="blue", label="Cluster 2")
plt.scatter(x[y_hc==2, 0], x[y_hc==2, 1], s=100, c="green", label="Cluster 3")
#plt.scatter(x[y_hc==3, 0], x[y_hc==3, 1], s=100, c="purple", label="Cluster 4")
#plt.scatter(x[y_hc==4, 0], x[y_hc==4, 1], s=100, c="pink", label="Cluster 5")
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


# In[ ]:




