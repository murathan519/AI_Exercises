#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[7]:


dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = []

for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])


# In[8]:


from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.0028, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)


# In[9]:


results = list(rules)
results


# In[10]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])


# In[11]:


resultsinDataFrame


# In[13]:


resultsinDataFrame.nlargest(n = 10, columns = 'Support')


# In[ ]:




