#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install apyori')


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[15]:


dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = []

for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])


# In[16]:


from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.0028, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)


# In[17]:


results = list(rules)
results


# In[30]:


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


# In[31]:


resultsinDataFrame.head(20)


# In[32]:


resultsinDataFrame.nlargest(n = 10, columns = 'Lift')


# In[ ]:




