#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[54]:


dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

X = pd.DataFrame(x)
Y = pd.DataFrame(y)

X


# In[55]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])
X = pd.DataFrame(x)
X


# In[57]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[1])], remainder="passthrough")
x = np.array(ct.fit_transform(x))
X = pd.DataFrame(x)
X


# In[58]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[60]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[62]:


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))


# In[64]:


ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
ann.fit(x_train, y_train, batch_size=32, epochs=100)


# In[74]:


print(ann.predict(sc.transform([[1,0,0,1,600,1,40,3,60000,2,1,1,50000]])))


# In[ ]:




