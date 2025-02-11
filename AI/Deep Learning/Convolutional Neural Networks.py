#!/usr/bin/env python
# coding: utf-8

# In[2]:


import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('.')


# In[4]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# In[5]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# In[6]:


from tensorflow.keras.layers import Input, Conv2D

cnn = tf.keras.models.Sequential()

cnn.add(Input(shape=[64, 64, 3]))
cnn.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[7]:


cnn.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[8]:


cnn.add(tf.keras.layers.Flatten())


# In[9]:


cnn.add(tf.keras.layers.Dense(units=120, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))


# In[10]:


cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[11]:


cnn.fit(x=training_set, validation_data=test_set, epochs=25)


# In[15]:


import numpy as np
from keras.preprocessing import image

test_image = image.load_img("kopke.jpg", target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"
    
print(prediction)


# In[17]:


test_image = image.load_img("kedke.jpg", target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"
    
print(prediction)


# In[ ]:




