# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 00:01:54 2020

@author: Marko Pejić
"""


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

#%%
pkl_file  = '../US_trending.pkl'

df = pd.read_pickle(pkl_file)
images = []
i = 1
data = [];

_images = []
_labels = []

for video_id,image_url,category in zip(df['video_id'],df['thumbnail_link'],df['category_name']):
    response = requests.get(image_url)
    #img = Image.open(BytesIO(response.content))
    #img.save("images/{}".format(image_url))
    size = 224,224
    with Image.open(BytesIO(response.content)) as img:
        img = img.resize((224, 224), Image.ANTIALIAS)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
       
        _images.append(x)
        _labels.append(category)
        #features = model.predict(x)
        #data.append([video_id,image_url,features])
        #print(len(_images))
        i+=1
        
#%%
_images1 = np.array(_images)
_images2=np.rollaxis(_images1, 1, 0)

labels_dummies = pd.get_dummies(_labels)

#%%
# Generate a model with all layers (with top)
model = VGG16(weights='imagenet', include_top=True)

#Add a layer where input is the output of the  second last layer 
x = Dense(15, activation='softmax', name='predictions')(model.layers[-2].output)

#Then create the corresponding model 
my_model = Model(inputs=model.input, outputs=x)

#%%
for layer in my_model.layers[:10]:
    layer.trainable = False

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
my_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#%%
print(np.shape(_images1))
print(np.shape(_images2))

#%%
from sklearn.model_selection import train_test_split

images_train,images_test,category_train,category_test = train_test_split(_images2[0],labels_dummies,test_size = 0.2,random_state = 3)
category_train = np.asarray(category_train)
category_test = np.asarray(category_test)

len(images_train)

#%%
my_model.fit(images_train,category_train,epochs = 5, validation_data = (images_test, category_test),verbose = 2)

