# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:49:22 2024

@author: BELBEKRI Mohammed Bouziane 
         BRAHIMI Youcef
         CHELABI Amine
"""

import splitfolders
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, regularizers, optimizers  
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import VGG16 
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import model_from_json
from keras.models import Model 



# splitfolders.ratio("C:/Users/ML/Desktop/Panneaux/archive/traffic_Data/DATA",
#                     output="output", seed=1337, ratio=(.8,.1,.1), group_prefix=None, 
#                     move = False)


datagen_train = ImageDataGenerator(rescale=1./255,
                            width_shift_range=0.1, 
                            height_shift_range=0.1, 
                            brightness_range=[0.9,1.0],
                            zoom_range=0.2,
                            shear_range = 0.1,
                            rotation_range=10
                            ) 

train_data = datagen_train.flow_from_directory(
                            directory= r"C:\Users\ML\Documents\PROJET\output\train",
                            batch_size=32, 
                            target_size=(224,224),
                            color_mode="rgb",
                            class_mode="categorical",
                            shuffle=True,
                            seed=1337
                            )

datagen_test = ImageDataGenerator(rescale=1./255) 

test_data = datagen_test.flow_from_directory(
                            directory= r"C:\Users\ML\Documents\PROJET\output\test",
                            batch_size=32, 
                            target_size=(224,224),
                            color_mode="rgb",
                            class_mode="categorical",
                            shuffle=False,
                            seed=1337
                            )

datagen_val = ImageDataGenerator(rescale=1./255) 

val_data = datagen_val.flow_from_directory(
                            directory= r"C:\Users\ML\Documents\PROJET\output\val",
                            batch_size=32, 
                            target_size=(224,224),
                            color_mode="rgb",
                            class_mode="categorical",
                            shuffle=True,
                            seed=1337
                            )

base_model = VGG16(include_top=False,weights='imagenet', input_shape=(224, 224, 3)) 


for layer in base_model.layers[:17]: 
 layer.trainable = False 


# Model Creation !!!!!

model=models.Sequential([ base_model])


# model.add(layers.Conv2D(64,(3,3), padding='same',
#                         input_shape=(150,150,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2),
                              strides=(2,2)))
model.add(layers.Dropout(0.3))

# model.add(layers.Conv2D(64,(3,3), padding='same',
#                         input_shape=(75,75,64), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D(pool_size=(2,2),
#                               strides=(2,2)))

Classes=58

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(Classes, activation='softmax'))

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

model.fit(train_data, batch_size=32, epochs=20,
          verbose=1, validation_data = val_data)



################################################################

Model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(Model_json)
    model.save_weights('model.h5')
    
######################################################Prediction 

#filename =  r"C:\Users\ML\Documents\PROJET\output\test\00\000_1_0009.png"
filename =  r"C:\Users\ML\Documents\PROJET\Panneaux\archive\traffic_Data\TEST\035_1_0015_1_j.png"
image = Image.open(filename)
plt.imshow(image)

Image_= np.asarray(image)
Image_.astype('float32')
Image2 = cv.resize(Image_,(224,224))/255
Image3_ = Image2.reshape(-1,224,224,3)

predictions = model.predict(Image3_)
print(predictions)
print(np.argmax(predictions))
print(model.evaluate(test_data))
Matrice_confusion = confusion_matrix(test_data,predictions)



########################### Chargé le model 

model_architecture='model.json'
model_weights='model.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()
base_model = model.layers[0] 
base_model.summary()

# Récupérer les filtres et les poids de la 1ère couche 
filters, biases = base_model.layers[1].get_weights() 

# normaliser les filtres sur [0 , 1] 
f_min, f_max = filters.min(), filters.max() 
filters = (filters - f_min) / (f_max - f_min)

f=filters[:,:,:,0] 
plt.subplot(1,3,1) 
plt.imshow(f[:, :, 0], cmap='gray') 
plt.subplot(1,3,2) 
plt.imshow(f[:, :, 1], cmap='gray') 
plt.subplot(1,3,3) 
plt.imshow(f[:, :, 2], cmap='gray') 
plt.show() 

#################### 

inter_model = Model(inputs=model.inputs, outputs=model.layers[1].output) 
inter_model.summary() 
feature_maps = inter_model.predict(train_data.next()[0]) 

plt.imshow(feature_maps[0,:,:,6]) 


