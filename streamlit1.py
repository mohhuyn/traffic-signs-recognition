# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:15:42 2024

@author: belbekri mohammed bouziane
         brahimi youcef 
         chelabi amine  
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
st.set_page_config(layout="wide")

@st.cache_data  # Pour mettre les données dans la cache


# Fonction pour charger le modèle 
def load_model():
  # model=tf.keras.models.load_model('model.h5')
  model_architecture='model.json'
  model_weights='model.h5'
  model = model_from_json(open(model_architecture).read())
  model.load_weights(model_weights)
  
  return model

# Afficher temporairement un message lors de l'exécution d'un bloc de code.
with st.spinner('Model is being loaded..'):
  model=load_model()


#Ajouter un logo 
st.image("universite-toulouse-iii-paul-sabatier-logo-vector.png", width=200) 
# st.markdown('<div class="logo-container"><img src="universite-toulouse-iii-paul-sabatier-logo-vector.png" style="width: 100px;"></div>', unsafe_allow_html=True)


# Afficher un message
st.write("""
          # Traffic Sign Recognition 
          """
        )

# Afficher un widget de téléchargement de fichiers
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])

# C'est juste pour désactiver un warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# Fonction pour charger une image, la redimensionner, et la classer avec le modèle
def upload_predict(upload_image, model):
    
        # size = (224,224)    
        # image = ImageOps.fit(upload_image, size, Image.LANCZOS)
        # image = np.asarray(image)
        # image=  image.astype('float32')
        
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img = cv2.normalize(img, None, alpha=0 , beta=255, norm_type = cv2.NORM_MINMAX)

        # img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        
        # img_reshape =img_resize.reshape(-1,224,224,3)
        label_dic = {  0:'Speed limit (5km/h)'
                      ,1:'Speed limit (15km/h)'
                      ,2:'Speed limit (30km/h)'
                      ,3:'Speed limit (40km/h)'
                      ,4:'Speed limit (50km/h)'
                      ,5:'Speed limit (60km/h)'
                      ,6:'Speed limit (70km/h)'
                      ,7:'speed limit (80km/h)'
                      ,8:'Dont Go straight or left'
                      ,9:'Dont Go straight or Right'
                      ,10:'Dont Go straight'
                      ,11:'Dont Go Left'
                      ,12:'Dont Go Left or Right'
                      ,13:'Dont Go Right'
                      ,14:'Dont overtake from Left'
                      ,15:'No Uturn'
                      ,16:'No Car'
                      ,17:'No horn'
                      ,18:'Speed limit (40km/h)'
                      ,19:'Speed limit (50km/h)'
                      ,20:'Go straight or right'
                      ,21:'Go straight'
                      ,22:'Go Left'
                      ,23:'Go Left or right'
                      ,24:'Go Right'
                      ,25:'keep Left'
                      ,26:'keep Right'
                      ,27:'Roundabout mandatory'
                      ,28:'watch out for cars'
                      ,29:'Horn'
                      ,30:'Bicycles crossing'
                      ,31:'Uturn'
                      ,32:'Road Divider'
                      ,33:'Traffic signals'
                      ,34:'Danger Ahead'
                      ,35:'Zebra Crossing'
                      ,36:'Bicycles crossing'
                      ,37:'Children crossing'
                      ,38:'Dangerous curve to the left'
                      ,39:'Dangerous curve to the right'
                      ,40:'Unknown1'
                      ,41:'Unknown2'
                      ,42:'Unknown3'
                      ,43:'Go right or straight'
                      ,44:'Go left or straight'
                      ,45:'Unknown4'
                      ,46:'ZigZag Curve'
                      ,47:'Train Crossing'
                      ,48:'Under Construction'
                      ,49:'Unknown5'
                      ,50:'Fences'
                      ,51:'Heavy Vehicle Accidents'
                      ,52:'Unknown6'
                      ,53:'Give Way'
                      ,54:'No stopping'
                      ,55:'No entry'
                      ,56:'Unknown7'
                      ,57:'Unknown8'}
        
        size = (224,224)
        image = ImageOps.fit(upload_image, size, Image.LANCZOS)
        
        image_= np.asarray(image)
        image_.astype('float32')
        image2 = cv2.resize(image_,(224,224))/255
        image3_ = image2.reshape(-1,224,224,3)
    
        prediction = model.predict(image3_)
        pred_class=np.argmax(prediction)
        score = prediction[0][pred_class]
        label = label_dic[pred_class]
        return pred_class,score,label
    
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    # pour afficher une image
    st.image(image, use_column_width=True)
    predictions,score,label= upload_predict(image, model)
    image_class = str(predictions)
    
    st.write("The image is classified as",image_class, ':',label )
    st.write("The similarity score is approximately",str(score))
    print("The image is classified as ",image_class, "with a similarity score of",score)