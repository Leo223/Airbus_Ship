from PIL import Image,ImageFilter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.utils import np_utils

import os
from glob import glob

###############################################################

#

##########################################
#########  Entrenamiento #################
##########################################

##### Cargamos los datos de test
ruta_Data = os.getcwd() + '/Data'
test = '/test'

def load_data_test(init=0,len_test = 10):
    ruta_Data = os.getcwd() + '/Data'
    x = []
    nombre_foto = []
    test = '/test'
    dataset_test = glob(os.path.join(ruta_Data + test, '*'))[init:len_test]
    for imgtest in dataset_test:
        im1 = Image.open(imgtest)
        im1 = im1.resize((299, 299))
        npi1 = image.img_to_array(im1)
        npi1 = preprocess_input(npi1)
        npi1 = npi1.reshape((1, 299, 299, 3))
        x.append(npi1)
        nombre_foto.append(imgtest.split('/')[-1])
        # [1.,0.] --> Barco
        # [0.,1.] --> Mar o No Barco
    return np.array(x),nombre_foto



def trans_image(image_in):
    im1 = image_in.resize((299, 299))
    npi1 = image.img_to_array(im1)
    npi1 = preprocess_input(npi1)
    npi1 = npi1.reshape((1, 299, 299, 3))
    return npi1



##### Modelo Red Neuronal

#########
########

y,nombre_fotos= load_data_test()

Model = load_model( os. getcwd() + '/Trained_model.h5')


results={}
for _y,nombre in zip(y,nombre_fotos):
    pred = Model.predict(_y)
    tag = list(pred[0]).index(max(list(pred[0])))
    if tag == 0: valor ='BARCO'
    else: tag = '----'
    results[nombre] = (pred,tag)



# Model.predict(trans_image(ship))
k=y[0]
kuni=[[j[0]] for i in k for j in i]



imagen = nombre_fotos[2]
def Anal_imag(imagen):
    ruta_Data = os.getcwd() + '/Data/test/'+ imagen
    im1 = Image.open(ruta_Data)
    im1 = im1.convert('L')
    npi1 = image.img_to_array(im1)
    # kuni = [[j[0]] for i in npi1 for j in i]
    km = KMeans(n_clusters=2, random_state=0).fit(npi1)
    clus = km.labels_.reshape(768,768)










from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)

x1 = np.array([elem.reshape(1,-1)[0] for elem in x])
y1 = np.array([elem[0] for elem in y])

knn.fit(x1,y1)

# print(knn.predict([[1.1]]))

x2,y2 = load_data(5000,5050)
y2_pred = np.array([knn.predict(_x.reshape(1,-1))[0]  for _x in x2])

y2_pred + y1[:100]








########################################################




# '/home/jcambronero/Escritorio/JCP/Cursos/Kaggle/Airbus_Ship/Data/test/501eff990.jpg'
# ship = im1.crop((620,630,720,760))


