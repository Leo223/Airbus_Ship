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


im1 = Image.open(os.getcwd() + '/Data/train' + '/0005d01c8.jpg')
imagen = nombre_fotos[5]
def Anal_imag(imagen):
    ruta_Data = os.getcwd() + '/Data/test/'+ imagen
    im1 = Image.open(ruta_Data)
    im2 = im1.convert('L')
    npi1 = image.img_to_array(im2)
    npi1 = npi1.reshape(768,768)
    # kuni = [j[0] for i in npi1 for j in i]
    # km = KMeans(n_clusters=2, random_state=0).fit(npi1)
    # clus = km.labels_.reshape(768,768)


imagen = '0005d01c8.jpg'

def ventanas(ruta_Data):
    # try:
    #     ruta_Data = os.getcwd() + '/Data/test/'+ imagen
    # except:
    #     ruta_Data = os.getcwd() + '/Data/train/' + imagen

    im1 = Image.open(ruta_Data)

    angles = [0,270]
    vertices_1=[]
    vertices_2=[]
    for angl in angles:

        im1 = im1.rotate(angl)
        im2 = im1.convert('L')
        npi1 = image.img_to_array(im2)
        npi1 = npi1.reshape(768, 768)

        km = KMeans(n_clusters=8, random_state=0).fit(npi1)
        loc = km.labels_

        cont_clus = [list(loc).count(clus) for clus in range(km.n_clusters)]
        No_barcos_clus = cont_clus.index(max(cont_clus))
        positions = [i[0] for i in enumerate(loc) if i[1]!= No_barcos_clus]

        intervals=[]
        n1 = np.array(positions)
        _pos = positions[1:]
        _pos.append(positions[-1]+1)
        n2 = np.array(_pos)
        dif = n2-n1
        # _loc_w = [val[0] for val in enumerate(dif) if val[1]!=1]
        _loc_w = []
        for val in enumerate(dif):
            if val[1]!=1:
                _loc_w.append(positions[val[0]])
                _loc_w.append(positions[val[0]+1])

        v0 = [positions[0]]
        vf = [positions[-1]]
        loc_w = v0 + _loc_w + vf
        # v1 = [(loc_w[elem - 1], loc_w[elem]) for elem in range(1, len(loc_w), 2)]
        if angl ==0: vertices_1 = loc_w
        else: vertices_2=loc_w

    im1 = im1.rotate(-angl)
    return vertices_2,vertices_1

ruta_Data = os.getcwd() + '/Data/train' + '/0005d01c8.jpg'
# im1 = Image.open(os.getcwd() + '/Data/train' + '/0005d01c8.jpg')
# imagen = '0005d01c8.jpg'
imag=im1

x,y = ventanas(ruta_Data)
# [[(263, 308), (616, 723)], [(85, 184), (186, 215), (484, 534), (536, 548)]]
for _x in x:
    for _y in y: imag.putpixel((_x,_y), (0, 255, 0))








########################################################




# '/home/jcambronero/Escritorio/JCP/Cursos/Kaggle/Airbus_Ship/Data/test/501eff990.jpg'
# ship = im1.crop((620,630,720,760))




# loc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])






