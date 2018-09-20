from PIL import Image,ImageFilter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import np_utils

import os
from glob import glob

ruta = os.getcwd() + '/Data/train/'
foto = '0005d01c8.jpg'
data = ruta + foto

csv_targets = os.getcwd() + '/Data/train_ship_segmentations.csv'

# df de los targets
df = pd.read_csv(csv_targets)
df = df.fillna(value=0)

targets = {}
for row in df.iterrows():
    foto = row[1][0]
    data_foto = row[1][1]
    if targets.get(foto) == None:
        targets[foto] = {'Data': [], 'Cuadriculas': []}
    Data = targets.get(foto).get('Data')
    Cuadriculas = targets.get(foto).get('Cuadriculas')
    Data.append(data_foto)
    if data_foto == 0: continue
    else: data_foto = data_foto.split()
    pix_pos = np.array(data_foto[::2] ,dtype=np.int64)
    pix_len = np.array(data_foto[1::2],dtype=np.int64)

    # Posiciones de los vertices del recuadro que encuadra el barco
    left  = min(pix_pos//768)-15 #(left,i)
    right = max(pix_pos//768)+15 #(right,i)
    top = min(pix_pos%768)-15 #(i,top)
    bottom = max(pix_pos%768 + pix_len)+15 #(i,bottom)

    if top < 0: top = 0
    if bottom > 767: bottom=767

    barco = [top,bottom,right,left]
    cuadricula = [(left,top),(right,top),(left,bottom),(right,bottom)]

    Cuadriculas.append(cuadricula)
    targets[foto] = {'Data': Data, 'Cuadriculas': Cuadriculas}


##########################################
#########  Entrenamiento #################
##########################################

##### Cargamos los datos

def load_data(init=0, len_train=100):
    ruta_Data = os.getcwd() + '/Data'
    x = []
    y = []
    train_1 = '/Data_generated_1'
    train_0 = '/Data_generated_0'
    dataset_tain_1 = glob(os.path.join(ruta_Data + train_1, '*'))[init:len_train]
    dataset_tain_0 = glob(os.path.join(ruta_Data + train_0, '*'))[init:len_train]

    for _t1,_t0 in zip(dataset_tain_1,dataset_tain_0):
        # clname1 = os.path.basename(_t1)
        # clname2 = os.path.basename(_t0)
        img1 = image.load_img(_t1)
        npi1 = image.img_to_array(img1)
        npi1 = preprocess_input(npi1)
        #
        img0 = image.load_img(_t0)
        npi0 = image.img_to_array(img0)
        npi0 = preprocess_input(npi0)
        #
        x.append(npi1); y.append([1.,0.])
        x.append(npi0); y.append([0.,1.])


    return np.array(x), np.array(y)


x_train,y_train = load_data(0,100)
# y_train = np_utils.to_categorical(y_train, 2)
# [1.,0.] --> Barco
# [0.,1.] --> Mar o No Barco


##### Modelo Red Neuronal

base_model = InceptionV3(weights='imagenet', include_top=False)

def add_new_last_layer(base_model, nb_classes):

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    neuronas_out = base_model.output.get_shape()[-1].value
    x = Dense(2048,activation = 'relu')(x)
    x_pred = Dense(nb_classes,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=x_pred)
    return model

nb_classes = 2
Model = add_new_last_layer(base_model,nb_classes)

Layers_to_freeze = 10

for layer in Model.layers[:Layers_to_freeze]:
    layer.trainable = False

Model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])



Model.fit(x_train, y_train, batch_size=10, epochs=1, verbose=1, validation_split=0.1)


#### Comprobaciones
#
# train_1 = '/Data_generated_1'
# ruta_Data = os.getcwd() + '/Data'
# k=glob(os.path.join(ruta_Data + train_1, '*'))
# k1=k[-1]
# imgp = image.load_img(k1)
# npip = image.img_to_array(imgp)
# npip = preprocess_input(npip)
# npip = np.array(npip)
# npip=npip.reshape((1,299,299,3))
#

# batch=2
# for n in range(0,len(x_train),batch):
#     _x_train = x_train[n:n+batch]
#     _y_train = y_train[n:n+batch]
#     Model.train_on_batch(_x_train,_y_train)



# Model.get_weights()[0][0][0][0]

############## Entrenamiento del tiron #################
# len_train = 100 (200 imagenes, 100 con barcos y 100 sin barcos)
#Train on 180 samples, validate on 20 samples
# Epoch 1/1
#  10/180 [>.............................] - ETA: 5:55 - loss: 0.7041 - acc: 0.6000
#  20/180 [==>...........................] - ETA: 3:17 - loss: 0.5082 - acc: 0.7000
#  30/180 [====>.........................] - ETA: 2:22 - loss: 0.3401 - acc: 0.8000
#  40/180 [=====>........................] - ETA: 1:52 - loss: 0.7206 - acc: 0.7750
#  50/180 [=======>......................] - ETA: 1:33 - loss: 0.7788 - acc: 0.7800
#  60/180 [=========>....................] - ETA: 1:18 - loss: 0.6492 - acc: 0.8167
#  70/180 [==========>...................] - ETA: 1:07 - loss: 0.8185 - acc: 0.8143
#  80/180 [============>.................] - ETA: 58s - loss: 0.9046 - acc: 0.8125
#  90/180 [==============>...............] - ETA: 50s - loss: 0.8347 - acc: 0.8222
# 100/180 [===============>..............] - ETA: 43s - loss: 0.7522 - acc: 0.8400
# 110/180 [=================>............] - ETA: 36s - loss: 0.6929 - acc: 0.8455
# 120/180 [===================>..........] - ETA: 30s - loss: 0.6355 - acc: 0.8583
# 130/180 [====================>.........] - ETA: 25s - loss: 0.5870 - acc: 0.8692
# 140/180 [======================>.......] - ETA: 19s - loss: 0.6351 - acc: 0.8500
# 150/180 [========================>.....] - ETA: 14s - loss: 0.6125 - acc: 0.8533
# 160/180 [=========================>....] - ETA: 9s - loss: 0.5866 - acc: 0.8625
# 170/180 [===========================>..] - ETA: 4s - loss: 0.6547 - acc: 0.8471
# 180/180 [==============================] - 91s 504ms/step - loss: 0.6284 - acc: 0.8500 - val_loss: 2.4055 - val_acc: 0.8500
# Out[2]: <keras.callbacks.History at 0x7f4508ad55f8>
#
# Pesos
# Model.get_weights()[0][0][0][0]
# array([-0.45910555, -0.04145266, -0.00362577, -0.09876725, -0.03370709,
       #  0.0479929 ,  0.23254214,  0.32392767,  0.05901601,  0.09477382,
       #  0.04249961,  0.12662047,  0.13321598,  0.12274183, -0.07926863,
       #  0.0208228 , -0.19964783, -0.30268797, -0.21065992, -0.35289842,
       # -0.5580231 ,  0.3202231 ,  0.00453596, -0.03092664, -0.06869579,
       #  0.20096852,  0.11454275,  0.24037288,  0.01529435,  0.05962313,
       # -0.05857147,  0.87817335], dtype=float32)
########################################################

############## Entrenamiento incremental con Model.fit #################
# x_train,y_train = load_data(80,90)
# Model.fit(x_train, y_train, batch_size=10, epochs=1, verbose=1, validation_split=0.1)
# Train on 18 samples, validate on 2 samples
# Epoch 1/1
# 10/18 [===============>..............] - ETA: 3s - loss: 0.0477 - acc: 1.0000
# 18/18 [==============================] - 8s 437ms/step - loss: 0.0399 - acc: 1.0000 - val_loss: 8.0151 - val_acc: 0.5000
# Out[22]: <keras.callbacks.History at 0x7f27ffc93a20>
# x_train,y_train = load_data(90,100)
# Model.fit(x_train, y_train, batch_size=10, epochs=1, verbose=1, validation_split=0.1)
# Train on 18 samples, validate on 2 samples
# Epoch 1/1
# 10/18 [===============>..............] - ETA: 3s - loss: 0.0597 - acc: 1.0000
# 18/18 [==============================] - 9s 491ms/step - loss: 0.1756 - acc: 0.9444 - val_loss: 9.3887 - val_acc: 0.0000e+00
# Out[24]: <keras.callbacks.History at 0x7f280821c630>

# array([-0.45910555, -0.04145266, -0.00362577, -0.09876725, -0.03370709,
#         0.0479929 ,  0.23254214,  0.32392767,  0.05901601,  0.09477382,
#         0.04249961,  0.12662047,  0.13321598,  0.12274183, -0.07926863,
#         0.0208228 , -0.19964783, -0.30268797, -0.21065992, -0.35289842,
#        -0.5580231 ,  0.3202231 ,  0.00453596, -0.03092664, -0.06869579,
#         0.20096852,  0.11454275,  0.24037288,  0.01529435,  0.05962313,
#        -0.05857147,  0.87817335], dtype=float32)
########################################################

############## Entrenamiento incremental con Model.train_on_batch #################


########################################################




