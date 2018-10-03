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
    dataset_train_1 = glob(os.path.join(ruta_Data + train_1, '*'))[init:len_train]
    dataset_train_0 = glob(os.path.join(ruta_Data + train_0, '*'))[init:len_train]

    for _t1,_t0 in zip(dataset_train_1,dataset_train_0):
        # clname1 = os.path.basename(_t1)
        # clname2 = os.path.basename(_t0)
        try:
            img1 = image.load_img(_t1)
            npi1 = image.img_to_array(img1)
            npi1 = preprocess_input(npi1)
            #
            img0 = image.load_img(_t0)
            npi0 = image.img_to_array(img0)
            npi0 = preprocess_input(npi0)
        except:
            continue
        #
        x.append(npi1); y.append([1.,0.])
        x.append(npi0); y.append([0.,1.])


    return np.array(x), np.array(y)

# size = 130000
# x_train,y_train = load_data(0,size)
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

Layers_to_freeze = 1

for layer in Model.layers[:Layers_to_freeze]:
    layer.trainable = False

Model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])



# Model.fit(x_train, y_train, batch_size=10, epochs=1, verbose=1, validation_split=0.1)


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

# batch=10
# for n in range(0,len(x_train),batch):
#     _x_train = x_train[n:n+batch]
#     _y_train = y_train[n:n+batch]
#     Model.train_on_batch(_x_train,_y_train)

size = 116224
step = 10
epochs=1
# for epoch in range(epochs):
for i in range(0,size,step):
    (print(i,i+step))
    x_train, y_train = load_data(i,i+step)
    Model.fit(x_train, y_train, batch_size=10, epochs=1, verbose=1)

Model.save('trained_model_local.h5')


######### knn #############
x,y = load_data(0,1000)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 4)

x1 = np.array([elem.reshape(1,-1)[0] for elem in x])
y1 = np.array([elem[0] for elem in y])

knn.fit(x1,y1)

# print(knn.predict([[1.1]]))

x2,y2 = load_data(3000,3050)
y2_pred = np.array([knn.predict(_x.reshape(1,-1))[0]  for _x in x2])

y2_pred + y1[:100]
############################





# Model.get_weights()[0][0][0][0]
# Model.get_weights()[-1]

############## Entrenamiento del tiron #################
# len_train = 100 (200 imagenes, 100 con barcos y 100 sin barcos)
# Train on 180 samples, validate on 20 samples
# Epoch 1/1
#  10/180 [>.............................] - ETA: 5:56 - loss: 0.7604 - acc: 0.5000
#  20/180 [==>...........................] - ETA: 3:16 - loss: 4.7164 - acc: 0.3500
#  30/180 [====>.........................] - ETA: 2:20 - loss: 3.4103 - acc: 0.5000
#  40/180 [=====>........................] - ETA: 1:50 - loss: 4.0688 - acc: 0.4750
#  50/180 [=======>......................] - ETA: 1:31 - loss: 3.5244 - acc: 0.5000
#  60/180 [=========>....................] - ETA: 1:17 - loss: 3.0070 - acc: 0.5500
#  70/180 [==========>...................] - ETA: 1:06 - loss: 2.6511 - acc: 0.5571
#  80/180 [============>.................] - ETA: 57s - loss: 2.3649 - acc: 0.6000
#  90/180 [==============>...............] - ETA: 49s - loss: 2.1154 - acc: 0.6444
# 100/180 [===============>..............] - ETA: 42s - loss: 1.9357 - acc: 0.6700
# 110/180 [=================>............] - ETA: 36s - loss: 1.7938 - acc: 0.6727
# 120/180 [===================>..........] - ETA: 31s - loss: 1.6534 - acc: 0.7000
# 130/180 [====================>.........] - ETA: 25s - loss: 1.5675 - acc: 0.6923
# 140/180 [======================>.......] - ETA: 19s - loss: 1.4604 - acc: 0.7143
# 150/180 [========================>.....] - ETA: 14s - loss: 1.3702 - acc: 0.7333
# 160/180 [=========================>....] - ETA: 9s - loss: 1.3493 - acc: 0.7438
# 170/180 [===========================>..] - ETA: 4s - loss: 1.2829 - acc: 0.7529
# 180/180 [==============================] - 90s 499ms/step - loss: 1.2171 - acc: 0.7667 - val_loss: 4.0902 - val_acc: 0.5500
#
# Pesos
# Model.get_weights()[-1]
# Out[7]: array([ 0.00231642, -0.00227893], dtype=float32)
# Model.get_weights()[-2]
# Out[8]:
# array([[ 0.02612577,  0.00689129],
#        [ 0.04905951, -0.04099087],
#        [ 0.00314174,  0.01676204],
#        ...,
#        [ 0.00102266, -0.04615838],
#        [ 0.01280051, -0.01962278],
#        [-0.03938761,  0.02135803]], dtype=float32)
########################################################

############## Entrenamiento incremental con Model.fit #################
# 0 10
# Epoch 1/1
# 10/20 [==============>...............] - ETA: 23s - loss: 0.7041 - acc: 0.5000
# 20/20 [==============================] - 27s 1s/step - loss: 5.2596 - acc: 0.4000
# 10 20
# Epoch 1/1
# 10/20 [==============>...............] - ETA: 3s - loss: 2.3462 - acc: 0.6000
# 20/20 [==============================] - 8s 394ms/step - loss: 1.1861 - acc: 0.8000
# 20 30
# Epoch 1/1
# 10/20 [==============>...............] - ETA: 3s - loss: 0.6056 - acc: 0.9000
# 20/20 [==============================] - 8s 389ms/step - loss: 0.7890 - acc: 0.8500
# 30 40
# Epoch 1/1
# 10/20 [==============>...............] - ETA: 3s - loss: 0.0687 - acc: 1.0000
# 20/20 [==============================] - 7s 363ms/step - loss: 0.0345 - acc: 1.0000
# 40 50
# Epoch 1/1
# 10/20 [==============>...............] - ETA: 3s - loss: 8.9221e-04 - acc: 1.0000
# 20/20 [==============================] - 7s 373ms/step - loss: 0.1441 - acc: 0.9500
# 50 60
# Epoch 1/1
# 10/20 [==============>...............] - ETA: 3s - loss: 1.0319 - acc: 0.8000
# 20/20 [==============================] - 7s 366ms/step - loss: 0.8605 - acc: 0.8500
# 60 70
# Epoch 1/1
# 10/20 [==============>...............] - ETA: 3s - loss: 1.0635 - acc: 0.8000
# 20/20 [==============================] - 7s 364ms/step - loss: 0.6646 - acc: 0.8500
# 70 80
# Epoch 1/1
# 10/20 [==============>...............] - ETA: 3s - loss: 0.2250 - acc: 0.9000
# 20/20 [==============================] - 7s 365ms/step - loss: 0.2535 - acc: 0.8500
# 80 90
# Epoch 1/1
# 10/20 [==============>...............] - ETA: 3s - loss: 3.0237 - acc: 0.7000
# 20/20 [==============================] - 7s 364ms/step - loss: 1.6133 - acc: 0.7500

# Model.get_weights()[-1]
# Out[4]: array([ 0.00074412, -0.00089787], dtype=float32)
# Model.get_weights()[-2]
# Out[5]:
# array([[-0.05434169,  0.01607815],
#        [ 0.00144881,  0.03503634],
#        [-0.01557056,  0.05056633],
#        ...,
#        [-0.02063462, -0.02879491],
#        [-0.01914319,  0.02615018],
#        [ 0.02248883,  0.03213151]], dtype=float32)

########################################################

############## Entrenamiento incremental con Model.train_on_batch #################

# Model.get_weights()[-1]
# Out[4]: array([ 0.00043822, -0.00042999], dtype=float32)
#
# Model.get_weights()[-2]
# Out[5]:
# array([[-0.03889852, -0.01649301],
#        [-0.03234083,  0.0070555 ],
#        [ 0.04612884, -0.03123381],
#        ...,
#        [ 0.04815121,  0.04477253],
#        [ 0.00787809, -0.02572643],
#        [-0.03482797, -0.0309715 ]], dtype=float32)
########################################################




