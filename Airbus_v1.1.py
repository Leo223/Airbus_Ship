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



###############################################################
# Clustering de imagenes test, Creacion de ventanas para leer la imagen
# foto = '0005d01c8.jpg'
# data = ruta + foto
# ref = Image.open(data)
# imag = Image.open(data)
# im2 = imag.convert('L')
# im2 = image.img_to_array(im2)
# im2.resize((768,768))
# kmeans = KMeans(n_clusters=8, random_state=0).fit(im2)

# Criterio de separacion
# n_clus = kmeans.n_clusters
# clusters_size = np.array([list(kmeans.labels_).count(clus) for clus in range(n_clus)])
# cluster_water = list(clusters_size).index(clusters_size.max())
# criterio = kmeans.cluster_centers_[cluster_water].max()
# criterio = round(criterio*1.1)


# Hacemos un kmeans y lo guardamos como imagen para
# verificar los

# foto = '0005d01c8.jpg'
# data = ruta + foto
# # ref = Image.open(data)
# imag = Image.open(data)
# im2 = imag.convert('L')
# im2 = image.img_to_array(im2)
# im2.resize((768,768))
# im3=[]
# for i in im2:
#     for j in i: im3.append([j])
#
# km2 = KMeans(n_clusters=8, random_state=0).fit(im3)
# n_clus = km2.n_clusters
# clusters_size = np.array([list(km2.labels_).count(clus) for clus in range(n_clus)])
# cl_ship = list(clusters_size).index(clusters_size.min())
#
# _h = imag.size[0]
# _w = imag.size[1]
# ref = _h*_w
# x=np.array([km2.labels_[i*_h:i*_h+_h] for i in range(_h)])
# row = np.array([list(i).count(cl_ship) for i in x])
# col = np.array([list(i).count(cl_ship) for i in x.T])
#
#
# import scipy.misc
# scipy.misc.imsave('outfile.jpg', x)






# for i in range(768):
#     imag.putpixel((i,263), (0, 255, 0))
#     imag.putpixel((i,308), (0, 255, 0))
#     imag.putpixel((i,616), (0, 255, 0))
#     imag.putpixel((i,723), (0, 255, 0))





# criterio = 30
#
#

##########################################
#########  Entrenamiento #################
##########################################

##### Cargamos los datos
train_1 = '/Data_generated_1'
ruta_Data = os.getcwd() + '/Data'
k=glob(os.path.join(ruta_Data + train_1, '*'))
k1=k[-1]
imgp = image.load_img(k1)
npip = image.img_to_array(imgp)
npip = preprocess_input(npip)
npip = np.array(npip)
npip=npip.reshape((1,299,299,3))


def load_data():
    ruta_Data = os.getcwd() + '/Data'
    x = []
    y = []
    len_train = 10
    train_1 = '/Data_generated_1'
    train_0 = '/Data_generated_0'
    dataset_tain_1 = glob(os.path.join(ruta_Data + train_1, '*'))[:len_train]
    dataset_tain_0 = glob(os.path.join(ruta_Data + train_0, '*'))[:len_train]

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

x_train,y_train = load_data()
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


Model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1, validation_split=0.1)







# batch=2
# for n in range(0,len(x_train),batch):
#     _x_train = x_train[n:n+batch]
#     _y_train = y_train[n:n+batch]
#     Model.train_on_batch(_x_train,_y_train)




############## Entrenamiento del tiron #################
# Train on 18 samples, validate on 2 samples
# Epoch 1/5
# 18/18 [==============================] - 31s 2s/step - loss: 0.7961 - acc: 0.3889 - val_loss: 4.9066 - val_acc: 0.5000
# Epoch 2/5
# 18/18 [==============================] - 9s 491ms/step - loss: 3.1245 - acc: 0.5000 - val_loss: 0.0166 - val_acc: 1.0000
# Epoch 3/5
# 18/18 [==============================] - 7s 397ms/step - loss: 0.0826 - acc: 0.9444 - val_loss: 4.1257 - val_acc: 0.5000
# Epoch 4/5
# 18/18 [==============================] - 7s 384ms/step - loss: 2.8074 - acc: 0.6667 - val_loss: 2.8865 - val_acc: 0.5000
# Epoch 5/5
# 18/18 [==============================] - 7s 382ms/step - loss: 0.4856 - acc: 0.8889 - val_loss: 4.4753 - val_acc: 0.5000
# Out[4]: <keras.callbacks.History at 0x7f089ad09080>
########################################################

############## Entrenamiento incremental #################





########################################################








