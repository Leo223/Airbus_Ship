from PIL import Image,ImageFilter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.layers import GlobalAveragePooling2D

import os
import glob


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
foto = '0005d01c8.jpg'
data = ruta + foto
ref = Image.open(data)
imag = Image.open(data)
im2 = imag.convert('L')
im2 = image.img_to_array(im2)
im2.resize((768,768))
kmeans = KMeans(n_clusters=8, random_state=0).fit(im2)

# Criterio de separacion
# n_clus = kmeans.n_clusters
# clusters_size = np.array([list(kmeans.labels_).count(clus) for clus in range(n_clus)])
# cluster_water = list(clusters_size).index(clusters_size.max())
# criterio = kmeans.cluster_centers_[cluster_water].max()
# criterio = round(criterio*1.1)


# Hacemos un kmeans y lo guardamos como imagen para
# verificar los

foto = '0005d01c8.jpg'
data = ruta + foto
# ref = Image.open(data)
imag = Image.open(data)
im2 = imag.convert('L')
im2 = image.img_to_array(im2)
im2.resize((768,768))
im3=[]
for i in im2:
    for j in i: im3.append([j])

km2 = KMeans(n_clusters=8, random_state=0).fit(im3)
n_clus = km2.n_clusters
clusters_size = np.array([list(km2.labels_).count(clus) for clus in range(n_clus)])
cl_ship = list(clusters_size).index(clusters_size.min())

_h = imag.size[0]
_w = imag.size[1]
ref = _h*_w
x=np.array([km2.labels_[i*_h:i*_h+_h] for i in range(_h)])
row = np.array([list(i).count(cl_ship) for i in x])
col = np.array([list(i).count(cl_ship) for i in x.T])






import scipy.misc
scipy.misc.imsave('outfile.jpg', x)






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

def load_data(path, pattern):
    class_names={}
    class_id=0
    x = []
    y = []

    for d in glob.glob(os.path.join(path, '*')):
        clname = os.path.basename(d)
        for f in glob.glob(os.path.join(d, pattern)):
            if not clname in class_names:
                class_names[clname]=class_id
                class_id += 1

            img = image.load_img(f)
            npi = image.img_to_array(img)
            npi = preprocess_input(npi)
            # for i in range(4):
                # npi=np.rot90(npi, i)
            x.append(npi)
            y.append(class_names[clname])

    return np.array(x), np.array(y), class_names





##### Modelo Red Neuronal

base_model = InceptionV3(weights='imagenet', include_top=False)

def add_new_last_layer(base_model, nb_classes):

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(500,activation = 'relu')(x)
    x_pred = Dense(nb_classes,activation='softmax')(x)
    model = Model(input=base_model.inputs,output=x_pred)

    return model

nb_classes = 2
Model = add_new_last_layer(base_model,nb_classes)

Layers_to_freeze = 500

for layer in model.layers[:Layers_to_freeze]:
    layer.trainable = False

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.1)





