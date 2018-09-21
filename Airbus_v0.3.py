from PIL import Image,ImageFilter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.layers import GlobalAveragePooling2D

import os

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
n_clus = kmeans.n_clusters
clusters_size = np.array([list(kmeans.labels_).count(clus) for clus in range(n_clus)])
cluster_water = list(clusters_size).index(clusters_size.max())
criterio = kmeans.cluster_centers_[cluster_water].max()
criterio = round(criterio*1.1)

centros=[]
data = kmeans.labels_
cluster = kmeans.cluster_centers_
for clus in enumerate(data)[1:-1]:
    if clus[1] == cluster_water:    continue
    index = clus[0]
    clust = clus[1]
    if data[index+1] == cluster_water or data[index-1] == cluster_water:
        pos = [[pto] for pto in cluster[clust]]
        x=index
        # se crea un cluster para la linea, para encontrar la segunda coordenada
        subkmeans = KMeans(n_clusters=2, random_state=0).fit(pos)
        _ship_clus = np.array([list(subkmeans.labels_).count(clus) for clus in range(subkmeans.n_clusters)])
        _clus_ship = list(_ship_clus).index(_ship_clus.min())

        y=[]
        data2 =subkmeans.labels_
        vertices_aux=[]
        for sub in enumerate(data2)[1:-1]:
            index2=sub[0]
            valor =sub[1]
            if valor != _clus_ship: continue
            if data2[index2-1] != _clus_ship or data2[index2+1] != _clus_ship:
                y.append(index2)

            for _y in y:
                vertices_aux.append((_y,x))

            vertices.append(vertices_aux)

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
    for j in i: im3.append([j[0]])

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





# Array to image
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


