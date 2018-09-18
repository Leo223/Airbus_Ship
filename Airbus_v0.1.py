from PIL import Image,ImageFilter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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


# foto = '00021ddc3.jpg'
# data = ruta + foto
# imag = Image.open(data)
# barcos=targets.get(foto).get('Cuadriculas')
# b1=barcos[0]
# v1=b1[0]
# v2=b1[-1]
# box = (v1[0],v1[1],v2[0],v2[1])

### Crear las imagenes input
# for picture in targets:
#     datos = targets.get(picture).get('Data')
#     if datos[0] == 0: continue
#     data = ruta + picture
#     imag = Image.open(data)
#     barcos = targets.get(picture).get('Cuadriculas')
#     for barco in enumerate(barcos):
#         v1 = barco[1][0]
#         v2 = barco[1][-1]
#         box = (v1[0], v1[1], v2[0], v2[1])
#         ship = imag.crop(box)
#         ship = ship.resize((299,299)).filter(ImageFilter.BLUR)
#         giros = [0,90,180,270]
#         for giro in giros:
#             ship_giro = ship.rotate(giro)
#             ship_giro.save(os.getcwd()+'/Data_generated/'+picture +'_barco_'+str(giro) +'_'+ str(barco[0]+1)+'.png')

# def reset():
#     foto = '0005d01c8.jpg'
#     data = ruta + foto
#     imag = Image.open(data)
#     imag.save('p2.png')
#
# imag.save('p2.png')


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
# kmeans.labels_
# kmeans.cluster_centers_
# kmeans.n_clusters

# for i in range(kmeans.n_clusters):
#     max=kmeans.cluster_centers_[i].max()
#     min=kmeans.cluster_centers_[i].min()
#     mean=kmeans.cluster_centers_[i].mean()
#     print ('*********')
#     print (str(i) +' ---->'+ str(list(kmeans.labels_).count(i)))
#     print ('max: '+str(max)+'// min: ' + str(min) +'// mean: '+ str(mean))
#     print ('*********')

# Criterio de separacion

n_clus = kmeans.n_clusters
clusters_size = np.array([list(kmeans.labels_).count(clus) for clus in range(n_clus)])
cluster_water = list(clusters_size).index(clusters_size.max())
criterio = kmeans.cluster_centers_[cluster_water].max()
criterio = round(criterio*1.1)

vertices=[]
data = kmeans.labels_
cluster = kmeans.cluster_centers_
for clus in enumerate(data):
    if clus[1] == cluster_water:    continue
    try:
        if data[clus[0]+1] == cluster_water or data[clus[0]-1] == cluster_water:
            pos = [[pto] for pto in cluster[clus[1]]]
            x=clus[0]
            # se crea un cluster para la linea, para encontrar la segunda coordenada
            subkmeans = KMeans(n_clusters=2, random_state=0).fit(pos)
            _ship_clus = np.array([list(subkmeans.labels_).count(clus) for clus in range(subkmeans.n_clusters)])
            _clus_ship = list(_ship_clus).index(_ship_clus.min())

            y=[]
            data2 =subkmeans.labels_
            vertices_aux=[]
            for sub in enumerate(data2):
                if sub[1] != _clus_ship: continue
                try:
                    if data2[sub[0]-1] != _clus_ship or data2[sub[0]+1] != _clus_ship:
                        y.append(sub[0])
                except:
                    continue
            for _y in y:
                vertices_aux.append((_y,x))

            vertices.append(vertices_aux)
    except:
        continue



# for i in range(768):
#     imag.putpixel((i,263), (0, 255, 0))
#     imag.putpixel((i,308), (0, 255, 0))
#     imag.putpixel((i,616), (0, 255, 0))
#     imag.putpixel((i,723), (0, 255, 0))



for vert in vertices:
    for v in vert:
        imag.putpixel(v,(255,0,0))


def windows(vertices):
    for _v in vertices:
        y =_v[1]













quit()

# criterio = 30
#
intervals = []
for cluster in range(kmeans.n_clusters):
    limites = np.array([int(i[0]) for i in enumerate(kmeans.cluster_centers_[cluster]) if i[1] > criterio])
    if len(limites)==0: continue
    min = limites.min()
    max = limites.max()
    intervals.append((min,max))
    for i in range(768):
        imag.putpixel((min,i),(0,255,0))
        imag.putpixel((max,i),(0,255,0))

imag = imag.transpose(Image.ROTATE_270)
im2 = ref.transpose(Image.ROTATE_270).convert('L')
im2 = image.img_to_array(im2)
im2.resize((768,768))
kmeans = KMeans(n_clusters=8, random_state=0).fit(im2)

for cluster in range(kmeans.n_clusters):
    limites = np.array([int(i[0]) for i in enumerate(kmeans.cluster_centers_[cluster]) if i[1] > criterio])
    if len(limites)==0: continue
    min = limites.min()
    max = limites.max()
    intervals.append((min,max))
    for i in range(768):
        imag.putpixel((min,i),(0,255,0))
        imag.putpixel((max,i),(0,255,0))

imag = imag.transpose(Image.ROTATE_90)
####################################################################









imag = Image.open(data)











# Tamano de la imagen
size = imag.size #(768,768)
size_x = size[0]


##### Modelo Red Neuronal

# base_model = InceptionV3(weights='imagenet', include_top=False)
#
# def add_new_last_layer(base_model, nb_classes):
#
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(500,activation = 'relu')(x)
#     x_pred = Dense(nb_classes,activation='softmax')(x)
#     model = Model(input=base_model.inputs,output=x_pred)
#     return model
#
# Model = add_new_last_layer(base_model,)
#
# Layers_to_freeze = 500

# for layer in




