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



# se crean las imagenes para entrenar (imagenes con barcos) --> 1
for picture in targets:
    datos = targets.get(picture).get('Data')
    if datos[0] == 0: continue
    data = ruta + picture
    imag = Image.open(data)
    barcos = targets.get(picture).get('Cuadriculas')
    for barco in enumerate(barcos):
        v1 = barco[1][0]
        v2 = barco[1][-1]
        box = (v1[0], v1[1], v2[0], v2[1])
        ship = imag.crop(box)
        ship = ship.resize((299,299)).filter(ImageFilter.BLUR)
        giros = [0,90,180,270]
        for giro in giros:
            ship_giro = ship.rotate(giro)
            ship_giro.save(os.getcwd()+'/Data/Data_generated_1/'+picture +'_barco_'+str(giro) +'_'+ str(barco[0]+1)+'.png')


# se crean las imagenes para entrenar (imagenes sin barcos) --> 0
for picture in targets:
    datos = targets.get(picture).get('Data')
    if datos[0] != 0: continue
    data = ruta + picture
    imag = Image.open(data)
    imag = imag.resize((299, 299)).filter(ImageFilter.BLUR)
    giros = [0, 90, 180, 270]
    for giro in giros:
        imag_giro = imag.rotate(giro)
        imag_giro.save(os.getcwd()+'/Data/Data_generated_0/'+picture.split('.')[0]+'_mar_'+str(giro)+'_'+str(barco[0] + 1)+'.png')



