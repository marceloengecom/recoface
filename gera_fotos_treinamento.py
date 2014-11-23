# -*- coding: UTF-8 -*-

"""
#####################################################################################
# gera_fotos_treinamento.py
#
# Este arquivo permite gerar 100 fotos frontais dos usuários para treinamento.
# O tempo é de 15 segundos para a geração dessas 100 imagens.
#
# As imagens são em escala de cinza e equalizadas por histograma
# Tamanho das fotos: 100x100 pixels
# Formato: jpg
# Pasta onde serão salvas as imagens: <imagens_treinamento_usuarios>
# Uso: python gera_fotos_treinamento.py <nome do usuário>
#
#####################################################################################
"""

import cv2, sys, numpy, os
size = 2
fn_haar = 'classificadores/haarcascades/haarcascade_frontalface_default.xml'
fn_dir = 'imagens_treinamento_usuarios'
fn_name = sys.argv[1]
path = os.path.join(fn_dir, fn_name)
if not os.path.isdir(path):
    os.mkdir(path)
(im_width, im_height) = (100, 100)
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)
key = cv2.waitKey(250)
count = 0
while count < 50:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 2, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #histo = cv2.equalizeHist(gray, gray)
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    histo = cl1
    #bilateral = cv2.bilateralFilter(cl1,0,20.0,2.0)
    mini = cv2.resize(histo, (histo.shape[1] / size, histo.shape[0] / size))
    faces = haar_cascade.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        face = histo[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
               if n[0]!='.' ]+[0])[-1] + 1
        cv2.imwrite('%s/%s.jpg' % (path, pin), face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
            1,(0, 255, 0))
        count += 1
    cv2.imshow('OpenCV - Gerando fotos...', im)
    key = cv2.waitKey(250)
    if key == 100:
        break
