recoface
========
BIOMETRIA PESSOAL UTILIZANDO TÉCNICAS PARA A DETECÇÃO E RECONHECIMENTO FACIAL
Trabalho de Conclusão do Curso de Engenharia de Computação

Recomenda-se a leitura da documentação, localizada na pasta "docs".



Este trabalho tem como referência, os trabalho de:
Philipp Wagner: http://docs.opencv.org/trunk/modules/contrib/doc/facerec/tutorial/facerec_video_recognition.html
Pierre Raufast: http://thinkrpi.wordpress.com/2013/06/15/opencvpi-cam-step-7-face-recognition/
Shervin Emami: http://www.shervinemami.info/faceRecognition.html


Operações básicas do algoritmo:
- Leitura do arquivo de configuração com as imagens para treinamento
- Inicializa alguns parametros e cria o modelo de Eigenfaces
- Treina o modelo de Eigenfaces com as respectivas imagens para treinamento
- Inicializa a webcam (ID=0) em loop infinito
- Captura imagem
- Faz o devido prée-processamento nos frames de entrada
- Detecta face e para cada face, tenta reconhecê-la (identity) a partir do treinamento feito anteriormente
- Faz a comparação baseado na distância euclidiana
- Reconstroi a imagem a partir do subsespaço gerado pelas faces treinadas.
- Calcula a similaridade entra a face de entrada e a face reconstruida
- Se detectado com sucesso ((similarity < UNKNOWN_PERSON_THRESHOLD) && (confidence > threshold_confidence) && (identity == identity_user))
	- Coloca o respectivo nome na tela, de acordo com uma das variaveis constante
	- Escreve em um arquivo texto o nome do usuário reconhecido e a respectiva data e horário
	- Utilizando voz sintetizada (software espeak), informa uma mensagem ao usuário
	- Tira uma foto da pessoa que está a frente da câmera.


Testado em Ubuntu 14.04 e OpenCV 2.4.9

