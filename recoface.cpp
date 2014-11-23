/** To compile: g++ -o recoface recoface.cpp `pkg-config opencv --libs --cflags` */
/** Use: ./recoface */
/** Necessary OpenCV 2.4.9 to run - www.opencv.org */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// recoface.cpp
//
// Este arquivo tem como referência, os trabalho de:
// 
// Philipp Wagner: http://docs.opencv.org/trunk/modules/contrib/doc/facerec/tutorial/facerec_video_recognition.html
// Pierre Raufast: http://thinkrpi.wordpress.com/2013/06/15/opencvpi-cam-step-7-face-recognition/
// Shervin Emami: http://www.shervinemami.info/faceRecognition.html
//
// Operações básicas do algoritmo:
//	- Leitura do arquivo de configuração com as imagens para treinamento
//	- Inicializa alguns parametros e cria o modelo de Eigenfaces
// 	- Treina o modelo de Eigenfaces com as respectivas imagens para treinamento
// 	- Inicializa a webcam (ID=0) em loop infinito
// 	- Captura imagem
//	- Faz o devido pre-processamento nos frames de entrada
//	- Detecta face e para cada face, tenta reconhecê-la (identity) a partir do treinamento feito anteriormente
//	- Faz a comparação baseado na distância euclidiana
//	- Reconstroi a imagem a partir do subsespaço gerado pelas faces treinadas.
//	- Calcula a similaridade entra a face de entrada e a face reconstruida
//	- Se detectado com sucesso ((similarity < UNKNOWN_PERSON_THRESHOLD) && (confidence > threshold_confidence) && (identity == identity_user))
//		- Coloca o respectivo nome na tela, de acordo com uma das variaveis constante
//		- Escreve em um arquivo texto o nome do usuário reconhecido e a respectiva data e horário
//		- Utilizando voz sintetizada (software espeak), informa uma mensagem ao usuário
//		- Tira uma foto da pessoa que está a frente da câmera.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "time.h"
#include <unistd.h>

#include <dirent.h>
#include <cstdlib>
#include <string>


//if false, use EUCLIDIAN_DISTANCE
//#define USE_MAHALANOBIS_DISTANCE   true


// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"

//////////////////////////////////////////////////////////////////////////////////////
// preprocessFace.cpp, by Shervin Emami (www.shervinemami.info) on 30th May 2012.
// Easily preprocess face images, for face recognition.
//////////////////////////////////////////////////////////////////////////////////////

const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;         // Should be atleast 0.5
const double FACE_ELLIPSE_H = 0.80;         // Controls how tall the face mask is.



using namespace cv;
using namespace std;


////////////////////////////////////////////////////////////////////////////////////////////////////
//
// CONSTANTS AND VARIABLES
// 
////////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined VK_ESCAPE
#define VK_ESCAPE 0x1B      // exit character (27)
#endif


// Sets how confident the Face Verification algorithm should be to decide if it is an unknown person or a known person.
// A value roughly around 0.5 seems OK for Eigenfaces or 0.7 for Fisherfaces, but you may want to adjust it for your
// conditions, and if you use a different Face Recognition algorithm.
// Note that a higher threshold value means accepting more faces as known people,
// whereas lower values mean more faces will be classified as "unknown".
const float UNKNOWN_PERSON_THRESHOLD = 0.46f;

	
// Haarcascades classifiers
string class_glasses = "classificadores/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
string class_eyes_left = "classificadores/haarcascades/haarcascade_lefteye_2splits.xml";
string class_eyes_right = "classificadores/haarcascades/haarcascade_righteye_2splits";
string class_eyes = "classificadores/haarcascades/haarcascade_eye.xml";
// frontal classifier
string class_frontal = "classificadores/haarcascades/haarcascade_frontalface_alt.xml";


string fn_csv;

// Initialize matrices
Mat gray, histo, frame, original, face, face_resized, hsv, bw, reconstructedFace;


Ptr<FaceRecognizer> model;
vector<Mat> testFaces;
vector<int> testLabels;

CascadeClassifier faceCascade;
CascadeClassifier eyeCascade1;
CascadeClassifier eyeCascade2;
CascadeClassifier eyes_cascade;
CascadeClassifier glasses_cascade;


// Some constants (ID) to manage the number of people for training
//
#define MAX_PEOPLE 		11
#define P_USER1			0
#define P_USER2			1
#define P_USER3			2
#define P_USER4			3
#define P_ANONIMOS		10

// name of people
string  people[MAX_PEOPLE];

// number of times talks
int speak[MAX_PEOPLE];

// numer of picture to learn by people
int nPictureById[MAX_PEOPLE];


////////////////////////////////////////////////////////////////////////////////////////////////////
//
// FUNCTIONS
// 
////////////////////////////////////////////////////////////////////////////////////////////////////


// C++ conversion functions between integers (or floats) to std::string.
//
template <typename T> string toString(T t){
    ostringstream out;
    out << t;
    return out.str();
}

template <typename T> T fromString(string t){
    T out;
    istringstream in(t);
    in >> out;
    return out;
}


// Convert the matrix row or column (float matrix) to a rectangular 8-bit image that can be displayed or saved.
// Scales the values to be between 0 to 255.
//
Mat getImageFrom1DFloatMat(const Mat matrixRow, int im_height){
	// Make a rectangular shaped image instead of a single row.
	Mat rectangularMat = matrixRow.reshape(1, im_height);
	// Scale the values to be between 0 to 255 and store them
	// as a regular 8-bit uchar image.
	Mat dst;
	normalize(rectangularMat, dst, 0, 255, NORM_MINMAX,CV_8UC1);
	return dst;
}


// Check time.
double old_time = 0;
double current_time = (double)getTickCount();
double timeDiff_seconds = (current_time - old_time)/getTickFrequency();


// Generate an approximately reconstructed face by back-projecting the eigenvectors & eigenvalues of the given (preprocessed) face.
// Parameters: model, face_resized.
//
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat face_resized){

	// Since we can only reconstruct the face for some types of FaceRecognizer models (ex: Eigenfaces or Fisherfaces),
	// we should surround the OpenCV calls by a try/catch block so we don't crash for other models.
	try {
	//string result_message = format("Classe de Previsao= %d / Classe Atual = %d.", identity, faceLabels);
	
	 // Here is how to get the eigenvalues of this Eigenfaces model: 
   	Mat eigenvalues = model->getMat("eigenvalues");
	//cout<<"EigenValores - Autovalores= "<<eigenvalues<<"\n";
	
	// Get some required data from the FaceRecognizer model.	

	// Convariance matrix which contains eigenvectors
	Mat eigenvectors = model->get<Mat>("eigenvectors");
	//cout<<"EigenVetores - Autovetores= "<<eigenvectors<<"\n";

	Mat averageFaceRow = model->get<Mat>("mean");
	//cout<<"Eigenface Media= "<<averageFaceRow<<"\n";
	
 	// int im_width = images[0].cols;
   	int faceHeight = face_resized.rows;

	// Show the best 20 eigenfaces
	for (int i = 0; i < min(20, eigenvectors.cols); i++) {
		// Create a column vector from eigenvector #i.		
		// Note that the FaceRecognizer class already gives us L2 normalized eigenvectors, so we don't have to normalize them ourselves.
		Mat eigenvectorColumn = eigenvectors.col(i).clone();
		//cout << "eigenvector: "<<eigenvectorColumn<< endl;	

		Mat eigenface = getImageFrom1DFloatMat(eigenvectorColumn, faceHeight);
		//cout << "eigenface: "<<eigenface<< endl;		
		//imshow(format("Eigenface%d", i), eigenface);
        }       


   		
	// Project the input image onto the PCA subspace.
	Mat projection = subspaceProject(eigenvectors, averageFaceRow, face_resized.reshape(1,1));
	//cout << "projection: "<<projection<< endl;
	
	//Generate the reconstructed face back from the PCA subspace.
	Mat reconstructionRow = subspaceReconstruct(eigenvectors, averageFaceRow, projection);
	
	// Convert the float row matrix to a regular 8-bit image. Note that we shouldn't use "getImageFrom1DFloatMat()"
	// because we don't want to normalize the data since it is already at the perfect scale.
	// Make it a rectangular shaped image instead of a single row.
	Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
	
	// Convert the floating-point pixels to regular 8-bit uchar pixels.
	reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
	reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
	//imshow("Imagem Reconstruida", reconstructedFace);


	return reconstructedFace;

	} catch (cv::Exception e) {
	cout << "Aviso: Problema na classe 'reconstructFace()'." << endl;
	return Mat();
	}
}


// Compare two images by getting the L2 error (square-root of sum of squared error).
//
double getSimilarity(const Mat A, const Mat B) {
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        // Calculate the L2 relative error between the 2 images.
        double errorL2 = norm(A, B, CV_L2);
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else {
        cout << "AVISO: Imagens tem duferentes tamanhos em 'getSimilarity()'." << endl;
        return 100000000.0;  // Returno invalid value
    }
}



// Reading the images CSV.
//
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "(E) No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    int nLine=0;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) 
        {
        	// read the file and build the picture collection
		images.push_back(imread(path, 0));
		labels.push_back(atoi(classlabel.c_str()));
		nPictureById[atoi(classlabel.c_str())]++;
        	nLine++;
        }
    }
    
	// write number of picture by people
	char sTmp[128];
	sprintf(sTmp,"(init) %d pictures read to train",nLine);
	cout <<((string)(sTmp))<< endl;
	for (int j=0;j<MAX_PEOPLE;j++){
		sprintf(sTmp,"(init) %d pictures of %s (%d) read to train",nPictureById[j],people[j].c_str(),j);
   	 	cout <<((string)(sTmp))<< endl;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 
// To identify a person registered, the system writes in the report file and
// informs with synthesized speech. Software "espeak".
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void sintetizador(int nPerson){
	FILE *arq = NULL;
	char key;
	

//Date functions using time.h
	struct tm *local;
	time_t t;
	t= time(NULL);
	local=localtime(&t);

	int dia=local->tm_mday;
	int mes=local->tm_mon+1;
	int ano=local->tm_year+1900;

	char buffer[64];


	if (speak[nPerson]==0)
	{
		
		// Open temporary file to data storage access
		arq= fopen("relatorio_acesso.txt","a");
		if (arq==NULL)
		{
			cout<<"(E) Incapaz de escrever no arquivo\n";
			return;
		}
		else
		{
				
			// Say welcome, store acess data in file and take a picture
			if (nPerson==P_USER1)
			{
				sleep (2);				
				fprintf(arq,"Nome: Usuário 1\n");
				fprintf(arq,"Data: %.2d:%.2d  -  %.2d/%.2d/%.2d\n",local->tm_hour,local->tm_min, dia,mes,ano);
				system("espeak -vpt+f5 -s140 'Bemvindo Usuário 1!'");
				sprintf(buffer,"imagens_usuarios/%.2d:%.2d_%.2d-%.2d-%.2d-user1.jpg",local->tm_hour,local->tm_min,dia,mes,ano);
				imwrite( buffer, gray);
				fprintf(arq,"Foto salva: %.2d:%.2d_%.2d-%.2d-%.2d-user1.jpg\n\n",local->tm_hour,local->tm_min,dia,mes,ano);
				waitKey(1000);
				destroyWindow("Reconhecimento-Facial");
            			//exit(0);				

			}
			if (nPerson==P_USER2)
			{
				sleep (2);				
				fprintf(arq,"Nome: Usuário 2\n");
				fprintf(arq,"Data: %.2d:%.2d  -  %.2d/%.2d/%.2d\n",local->tm_hour,local->tm_min, dia,mes,ano);
				system("espeak -vpt+f5 -s140 'Bemvindo Usuário 2!'");
			sprintf(buffer,"imagens_usuarios/%.2d:%.2d_%.2d-%.2d-%.2d-user2.jpg",local->tm_hour,local->tm_min,dia,mes,ano);
				imwrite( buffer, gray);
			fprintf(arq,"Foto salva: %.2d:%.2d_%.2d-%.2d-%.2d-user2.jpg\n\n",local->tm_hour,local->tm_min,dia,mes,ano);
				waitKey(1000);
				destroyWindow("Reconhecimento-Facial");
            			//exit(0);
			}

			if (nPerson==P_USER3)
			{
				sleep (2);				
				fprintf(arq,"Nome: Usuário 3\n");
				fprintf(arq,"Data: %.2d:%.2d  -  %.2d/%.2d/%.2d\n",local->tm_hour,local->tm_min, dia,mes,ano);
				system("espeak -vpt+f5 -s140 'Bemvindo Usuário 3!'");
			sprintf(buffer,"imagens_usuarios/%.2d:%.2d_%.2d-%.2d-%.2d-user3.jpg",local->tm_hour,local->tm_min,dia,mes,ano);
				imwrite( buffer, gray);
			fprintf(arq,"Foto salva: %.2d:%.2d_%.2d-%.2d-%.2d-user3.jpg\n\n",local->tm_hour,local->tm_min,dia,mes,ano);
				waitKey(1000);
				destroyWindow("Reconhecimento-Facial");
            			//exit(0);
			}

			if (nPerson==P_USER4)
			{
				sleep (2);				
				fprintf(arq,"Nome: Usuário 4\n");
				fprintf(arq,"Data: %.2d:%.2d  -  %.2d/%.2d/%.2d\n",local->tm_hour,local->tm_min, dia,mes,ano);
				system("espeak -vpt+f5 -s140 'Bemvindo Usuário 4!'");
			sprintf(buffer,"imagens_usuarios/%.2d:%.2d_%.2d-%.2d-%.2d-user4.jpg",local->tm_hour,local->tm_min,dia,mes,ano);
				imwrite( buffer, gray);
			fprintf(arq,"Foto salva: %.2d:%.2d_%.2d-%.2d-%.2d-user4.jpg\n\n",local->tm_hour,local->tm_min,dia,mes,ano);
				waitKey(1000);
				destroyWindow("Reconhecimento-Facial");
            			//exit(0);
			}

			if (nPerson==P_ANONIMOS)
			{
				sleep (2);				
				fprintf(arq,"Nome: NÃO IDENTIFICADO\n");
				fprintf(arq,"Data de acesso: Sem Acesso");

			}
			
			// Close file
			fclose(arq);
			
			// espeak
			// -vpt = Voice in portuguese
			// +f5 : Fifth female voice
			// -s140 : Voice speed. Default is 160.
			// 2>/dev/null: If espeak generate error, send to /dev/null
			
			
		}
	}
	
	// increment
	speak[nPerson]++;	
	
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// MAIN
//
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {

	int n;
	double threshold_confidence;
	int identity_user;

	cout << "DIGITE O SEU NUMERO DE USUARIO: \n" << endl;
	scanf("%d",&n);
	cout << "\nO valor digitado foi: " << n << endl;

	if (n == 0){
		fn_csv = "banco_de_faces/user1-banco_de_faces.csv";
		waitKey(5000);
		cout <<"Analisando faces de Usuário 1"<< endl;
		
		threshold_confidence = 1500.0;
		cout <<"Valor minimo de Threshold, baseado na distancia euclidiana de 1-vizinho mais próximo, definido em: "<< threshold_confidence << endl;

		identity_user = 0;
		cout <<"Identificação do usuario, baseado no arquivo CSV: " << identity_user << endl;
	}
	if (n == 1){
		fn_csv = "banco_de_faces/user2-banco_de_faces.csv";
		waitKey(5000);
		cout <<"Analisando faces de Usuário 2"<< endl;
		
		threshold_confidence = 3000.0;
		cout <<"Valor minimo de Threshold, baseado na distancia euclidiana de 1-vizinho mais próximo, definido em: "<< threshold_confidence << endl;
		
		identity_user = 1;
		cout <<"Identificação do usuario, baseado no arquivo CSV: " << identity_user << endl;

	}
	if (n == 2){
		fn_csv = "banco_de_faces/user3-banco_de_faces.csv";
		waitKey(5000);
		cout <<"Analisando faces de Usuário 3"<< endl;
		
		threshold_confidence = 3000.0;
		cout <<"Valor minimo de Threshold, baseado na distancia euclidiana de 1-vizinho mais próximo, definido em: "<< threshold_confidence << endl;
		
		identity_user = 2;
		cout <<"Identificação do usuario, baseado no arquivo CSV: " << identity_user << endl;

	}

	if (n == 3){
		fn_csv = "banco_de_faces/user4-banco_de_faces.csv";
		waitKey(5000);
		cout <<"Analisando faces de Usuário 4"<< endl;
		
		threshold_confidence = 1000.0;
		cout <<"Valor minimo de Threshold, baseado na distancia euclidiana de 1-vizinho mais próximo, definido em: "<< threshold_confidence << endl;
		
		identity_user = 3;
		cout <<"Identificação do usuario, baseado no arquivo CSV: " << identity_user << endl;

	}

	// Load Haarcascades classifiers to eyes detection.
	
	CascadeClassifier glasses_cascade;
	    if (!glasses_cascade.load(class_glasses))
		{
    			cout <<"(E) Classificador de oculos nao carregado :"+class_glasses+"\n"; 
    			return -1;
		}
	cout << "Classificador Glasses carregado" << endl;

	CascadeClassifier eyes_cascade;
	    if (!eyes_cascade.load(class_eyes))
		{
    			cout <<"(E) Classificador de olhos nao carregado :"+class_eyes+"\n"; 
    			return -1;
		}
	cout << "Classificador Olhos carregado" << endl;

	CascadeClassifier eyes_left_cascade;
	    if (!eyes_left_cascade.load(class_eyes_left))
		{
    			cout <<"(E) Classificador de olho esquerdo nao carregado :"+class_eyes_left+"\n"; 
    			return -1;
		}
	cout << "Classificador Olho Esquerdo carregado" << endl;

	CascadeClassifier eyes_right_cascade;
	    if (!eyes_right_cascade.load(class_eyes_left))
		{
    			cout <<"(E) Classificador de olho direito nao carregado :"+class_eyes_right+"\n"; 
    			return -1;
		}
	cout << "Classificador Olho Direito carregado" << endl;

	
	// Load Haarcascades classifiers to face.
	CascadeClassifier face_cascade;
	    if (!face_cascade.load(class_frontal))
		{
    			cout <<"(E) Classificador de face nao carregado :"+class_frontal+"\n"; 
    			return -1;
		}
	cout << "Classificador de face carregado com sucesso\n";
	

	// init people, should be do in a config file.
	people[P_USER1] 	= "Usuário 1";
	people[P_USER2] 	= "Usuário 2";
	people[P_USER3] 	= "Usuário 3";
	people[P_USER4] 	= "Usuário 4";
	people[P_ANONIMOS] 	= "Nao Identificado";
	
	// init...
	// reset counter.
	for (int i=0;i < MAX_PEOPLE;i++) 
	{
		speak[i] =0;
		nPictureById[i]=0;
	}
	int bFirstDisplay = 1;
	cout << "Pessoas inicializadas" << endl;


    
	vector<Mat> images;
	vector<int> labels;

    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
	
     
    try {
        read_csv(fn_csv, images, labels);
		cout<<"(OK) Leitura base de dados CSV concluida\n";
    	} 
    catch (cv::Exception& e) 
    {
        cerr << "Erro ao abrir o arquivo \"" << fn_csv << "\". Razao: " << e.msg << endl;
        exit(1);
    }

	// Get height and width.  All images need a same size.
   	int im_width = images[0].cols;
   	int im_height = images[0].rows;
	cout << "Leitura de imagens da base de dados concluida\n";


	// The following lines simply get the last images from your dataset and remove it from the vector.
	// This is done, so that the training data (which we learn the cv::FaceRecognizer on) and the
	// test data we test the model with, do not overlap.
	Mat testSample = images[images.size() - 1];
    	int testLabel = labels[labels.size() - 1];
    	images.pop_back();
    	labels.pop_back();

	
	//int ncomponents = 80;
  	//double threshold = DBL_MAX;

	//Ptr<FaceRecognizer> model =  createFisherFaceRecognizer();

	// this a Eigen model, but you could replace with Fisher model (in this case threshold value should be lower)
	//
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();

	// And this line sets the threshold to 0.0:
	//model->set("threshold", 2000.0);

	model->train(images, labels);
	cout << "Treinamento realizado com " << images.size() << " imagens\n";
    	
	model->save("eigenfaces.yml"); // save the model to eigenfaces_at.yaml
	cout << "Modelo de treinamento salvo\n";

   	model->load("eigenfaces.yml"); // load the model
	cout << "Modelo carregado\n";


	// Get the eigenvectors
        // Mat eigenvectors = model->get<Mat>("eigenvectors");
	//cout<<"eigenvectors: "<<eigenvectors<< endl;
	
	// Get the eigenvalues
        // Mat eigenvalues = model->get<Mat>("eigenvalues");
	//cout<<"eigenvalues: "<<eigenvalues<< endl;

       

        int ncomponents = model->get<int>("ncomponents");
        cout << "ncomponents = " << ncomponents << endl;
	
	// And this line sets the ncomponents to 80:
	//model->set("ncomponents", 80);
	//cout << "ncomponents2 = " << ncomponents << endl;

	// The following line reads the threshold from the Eigenfaces model:
	//double current_threshold = model->getDouble("threshold");
	//cout << "current_threshold1: "<<current_threshold<< endl;

	


	//identity  will be the label number that we originally used when collecting faces for training.
	//For example, 0 for the first person, 1 for the second person, and so on.
	// The following line predicts the label of a given test image:
	int predictedLabel = model->predict(testSample);

	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;

// capture video
/*
    VideoCapture capture(0); // open the default camera
    if(!capture.isOpened()){  // check if we succeeded
	cout << "Camera Fechada"<< endl;        
	return -1;
	}   
    //namedWindow("webcam",1);
*/

    CvCapture* capture;
 	capture = cvCaptureFromCAM(0);
 	
 	// set size of webcam 640x480
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,640);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);
	cout << "(E) WebCAM inicializada." << endl;
	
	// can't capture, doc'
	if (!capture)
	{   
		cout << "(E) Capture Device cannot be opened." << endl;
        return -1;
    }

    for(;;){
        Mat frame;
        //capture >> frame; // get a new frame from camera

	// get the picture from webcam
		frame= cvQueryFrame( capture);
		char key; 


	// Use faces.
	int desiredFaceWidth = im_width;
    	int desiredFaceHeight = im_height;

	// Convert the current frame to grayscale:
        cvtColor(frame, gray, CV_RGB2GRAY);

//	normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);
//	imshow("normalize", face_resized);

	double angle = 0.0;
	double scale = 1.0;
	Mat rot_mat = getRotationMatrix2D(Point2f(0,0), angle, scale);

	// Rotate and scale and translate the image to the desired angle & size & position!
	// Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
        Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); // Clear the output image to a default grey.
        warpAffine(gray, warped, rot_mat, warped.size());
        //imshow("warped", warped);


	//equalizeHist(warped, histo);
        
        // Apply Histogram Equalization
	//equalizeHist(gray, histo);

	//Contrast Limited Adaptive Histogram Equalization
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(3);

	Mat dstClahe;
	clahe->apply(gray,dstClahe);
	imshow("dstClahe",dstClahe);

	
        vector< Rect_<int> > faces;

	// Detec faces in video
	//cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);
	face_cascade.detectMultiScale(dstClahe, faces, 1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(24, 24));	

		// To each face founded
		int i;
		for (i = 0; i < faces.size(); i++ ){

			// Process face by face:
            		Rect face_i = faces[i];
			// Crop the face from the image
		        face = dstClahe(face_i);

			// Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
			// CV_8U is unsigned 8bit/pixel. Eg: A pixel can have values 0-255, this is the normal range for most formats.
			Mat filtered = Mat(warped.size(), CV_8U);
        		bilateralFilter(face, filtered, 0, 20.0, 2.0);
			//resize(filtered, filtered, Size(im_width, im_height), 1.0, 1.0);
			imshow("filtered", filtered);	
			
			
			// Filter out the corners of the face, since we mainly just care about the middle parts.
		        // Draw a filled ellipse in the middle of the face-sized image.
		        //Mat mask = Mat(warped.size(), CV_8U, Scalar(0)); // Start with an empty mask.
            		//Point faceCenter = Point(desiredFaceWidth/2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY) );
            		//Size size = Size( cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H) );
            		//ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
            		//imshow("mask", mask);

			// Use the mask, to remove outside pixels.
			//Mat dstImg = Mat(warped.size(), CV_8U, Scalar(255)); // Clear the output image to a default gray.
			/*
			namedWindow("filtered");
           		imshow("filtered", filtered);

            		namedWindow("dstImg");
            		imshow("dstImg", dstImg);
            		namedWindow("mask");
          		imshow("mask", mask);
		        */
            		// Apply the elliptical mask on the face.
            		//filtered.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
			//imshow("dstImg", dstImg);
		
			face = filtered;

		vector< Rect_<int> > eyes;
/*		
		//--To each face, detect eyes
       	 	//eyes_cascade.detectMultiScale( face, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
       	 	eyes_left_cascade.detectMultiScale( face, eyes, 1.1, 4, 0 |CASCADE_SCALE_IMAGE, Size(24, 24) );
       	 	eyes_right_cascade.detectMultiScale( face, eyes, 1.1, 4, 0 |CASCADE_SCALE_IMAGE, Size(24, 24) );

		for ( size_t j = 0; j < eyes.size(); j++ )
			{
		            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
		            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.10 );
		            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 1, 1, 0 );
		        }
*/
	// create a rectangle around the face      
	rectangle(frame, face_i, CV_RGB(60, 220 , 100), 5);
            
	// Resize and show
	//cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
	resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0);
	namedWindow("FACE", CV_WINDOW_AUTOSIZE);
	imshow("FACE",face_resized);
/*
			char buffer[64];
			for (int i = 1; i < 101; i++ ){
				sprintf(buffer,"%d.jpg",i);
				imwrite(buffer, face_resized);
			}
*/
	char sTmp[256];


	
	// According to "Shervin Emami's" Tutorial about face recognition using Eigenfaces,
	// the confidence level is derived based on the opposite of Euclidean distance
	// confidence = 1.0f - sqrt( distSq / (float)(nTrainFaces * nEigens) ) / 255.0f


	double confidence = 0.0;

	int identity = -1;
	
	// Get the prediction (identity) and associated confidence from the model
	model->predict(face_resized,identity,confidence);
	
	// Show the actual confidence	
	cout << "confidence: "<<confidence<< endl;

	// Now perform the prediction
	//identity = model->predict(face_resized);
	//cout << "identity: "<<identity<<"\n";


	
	// Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
        Mat reconstructedFace;
	reconstructedFace = reconstructFace(model, face_resized);
	imshow("Imagem Reconstruida", reconstructedFace);

		

	// Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person.
	double similarity = getSimilarity(face_resized,reconstructedFace);
	cout << "SIMILARIDADE: "<<similarity<< endl;
	
	// Crop the confidence rating between 0.0 to 1.0, to show in the bar.
	//confidence = 1.0 - min(max(similarity, 0.0), 1.0);	

				
		// 
		if ((similarity < UNKNOWN_PERSON_THRESHOLD) && (confidence > threshold_confidence) && (identity == identity_user)) {

			identity = model->predict(face_resized);
			
			// Person found
			cout << "Nome da Pessoa: " << people[identity].c_str()<< endl;
			cout << "Identidade: " << identity << ". Similaridade: " << similarity << endl;
			cout << "Confiança: "<<(int)confidence<<"\n";		
			 	
			// Show the name of person found
			string box_text;
			if (identity<MAX_PEOPLE){
					box_text = "NOME="+people[identity];
			}
			else {
					cout << "(E) ID da previsão incoerente\n";
			}
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);			   
			putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
				
			/// Show the result:
			namedWindow("Reconhecimento-Facial", CV_WINDOW_AUTOSIZE);
			imshow("Reconhecimento-Facial",frame);
			waitKey(27);

			// Actie sintetizador class
			sintetizador(identity);
			}

			else {
			
			identity = -1;

			// Since the confidence is low, assume it is an unknown person.
			//
			cout << "NAO CADASTRADO" << endl;
	                cout << "Identidade: " << identity << ". Similaridade: " << similarity << endl;
			cout << "confidence: "<<(int)confidence<<"\n";
			//sprintf(sTmp,"- Previsão muito baixa = %s (%d) Confiança = (%d)",people[identity].c_str(),identity,(int)confidence);

				string box_text;
				box_text = "NAO CADASTRADO ou NUMERO INCOMPATIVEL";
				
				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);			   
				putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
				} 
			}
	
			// Show the result
			// notice we display twice the picture, first time before .predict, one after. 
			// to double display freq.
			namedWindow("Reconhecimento-Facial", CV_WINDOW_AUTOSIZE);
			imshow("Reconhecimento-Facial",frame);

        		// IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
        		// Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
        		char keypress = waitKey(20);  // This is needed if you want to see anything!
        		if (keypress == VK_ESCAPE) {   // Escape Key
            			// Quit the program!
            			break;
			}

		}
	return 0;
}
