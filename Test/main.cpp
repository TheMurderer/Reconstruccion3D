#include <iostream>
#include <vector>
#include<opencv2/opencv.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include <fstream>
using namespace cv;
using namespace std;

#define _SURF_ 1
//#define _SIFT_ 1

// Código de ayuda para el uso de los detectores SURF o SIFT
void surf(Mat& img1, vector<KeyPoint>& keypoints){
        #if  _SURF_
        int minHessian = 5000; // valor de unbral para la deteccion de puntos, cambiar si es necesario
        SurfFeatureDetector detector( minHessian );
        detector.detect(img1, keypoints);
        #else if _SIFT_
        double sigma=1;
        int puntos=1000;
        SiftFeatureDetector detector(puntos,3,0.06,10,sigma);
        detector.detect(img1, keypoints);
        #endif
}

void computeMatching(Mat& img1, Mat& img2,vector<KeyPoint>& keypoints1,vector<KeyPoint>& keypoints2, vector<DMatch>& matches ){
        // computing descriptors
        #if _SURF_
        SurfDescriptorExtractor extractor;
        #else if _SIFT_
        SiftDescriptorExtractor extractor;
        #endif
        Mat descriptors1, descriptors2;
        extractor.compute(img1, keypoints1, descriptors1);
        extractor.compute(img2, keypoints2, descriptors2);

        // matching descriptors
        bool crossCheck=1;

        //BFMatcher match;
        BFMatcher matcher(NORM_L2,crossCheck);
        matcher.match(descriptors1, descriptors2, matches);
}

void imprimirKeipoints(vector<KeyPoint> puntos){
    for(unsigned int i = 0; i < puntos.size(); i++){
        cout << "("  << puntos[i].pt.x << " , " << puntos[i].pt.y << ")" << endl;
    }
}



int main(int argc, char* argv[]) {

        /*if (argc != 3){
            cerr << "Error: ./ejecutable [imagen1] [imagen2]" << endl;
            return -1;
        }*/

        Mat img1 = imread("Imagenes/img1.bmp",CV_LOAD_IMAGE_GRAYSCALE);
        Mat img2 = imread("Imagenes/img2.bmp",CV_LOAD_IMAGE_GRAYSCALE);

        //Aplicacmos Canny  -- Detección de bordes
        Mat img1Canny,img2Canny;
        Canny(img1,img1Canny,50,200,3);
        Canny(img1,img2Canny,50,200,3);

        // Obtenemos los puntos en correspondencias de forma automática usando SURF
        vector<KeyPoint> keypoints1, keypoints2;
        vector<DMatch> matches;

        surf(img1Canny, keypoints1);
        surf(img2Canny, keypoints2);

        //Señalamos los puntos encontrados sobre la imangen

        //Mat salidaImg1,salidaImg2;
        //drawKeypoints(img1,keypoints1,salidaImg1);
        //drawKeypoints(img2,keypoints2,salidaImg2);

        //Hacemos el matching de los puntos encontrados en las imagenes
        computeMatching(img1Canny, img2Canny, keypoints1, keypoints2, matches);

        Mat matchImg;
        drawMatches(img1Canny,keypoints1,img2Canny,keypoints2,matches,matchImg);


        imshow("Match img1 | img2",matchImg);


        //vector <Point2f> puntosImagen1, puntosImagen2;

        //for(int j=0; j<matches.size(); j++){
        //        puntosImagen1.push_back(Point2f(keypoints1[matches[j].queryIdx].pt));
        //        puntosImagen2.push_back(Point2f(keypoints2[matches[j].trainIdx].pt));
        //}
/*
        // Calculamos la matriz fundamental por el algoritmo 8-puntos y RANSAC.
        Mat fundamentalMat = Mat(3,3,CV_32F);
        Mat F2 = findFundamentalMat(puntosImagen1, puntosImagen2, fundamentalMat, CV_FM_8POINT|CV_FM_RANSAC, 1.0, 0.99);
        cout << "Matriz Fundamental: " << endl << F2 << endl;

        // Pintamos las líneas epipolares resultantes de la matriz fundamental.
        vector<Vec3f> lineasImg1, lineasImg2;
        computeCorrespondEpilines(Mat(puntosImagen1), 1, F2, lineasImg1);
        for (vector<Vec3f>::const_iterator it= lineasImg1.begin(); it!=lineasImg1.end(); ++it) {
           line(img1, Point2d(0, -(*it)[2]/(*it)[1]), Point2d(img1.cols, -((*it)[2]+(*it)[0]*img1.cols)/(*it)[1]), Scalar(255,255,255));

        }

        computeCorrespondEpilines(Mat(puntosImagen2), 2, F2, lineasImg2);
        for (vector<cv::Vec3f>::const_iterator it= lineasImg2.begin(); it!=lineasImg2.end(); ++it) {
           line(img2, Point2d(0, -(*it)[2]/(*it)[1]), Point2d(img2.cols, -((*it)[2]+(*it)[0]*img2.cols)/(*it)[1]), Scalar(255,255,255));
        }

        // A partir de la matriz fundamental, calculamos los epipolos.
        Mat transpuesta;
        Mat epipoloIzq;
        Mat epipoloDer;
        transpose(F2, transpuesta);
        Mat simetrica = transpuesta * F2;
        Mat simetrica2 = F2 * transpuesta;

        eigen(simetrica, epipoloIzq);
        eigen(simetrica2, epipoloDer);

        cout << "Epipolo Izquierdo : " << epipoloIzq << endl;
        cout << "Epipolo Derecho : " << epipoloDer << endl;
*/
        //imshow("imagen1", img1);
        //imshow("imagen2", img2);
        waitKey();
        //destroyWindow("imagen1");
        //destroyWindow("imagen2");
        return 1;
}
