#include <iostream>
#include <vector>
#include<opencv2/opencv.hpp>
#include<opencv2/nonfree/features2d.hpp>

#include <opencv/cxcore.h>
#include <opencv/cvaux.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include <fstream>
using namespace cv;
using namespace std;

#define _SURF_ 1
//#define _SIFT_ 1

//Variables globales
int thresh = 150;
int max_thresh = 255;
/*
//Harris - detector de esquinas
void harris(Mat src,Mat &src_gray, vector<KeyPoint>& keypoints){
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( src.size(), CV_32FC1 );

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

    /// Normalizing
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    /// Drawing a circle around corners
    for( int j = 0; j < dst_norm.rows ; j++ )
       { for( int i = 0; i < dst_norm.cols; i++ )
            {
              if( (int) dst_norm.at<float>(j,i) > thresh )
                {
                 circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
                 Point p = Point( i, j );
                 keypoints.push_back(KeyPoint(p.x,p.y,5));
                }
            }
       }
    /// Showing the result
    //namedWindow( corners_window, CV_WINDOW_AUTOSIZE );
    //imshow("Harris",dst_norm_scaled );
}

// Código de ayuda para el uso de los detectores SURF o SIFT
void surf(Mat& img1, vector<KeyPoint>& keypoints){
        #if  _SURF_
        int minHessian = 1000; // valor de unbral para la deteccion de puntos, cambiar si es necesario
        SurfFeatureDetector detector( minHessian );
        detector.detect(img1, keypoints);
        #else if _SIFT_
        double sigma=1;
        int puntos=10000;
        SiftFeatureDetector detector(puntos,3,0.06,10,sigma);
        detector.detect(img1, keypoints);
        #endif
}
*/
//Visualizador de nube de puntos
boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  viewer->addCoordinateSystem ( 1.0 );
  viewer->initCameraParameters ();
  return (viewer);
}
/*
//Realiza el Matching entre puntos
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

        FlannBasedMatcher matcher;
        matcher.match(descriptors1,descriptors2,matches);

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors1.rows; i++ ){
           double dist = matches[i].distance;
           if( dist < min_dist ) min_dist = dist;
           if( dist > max_dist ) max_dist = dist;
         }

         printf("-- Max dist : %f \n", max_dist );
         printf("-- Min dist : %f \n", min_dist );

         //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
         //-- PS.- radiusMatch can also be used here.
         std::vector< DMatch > good_matches;

         for( int i = 0; i < descriptors1.rows; i++ ){
             if( matches[i].distance < 2*min_dist ){
                 good_matches.push_back( matches[i]);
             }
         }

         //-- Draw only "good" matches
         Mat img_matches;
         drawMatches( img1, keypoints1, img2, keypoints2,
                        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

         //-- Show detected matches
         imshow( "Good Matches", img_matches );
}

//Función que imprime por consola los puntos característicos encontrados
void imprimirKeipoints(vector<KeyPoint> puntos){
    for(unsigned int i = 0; i < puntos.size(); i++){
        cout << "("  << puntos[i].pt.x << " , " << puntos[i].pt.y << ")" << endl;
    }
}

*/

//int main(int argc, char* argv[]) {

//        /*if (argc != 3){
//            cerr << "Error: ./ejecutable [imagen1] [imagen2]" << endl;
//            return -1;
//        }*/

//        Mat img1,img2,img1_gray,img2_gray;
//        img1 = imread("Imagenes/Dino/viff.002.ppm");
//        img2 = imread("Imagenes/Dino/viff.003.ppm");
//        cvtColor( img1, img1_gray, CV_BGR2GRAY );
//        cvtColor( img2, img2_gray, CV_BGR2GRAY );


//        // Obtenemos los puntos en correspondencias de forma automática usando SURF
//        vector<KeyPoint> keypoints1, keypoints2;
//        vector<DMatch> matches;

//        surf(img1, keypoints1);
//        surf(img2, keypoints2);
//        //harris(img1,img1_gray,keypoints1);
//        //harris(img2,img2_gray,keypoints2);

//        //Hacemos el matching de los puntos encontrados en las imagenes
//        computeMatching(img1, img2, keypoints1, keypoints2, matches);


//        vector <Point2f> puntosImagen1, puntosImagen2;

//        for(int j=0; j<matches.size(); j++){
//                puntosImagen1.push_back(Point2f(keypoints1[matches[j].queryIdx].pt));
//                puntosImagen2.push_back(Point2f(keypoints2[matches[j].trainIdx].pt));
//        }

//        // Calculamos la matriz fundamental por el algoritmo 8-puntos y RANSAC.
//        Mat fundamentalMat = Mat(3,3,CV_32F);
//        Mat F2 = findFundamentalMat(puntosImagen1, puntosImagen2, fundamentalMat, CV_FM_8POINT|CV_FM_RANSAC, 1.0, 0.99);
//        cout << "Matriz Fundamental: " << endl << F2 << endl;

////        // Pintamos las líneas epipolares resultantes de la matriz fundamental.
////        vector<Vec3f> lineasImg1, lineasImg2;
////        computeCorrespondEpilines(Mat(puntosImagen1), 1, F2, lineasImg1);
////        for (vector<Vec3f>::const_iterator it= lineasImg1.begin(); it!=lineasImg1.end(); ++it) {
////           line(img1, Point2d(0, -(*it)[2]/(*it)[1]), Point2d(img1.cols, -((*it)[2]+(*it)[0]*img1.cols)/(*it)[1]), Scalar(255,255,255));

////        }

////        computeCorrespondEpilines(Mat(puntosImagen2), 2, F2, lineasImg2);
////        for (vector<cv::Vec3f>::const_iterator it= lineasImg2.begin(); it!=lineasImg2.end(); ++it) {
////           line(img2, Point2d(0, -(*it)[2]/(*it)[1]), Point2d(img2.cols, -((*it)[2]+(*it)[0]*img2.cols)/(*it)[1]), Scalar(255,255,255));
////        }

////        // A partir de la matriz fundamental, calculamos los epipolos.
////        Mat transpuesta;
////        Mat epipoloIzq;
////        Mat epipoloDer;
////        transpose(F2, transpuesta);
////        Mat simetrica = transpuesta * F2;
////        Mat simetrica2 = F2 * transpuesta;

////        eigen(simetrica, epipoloIzq);
////        eigen(simetrica2, epipoloDer);

//        //Calculo de las matrices de disparidad H1, H2
//        Mat H1, H2;
//        stereoRectifyUncalibrated(puntosImagen1,puntosImagen2,F2,img2.size(),H1,H2);

//        // create the image in which we will save our disparities
//        Mat imgDisparity16S = Mat( img1.rows, img1.cols, CV_16S );
//        Mat imgDisparity8U = Mat( img1.rows, img1.cols, CV_8UC1 );

//        int ndisparities = 16*5;      // < Range of disparity >
//        int SADWindowSize = 5;

//        //Calculamos la disparidad a partir de imagenes en escala de grises
//        StereoBM sbm( StereoBM::BASIC_PRESET,ndisparities,SADWindowSize );

//        sbm( img1_gray,img2_gray , imgDisparity16S, CV_16SC1 );

//        double minVal; double maxVal;

//        minMaxLoc( imgDisparity16S, &minVal, &maxVal );

//        printf("Min disp: %f Max value: %f \n", minVal, maxVal);

//        // Display it as a CV_8UC1 image
//        imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));

//        namedWindow( "windowDisparity", CV_WINDOW_NORMAL );
//        imshow( "windowDisparity", imgDisparity8U );

//        waitKey();
//        //Calculamos la nube de puntos

//        //Cargamos la matriz Q
//        //Load Matrix Q
//        FileStorage fs("Q.xml", FileStorage::READ);
//        Mat Q;

//        fs["Q"] >> Q;

//        //If size of Q is not 4x4 exit
//        if (Q.cols != 4 || Q.rows != 4)
//        {
//          std::cerr << "ERROR: Could not read matrix Q (doesn't exist or size is not 4x4)" << std::endl;
//          return 1;
//        }

//        //Mostrar la matriz Q
////        for (int y = 0; y < Q.rows; y++)
////        {
////          const double* Qy = Q.ptr<double>(y);
////          for (int x = 0; x < Q.cols; x++)
////          {
////            std::cout << "Q(" << x << "," << y << ") = " << Qy[x] << std::endl;
////          }
////        }

//        Mat imgDisparity8U = imread("imagenDisparidad.png", CV_LOAD_IMAGE_GRAYSCALE);

//        namedWindow("disparity-image",CV_WINDOW_NORMAL);
//        imshow("disparity-image", imgDisparity8U);
//        waitKey();

//        Mat recons3D(imgDisparity8U.size(), CV_32FC3);

//        //Reproject image to 3D
//        cout << "Reprojecting image to 3D..." << endl;
//        reprojectImageTo3D( imgDisparity8U, recons3D, Q, false, CV_32F );

//        //Create point cloud and fill it
//        std::cout << "Creating Point Cloud..." <<std::endl;
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

//        double px, py, pz;
//        uchar pr, pg, pb;

//          for (int i = 0; i < img1.rows; i++)
//          {
//            uchar* rgb_ptr = img1.ptr<uchar>(i);
////        #ifdef CUSTOM_REPROJECT
////            uchar* disp_ptr = img_disparity.ptr<uchar>(i);
////        #else
//            double* recons_ptr = recons3D.ptr<double>(i);
//        //#endif
//            for (int j = 0; j < img1.cols; j++)
//            {
//              //Get 3D coordinates
////        #ifdef CUSTOM_REPROJECT
////              uchar d = disp_ptr[j];
////              if ( d == 0 ) continue; //Discard bad pixels
////              double pw = -1.0 * static_cast<double>(d) * Q32 + Q33;
////              px = static_cast<double>(j) + Q03;
////              py = static_cast<double>(i) + Q13;
////              pz = Q23;

////              px = px/pw;
////              py = py/pw;
////              pz = pz/pw;

////        #else
//              px = recons_ptr[3*j];
//              py = recons_ptr[3*j+1];
//              pz = recons_ptr[3*j+2];
////        #endif

//              //Get RGB info
//              pb = rgb_ptr[3*j];
//              pg = rgb_ptr[3*j+1];
//              pr = rgb_ptr[3*j+2];

//              //Insert info into point cloud structure
//              pcl::PointXYZRGB point;
//              point.x = px;
//              point.y = py;
//              point.z = pz;
//              uint32_t rgb = (static_cast<uint32_t>(pr) << 16 |
//                      static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
//              point.rgb = *reinterpret_cast<float*>(&rgb);
//              point_cloud_ptr->points.push_back (point);
//            }
//          }
//          point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
//          point_cloud_ptr->height = 1;

//          //Create visualizer
//          boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
//          viewer = createVisualizer( point_cloud_ptr );

//          //Main loop
//          while ( !viewer->wasStopped())
//          {
//            viewer->spinOnce(100);
//            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//          }


//        return 1;
//}




#include <stdio.h>
#include <iostream>
#include "opencv2/calib3d/calib3d.hpp"
   #include "opencv2/core/core.hpp"
   #include "opencv2/highgui/highgui.hpp"

   using namespace cv;

   char *windowDisparity = "Disparity";

   void readme();

   /**
    * @function main
    * @brief Main function
    */
   int main( int argc, char** argv )
   {
     if( argc != 3 )
     { readme(); return -1; }

     //-- 1. Read the images
     Mat imgLeft = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
     Mat imgRight = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
     //-- And create the image in which we will save our disparities
     Mat imgDisparity16S = Mat( imgLeft.rows, imgLeft.cols, CV_16S );
     Mat imgDisparity8U = Mat( imgLeft.rows, imgLeft.cols, CV_8UC1 );

     if( !imgLeft.data || !imgRight.data )
     { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

     //-- 2. Call the constructor for StereoBM
     int ndisparities = 16*5;   /**< Range of disparity */
     int SADWindowSize = 21; /**< Size of the block window. Must be odd */

     StereoBM sbm( StereoBM::BASIC_PRESET,
                                                                   ndisparities,
                   SADWindowSize );

     //-- 3. Calculate the disparity image
     sbm( imgLeft, imgRight, imgDisparity16S, CV_16S );

     //-- Check its extreme values
     double minVal; double maxVal;

     minMaxLoc( imgDisparity16S, &minVal, &maxVal );

     printf("Min disp: %f Max value: %f \n", minVal, maxVal);

     //-- 4. Display it as a CV_8UC1 image
     imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));

     namedWindow( windowDisparity, CV_WINDOW_NORMAL );
     imshow( windowDisparity, imgDisparity8U );

     //-- 5. Save the image
     imwrite("SBM_sample.png", imgDisparity16S);

     waitKey(0);

     return 0;
   }

   /**
    * @function readme
    */
   void readme()
   { std::cout << " Usage: ./SBMSample <imgLeft> <imgRight>" << std::endl; }
