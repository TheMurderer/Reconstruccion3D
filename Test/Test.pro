#-------------------------------------------------
#
# Project created by QtCreator 2013-02-08T19:45:49
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = Test
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp

INCLUDEPATH += /usr/local/include/
INCLUDEPATH += /usr/include/pcl-1.6/
INCLUDEPATH += /usr/include/boost
INCLUDEPATH += /usr/include/eigen3/Eigen
INCLUDEPATH += /usr/include/vtk-5.8

LIBS += -L/usr/local/lib \
-lopencv_core \
-lopencv_imgproc \
-lopencv_highgui \
-lopencv_ml \
-lopencv_video \
-lopencv_features2d \
-lopencv_calib3d \
-lopencv_objdetect \
-lopencv_contrib \
-lopencv_legacy \
-lopencv_nonfree \
-lopencv_flann

CONFIG += link_pkgconfig
PKGCONFIG += pcl_registration-1.6
PKGCONFIG += pcl_geometry-1.6
PKGCONFIG += pcl_features-1.6
PKGCONFIG += pcl_search-1.6
PKGCONFIG += pcl_kdtree-1.6
PKGCONFIG += pcl_filters-1.6
PKGCONFIG += pcl_surface-1.6
PKGCONFIG += pcl_octree-1.6
PKGCONFIG += pcl_sample_consensus-1.6
PKGCONFIG += pcl_segmentation-1.6
PKGCONFIG += pcl_visualization-1.6
PKGCONFIG += pcl_io-1.6
PKGCONFIG += pcl_apps-1.6
PKGCONFIG += pcl_keypoints-1.6
PKGCONFIG += pcl_tracking-1.6


