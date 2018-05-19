
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

#include "MTCNN.h"




int main(int argc, char **argv)
{
    argv[1] = "../img/lena.jpg";
    MTCNN detector("../model");
    float factor = 0.709f;
    float threshold[3] = { 0.7f, 0.6f, 0.6f };
    int minSize = 15;

    cv::Mat image = cv::imread(argv[1]);
    int stage =3;

    double t = (double)cv::getTickCount();
    std::vector<FaceInfo> faceInfo = detector.Detect(image, minSize, threshold, factor, stage);
    std::cout <<" time," << (double)(cv::getTickCount() - t) / cv::getTickFrequency() << "s"<<std::endl;

    for (int i = 0; i < faceInfo.size(); i++){
        int x = (int)faceInfo[i].bbox.xmin;
        int y = (int)faceInfo[i].bbox.ymin;
        int w = (int)(faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
        int h = (int)(faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
        cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
    }


    for (int i = 0; i < faceInfo.size(); i++){
        float *landmark = faceInfo[i].landmark;
        for (int j = 0; j < 5; j++){
            cv::circle(image, cv::Point((int)landmark[2 * j], (int)landmark[2 * j + 1]), 1, cv::Scalar(255, 255, 0), 2);
        }
    }
    cv::imshow("image", image);
    cv::waitKey();

    return 1;
}