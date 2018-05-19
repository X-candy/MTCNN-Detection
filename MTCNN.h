//
// Created by xcandy on 2018/5/19.
//

#ifndef MTCNN_MTCNN_H
#define MTCNN_MTCNN_H
#include <caffe/caffe.hpp>
#include <string>
#include <opencv2/opencv.hpp>


using namespace caffe;
using namespace std;
using namespace cv;

#define __PNET_S__  2
#define __PNET_CELL_SIZE__  12
#define __PNET_MAX_DET__  5000
#define __MEAN_VAL__  127.5f
#define __STD_VAL__  0.0078125f
#define __MINBATCH__  128


typedef struct FaceBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
} FaceBox;

typedef struct FaceInfo {
    float bbox_reg[4];
    float landmark[10];
    FaceBox bbox;
} FaceInfo;

class MTCNN {
public:
    MTCNN(const string& proto_model_dir);
    vector<FaceInfo> Detect(const cv::Mat& img, const int min_size, const float* threshold, const float factor, const int stage);
protected:
    vector<FaceInfo> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
    vector<FaceInfo> NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
    void BBoxRegression(vector<FaceInfo>& bboxes);
    void BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height);
    void BBoxPad(vector<FaceInfo>& bboxes, int width, int height);
    void GenerateBBox(Blob<float>* confidence, Blob<float>* reg_box, float scale, float thresh);
    std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
    float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);

private:
    boost::shared_ptr<Net<float>> PNet_;
    boost::shared_ptr<Net<float>> RNet_;
    boost::shared_ptr<Net<float>> ONet_;

    std::vector<FaceInfo> candidate_boxes_;
    std::vector<FaceInfo> total_boxes_;
};



#endif //MTCNN_MTCNN_H
