#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class YOLODetector
{
    public:
        YOLODetector(const std::string &modelPath, const std::string &classesPath);
        std::vector <cv::Rect> detect(cv::Mat &frame);

    private:
        cv::dnn::Net net;
        std::vector<std::string> classNames;
        std::vector<std::string> loadClassTXT(const std::string &path);
};


#endif