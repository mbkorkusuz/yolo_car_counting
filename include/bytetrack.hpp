#ifndef BYTETRACK_H
#define BYTETRACK_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include "kalman_filter.hpp"

struct Track
{
    int id;
    cv::Rect box;
    int age;
    int missed;
    bool crossedLine = false;
    int startPoint;
    KalmanFilter kalman;
    std::vector<cv::Point> history;
};

class BYTETracker
{
    public:
        BYTETracker(int maxAge = 30, int minHits = 3, float iouThreshold = 0.3);
        std::vector<Track> update(const std::vector<cv::Rect> & detections, const std::vector<cv::Vec4i> &lines);
        int vehicleCount;

    private:
        int nextID;
        int maxAge;
        int minHits;
        float iouThreshold;
        std::map<int, Track> tracks;
        float computeIOU(const cv::Rect &box1, const cv::Rect &box2);
        bool hasCrossedLine(Track &track, const cv::Vec4i &line);
};

#endif