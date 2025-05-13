#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <opencv2/video/tracking.hpp>
#include <opencv2/core/types.hpp>
#include <chrono>

class KalmanFilter {
public:
    KalmanFilter();

    void init(const cv::Point2f &pt);
    cv::Point2f predict();
    void correct(const cv::Point2f &pt);
    cv::Point2f getVelocity() const;

private:
    void setTransitionMatrix(float dt);

    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat measurement;
    bool initialized;
    std::chrono::steady_clock::time_point last_time;
};


#endif