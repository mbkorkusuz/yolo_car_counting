#include "kalman_filter.hpp"
#include <chrono>

KalmanFilter::KalmanFilter() : kf(4, 2), initialized(false) {
    // Varsayılan dt = 1
    setTransitionMatrix(1.0f);

    measurement = cv::Mat::zeros(2, 1, CV_32F);

    // ölçüm matrisleri
    kf.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-3;
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-2;
    kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

    last_time = std::chrono::steady_clock::now();
}

void KalmanFilter::setTransitionMatrix(float dt)
{
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1);
}

void KalmanFilter::init(const cv::Point2f &pt) 
{
    state = (cv::Mat_<float>(4, 1) << pt.x, pt.y, 0, 0);
    kf.statePost = state;
    initialized = true;
    last_time = std::chrono::steady_clock::now();
}

cv::Point2f KalmanFilter::predict() 
{
    if (!initialized) return cv::Point2f(-1, -1);

    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - last_time).count();
    last_time = now;

    setTransitionMatrix(dt);

    cv::Mat prediction = kf.predict();
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

void KalmanFilter::correct(const cv::Point2f &pt) 
{
    if (!initialized) return;
    measurement.at<float>(0) = pt.x;
    measurement.at<float>(1) = pt.y;
    kf.correct(measurement);
}

cv::Point2f KalmanFilter::getVelocity() const
{
    return cv::Point2f(kf.statePost.at<float>(2), kf.statePost.at<float>(3));
}