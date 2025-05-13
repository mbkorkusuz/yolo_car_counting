#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "bytetrack.hpp"

void draw(cv::Mat &frame, std::vector<Track> &tracks, std::vector<cv::Vec4i> &crossLines);
cv::Vec4i computeCrossLine(const std::vector<float> &angles, const cv::Size &frameSize, int direction);
std::vector<cv::Vec4i> generateCrossLineFromTracks(const std::vector<Track> &tracks, const cv::Size &frameSize);
void readFrames(cv::VideoCapture &capture);
void processFrames(cv::Mat &firstFrame);

#endif // UTILS_HPP
