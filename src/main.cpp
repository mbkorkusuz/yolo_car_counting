#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <fstream>
#include "utils.hpp"

std::ofstream outputTXT;
cv::Scalar GREEN = CV_RGB(0,255,0);
cv::Scalar RED = CV_RGB(255,0,0);
cv::Scalar YELLOW = CV_RGB(130,130,65);
cv::Scalar BLACK = CV_RGB(0,0,0);

std::queue<cv::Mat> unprocessedFrameQueue;
std::queue<cv::Mat> processedFrameQueue;
std::mutex mut;
std::mutex mut2;
bool isVideoReadingFinished = false;
bool isVideoProcessingFinished = false;
std::condition_variable canReadFrame;
std::condition_variable canShowFrame;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Lütfen video dosyasını ve çıktı yolunu sağlayın" << std::endl;
        return -1;
    }

    // input video ve output txt yolları.
    std::string videoPath = argv[1];
    outputTXT.open(argv[2]);

    cv::VideoCapture capture(videoPath);

    if (!capture.isOpened())
    {
        std::cout << "Video açılamadı" << std::endl;
        return -1;
    }

    // first frame i videonun boyutunu öğrenmek için atıyorum. Çok da gerekli değildi aslında
    cv::Mat firstFrame;
    capture.read(firstFrame);
    
    // threadleri başlattım
    std::thread captureThread(readFrames, std::ref(capture));
    std::thread processThread(processFrames, std::ref(firstFrame));

    cv::Mat currentFrame;

    // FPS hesabı için değişkenler
    int currentFPS;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> timeElapsed;

    // her bir frame process edildikten sonra ekranda frame i ve fps i gösteriyorum.
    while(!isVideoProcessingFinished)
    {   
        // fps hesabı için başlangıç zamanı
        startTime = std::chrono::high_resolution_clock::now();

        // burada process frame i lock ediyorum çünkü onunla ortaklaşa kullanılan bir queue var (processedFrameQueue).
        {
            std::unique_lock<std::mutex> lock(mut2);
            canShowFrame.wait(lock, []{return !processedFrameQueue.empty() || isVideoProcessingFinished;});

            if (processedFrameQueue.empty() && isVideoProcessingFinished)
            {
                break;
            }
            currentFrame = processedFrameQueue.front();
            processedFrameQueue.pop();
        }

        
        endTime = std::chrono::high_resolution_clock::now();
        timeElapsed = endTime - startTime;
        currentFPS = (int)(1000/timeElapsed.count());
        
        // fps çizimi
        cv::putText(currentFrame, "FPS: " + std::to_string(currentFPS), cv::Point(0, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, GREEN, 2);
        
        // frame i göster
        cv::imshow("Tracking", currentFrame);
        
        // kullanıcı ESC ye bastığında çıkış yap.
        if(cv::waitKey(1) == 27)
        {
            break;
        }

    }

    // thread'lerin bitmesini bekle
    captureThread.join();
    processThread.join();
    

    return 0;
    
}