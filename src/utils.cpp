#include "utils.hpp"
#include "yolo_detector.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <queue>

// maindeki global değişkenler
extern std::ofstream outputTXT;
extern cv::Scalar GREEN, RED, YELLOW, BLACK;
extern std::queue<cv::Mat> unprocessedFrameQueue;
extern std::queue<cv::Mat> processedFrameQueue;
extern std::mutex mut, mut2;
extern bool isVideoReadingFinished, isVideoProcessingFinished;
extern std::condition_variable canReadFrame, canShowFrame;

// draw fonksiyonu detect edilen arabaların etrafına bir kutu bu kutuların üstüne de unique idleri yazdırıyor
// aynı zamanda geçiş kontrolü yaptığım line ları çiziyor.
// istenirse kapatılabilinir.
void draw(cv::Mat &frame, std::vector<Track> &tracks, std::vector<cv::Vec4i> &crossLines)
{
    
    for (int i = 0; i < tracks.size(); i++)
    {
        Track track = tracks[i];
        cv::rectangle(frame, track.box, GREEN, 2);
        cv::putText(frame, "ID: " + std::to_string(track.id), cv::Point(track.box.x, track.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, RED, 2);
    }
    cv::line(frame, cv::Point(crossLines[0][0], crossLines[0][1]), cv::Point(crossLines[0][2], crossLines[0][3]), YELLOW, 2);
    cv::line(frame, cv::Point(crossLines[1][0], crossLines[1][1]), cv::Point(crossLines[1][2], crossLines[1][3]), BLACK, 2);

    return;
}

// açılarını topladığım tracklerin açı ortalamasından bir tane final açı buluyorum ve bu açıyı arabaların yönüne göre 30 veya -30 derece döndürüyorum
// 30 ve -30 dereceyi deneyerek buldum. En iyi çalışan değerler bunlar oldu.
cv::Vec4i computeCrossLine(const std::vector<float> &angles, const cv::Size &frameSize, int direction)
{
    if (angles.empty())
    {
        // Ekranın orta noktasından geçen standart çizgi.
        // Eğer yeterince araba yoksa o an ekranda standart çizgiye geç.
        cv::Vec4i tempLine;
    
        tempLine[0] = 0;
        tempLine[1] = frameSize.height - (frameSize.height * 0.65);
        tempLine[2] = frameSize.width;
        tempLine[3] = frameSize.height - (frameSize.height * 0.65);

        return tempLine;
    } 

    float sumSin = 0.0f, sumCos = 0.0f;
    for (int i = 0; i < angles.size(); i++)
    {
        float angle = angles[i];
        sumSin += std::sin(angle);
        sumCos += std::cos(angle);
    }
    
    float avgAngle = std::atan2(sumSin, sumCos);
    float crossAngle;

    if (direction == 2)
    {
        crossAngle = avgAngle - CV_PI / 6.0f;
    }
    else
    {
        crossAngle = avgAngle + CV_PI / 6.0f;
    }
    
    float unitX = std::cos(crossAngle);
    float unitY = std::sin(crossAngle);

    float length = frameSize.width;
    
    cv::Point2f mid(frameSize.width / 2.0f, frameSize.height * 0.4);

    cv::Point2f pt1 = mid + cv::Point2f(unitX * length, unitY * length);
    cv::Point2f pt2 = mid - cv::Point2f(unitX * length, unitY * length);

    return cv::Vec4i(pt1.x, pt1.y, pt2.x, pt2.y);

}


std::vector<cv::Vec4i> generateCrossLineFromTracks(const std::vector<Track> &tracks, const cv::Size &frameSize)
{
    if (tracks.empty())
    {
        // Eğer bir anda ekranda araç yoksa standart line ı geçiş çizgim olarak belirliyorum.
        cv::Vec4i tempLine;
    
        tempLine[0] = 0;
        tempLine[1] = frameSize.height - (frameSize.height * 0.65);
        tempLine[2] = frameSize.width;
        tempLine[3] = frameSize.height - (frameSize.height * 0.65);

        return {tempLine, tempLine};
    }

    
    std::vector<float> upgoingAngles;
    std::vector<float> incomingAngles;

    Track track;
    // Giden ve gelen arabalar için ayrı ayrı track yollarının açılarını buluyorum
    for (int i = 0; i < tracks.size(); i++)
    {
        track = tracks[i];

        if (track.history.size() >= 2)
        {
            cv::Point p1 = track.history.front();
            cv::Point p2 = track.history.back();

            float dx = p2.x - p1.x;
            float dy = p2.y - p1.y;

            float angle = std::atan2(dy, dx);

            if (dy > 0) 
            {
                upgoingAngles.push_back(angle);   // Aşağı gidiyor
            }
            else
            {
                incomingAngles.push_back(angle); // Yukarı çıkıyor
            }
        }

    }

    cv::Vec4i lineForGoingCars = computeCrossLine(upgoingAngles, frameSize, 1); // giden arabalar
    cv::Vec4i lineForComingCars = computeCrossLine(incomingAngles, frameSize, 2); // gelen arabalar

    return {lineForGoingCars, lineForComingCars};
}

// bu thread sadece frameleri inputtan okuyup bir queueya (unprocessedFrameQueue) atıyor.
void readFrames(cv::VideoCapture &capture)
{
    while(true)
    {
        cv::Mat tempFrame;
        capture.read(tempFrame);

        if (tempFrame.empty())
        {
            std::cout << "All frames have been read." << std::endl;
            break;
        }

        {
            std::lock_guard<std::mutex> guard(mut);
            unprocessedFrameQueue.push(tempFrame);
        }
        canReadFrame.notify_one();
    }
    isVideoReadingFinished = true;
    canReadFrame.notify_one();
    return;
}

// Detection + Track + Count threadi
void processFrames(cv::Mat &firstFrame)
{
    // model yüklenmesi
    YOLODetector detector("../models/yolov8n.onnx", "../models/classesv8.txt");
    // Kendi tracker classımdan tracker objesi
    BYTETracker tracker(30, 3, 0.3f);

    // Standart linelar (ekranın orta noktasından düz çizgi)
    cv::Vec4i tempLineForIncoming, tempLineForOutGoing;
    
    tempLineForIncoming[0] = 0;
    tempLineForIncoming[1] = firstFrame.rows - (firstFrame.rows * 0.65);
    tempLineForIncoming[2] = firstFrame.cols;
    tempLineForIncoming[3] = firstFrame.rows - (firstFrame.rows * 0.65);
    tempLineForOutGoing = tempLineForIncoming;

    cv::Mat currentFrame;

    // currentFrame de detect edilen araçlar için kutularım ve Track objelerim
    std::vector<cv::Rect> boxes;
    std::vector<Track> tracks;

    // ilerde hesaplanacak geçiş kontrol çizgim için listem (2 elemanlı olacak giden ve gelen arabalar için)
    std::vector<cv::Vec4i> crossLines;

    // geçiş kontrol çizgilerimi her 30 frame de bir değiştiriyorum
    // yine bu değeri deneyerek buldum. Hem çok stabil hem de çok değişken olmasın istedim
    int periodOfLine = 30;

    // detection ve tracking ana döngüm
    while(true)
    {
        
        // burada okunulan frameleri çekiyorum
        // readFrames threadiyle senkronize çalışmalı çünkü ortak bir değişken var (unprocessedFrameQueue)
        {
            std::unique_lock<std::mutex> lock(mut);
            canReadFrame.wait(lock, []{return !unprocessedFrameQueue.empty() || isVideoReadingFinished;});

            if (unprocessedFrameQueue.empty() && isVideoReadingFinished)
            {
                break;
            }
            currentFrame = unprocessedFrameQueue.front();
            unprocessedFrameQueue.pop();
        }

        // geçiş kontrol çizgimi değiştirme zamanım geldi mi gelmedi mi
        if (periodOfLine % 30 == 0)
        {
            crossLines = generateCrossLineFromTracks(tracks, currentFrame.size());
            tempLineForOutGoing = crossLines[0];
            tempLineForIncoming = crossLines[1];
            periodOfLine = 0;
        }
        else // zamanı gelmediyse en son kullanılan çizgiyi kullanmaya devam et
        {
            crossLines = {tempLineForOutGoing, tempLineForIncoming};
        }
        periodOfLine++;
        
        // araç tespiti
        boxes = detector.detect(currentFrame);
        // araç takibi
        // id atama ve sayım işlemleri update metodunda gerçekleşiyor
        tracks = tracker.update(boxes, crossLines);

        // sonuçları ekrana çiz
        draw(currentFrame, tracks, crossLines);

        // toplam sayımı da ekrana yaz
        cv::putText(currentFrame, "Count: " + std::to_string(tracker.vehicleCount), cv::Point(0, 80), cv::FONT_HERSHEY_DUPLEX, 1.0, RED, 2);

        // burada main thread ile senkronize çalışmalı onun güvenliği
        {
            std::lock_guard<std::mutex> guard(mut2);
            processedFrameQueue.push(currentFrame);
        }

        // main threadi uyandır.
        canShowFrame.notify_one();

    }
    
    // tüm girdi okunduktan sonra outputa yazdır.
    std::string totalVehicleCount = std::to_string(tracker.vehicleCount);

    outputTXT << totalVehicleCount;
    outputTXT.close();
    isVideoProcessingFinished = true;
    canShowFrame.notify_one();
}