#include "bytetrack.hpp"
#include <iostream>
BYTETracker::BYTETracker(int maxAge, int minHits, float iouThreshold):
    nextID(0), maxAge(maxAge), minHits(minHits), iouThreshold(iouThreshold), vehicleCount(0) {}


int lineTrack(const cv::Point &center, const cv::Vec4i &line) // helper function.
{
    
    float crossProduct;

    crossProduct = (float)((line[2] - line[0])*(center.y - line[1])) - ((line[3] - line[1])*(center.x - line[0]));

    if (crossProduct < 0) // altında
    {
        return 1;
    }
    else // üstünde
    {
        return 2;
    }
}

// intersection over union hesaplaması
float BYTETracker::computeIOU(const cv::Rect &box1, const cv::Rect &box2)
{
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int interWidth = x2 - x1;
    int interHeight = y2 - y1;

    if (interWidth <= 0 || interHeight <= 0)
        return 0.0;

    int interArea = interWidth * interHeight;
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;

    float iou = (float) interArea / (box1Area + box2Area - interArea);
    
    return iou;
}

bool BYTETracker::hasCrossedLine(Track &track, const cv::Vec4i &line)
{
    // boxun merkezi
    cv::Point center(track.box.x + track.box.width/2, track.box.y + track.box.height/2);


    // aracın başlangıç noktası geçiş çizgisisinin altındaysa ve daha önce geçmemişse
    if (track.startPoint == 1 && !track.crossedLine)
    {   
        // eğer araç şu anda çizginin üstündeyse
        if(lineTrack(center, line) == 2)
        {
            track.crossedLine = true;
            return true;
        }
    }
    // aracın başlangıç noktası geçiş çizgisisinin üstündeyse ve daha önce geçmemişse
    if(track.startPoint == 2 && !track.crossedLine)
    {
        // eğer araç şu anda çizginin altındaysa
        if(lineTrack(center, line) == 1)
        {
            track.crossedLine = true;
            return true;
        }
    }
    
    return false;
}

// trackleri güncelleme fonksiyonu
std::vector<Track> BYTETracker::update(const std::vector<cv::Rect> &detections, const std::vector<cv::Vec4i> &lines)
{
    std::vector<Track> activeTracks;
    std::vector<int> assignedTrackIDs;
    cv::Point center;

    // eşleştirme ve id atama
    for (int i = 0; i < detections.size(); i++)
    {
        cv::Rect det = detections[i];

        int bestMatchID = -1;
        float bestIOU = 0.0f;

        for(auto & [id, track] : tracks)
        {
            cv::Point2f predicted = track.kalman.predict();
            // predict edilen merkezden yeni box oluştur
            int width = track.box.width;
            int height = track.box.height;
            track.box = cv::Rect(predicted.x - width / 2, predicted.y - height / 2, width, height);

            float iou = computeIOU(det, track.box);

            if (iou > iouThreshold && iou > bestIOU)
            {
                bestIOU = iou;
                bestMatchID = id;
            }
        }

        // yeni id atama
        if (bestMatchID == -1)
        {
            Track newTrack = {nextID++, det, 0, 0};
            newTrack.kalman.init(cv::Point2f(det.x + det.width/2, det.y + det.height/2));
            center = cv::Point(det.x + det.width / 2, det.y + det.height / 2);
            newTrack.history.push_back(center);

            // ilk tespitte geçiş çizgisinin altında mı yoksa üstünde mi
            if (lineTrack(center, lines[0]) == 1)
            {
                newTrack.startPoint = 1;
            }
            else
            {
                newTrack.startPoint = 2;
            }
            tracks[newTrack.id] = newTrack;
            activeTracks.push_back(newTrack);
        }

        else
        {
            // var olan nesneyi güncelle
            tracks[bestMatchID].age++;
            tracks[bestMatchID].missed = 0;
            cv::Point center(det.x + det.width / 2, det.y + det.height / 2);
            tracks[bestMatchID].history.push_back(center);
            assignedTrackIDs.push_back(bestMatchID);

            // Kalman filter düzeltme
            center = cv::Point(det.x + det.width / 2, det.y + det.height / 2);
            tracks[bestMatchID].kalman.correct(center);

            tracks[bestMatchID].box = cv::Rect(center.x - det.width / 2, center.y - det.height / 2, det.width, det.height);            

            activeTracks.push_back(tracks[bestMatchID]);

            // çizgi kontrolü
            if (tracks[bestMatchID].crossedLine)
            {
                // burada aslında geçmiş gözüken araç tracks listesinden silinebilir
                continue;
            }
            
            if (tracks[bestMatchID].startPoint == 1)
            {
                if(hasCrossedLine(tracks[bestMatchID], lines[1])) // gelen arabalar
                {
                    vehicleCount++;
                    //std::cout << "Vehicle " << tracks[bestMatchID].id << " Crossed the line\n" << "Vehicle Count: " << vehicleCount <<std::endl;
                }
            }
            else
            {   
                if(hasCrossedLine(tracks[bestMatchID], lines[0])) // giden arabalar
                {
                    vehicleCount++;
                    //std::cout << "Vehicle " << tracks[bestMatchID].id << " Crossed the line\n" << "Vehicle Count: " << vehicleCount << std::endl;
                }
            }    
        }
    }

    // kaybolan araçları güncelleme
    for(auto it = tracks.begin(); it != tracks.end();)
    {
        if (std::find(assignedTrackIDs.begin(),assignedTrackIDs.end(), it->first) == assignedTrackIDs.end())
        {
            it->second.missed++;
            if (it->second.missed > maxAge)
            {
                it = tracks.erase(it);
            }
            else
            {
                it++;
            }
        }
        else
        {
            it++;
        }
    }

    return activeTracks;    
}