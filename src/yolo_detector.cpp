#include <fstream>
#include "yolo_detector.hpp"
float INPUT_WIDTH = 640;
float INPUT_HEIGHT = 640;

float SCORE_THRESHOLD = 0.4;
float NMS_THRESHOLD = 0.2;

// sadece araba, kamyon ve motosiklet tespitlerini dikkate al
std::vector<std::string> acceptedClasses = {"car", "motorbike", "truck"};

//detector objem
YOLODetector::YOLODetector(const std::string &modelPath, const std::string &classesPath)
{
    net = cv::dnn::readNet(modelPath);
    
    // cuda varsa cuda kullan
    if (cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else // cpu
    {
        std::cout << "hello" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    classNames = loadClassTXT(classesPath);
}

// classları dışarıdan txt dosyasından okuyorum. Kendim direkt buraya da yapıştırabilirdim
std::vector<std::string> YOLODetector::loadClassTXT(const std::string &path)
{
    std::vector<std::string> listOfClasses;
    
    std::string className;

    std::fstream classTXT(path);

    while(getline(classTXT, className))
    {
        listOfClasses.push_back(className);
    }

    return listOfClasses;
}

// tespitleri yap
std::vector<cv::Rect> YOLODetector::detect(cv::Mat &frame)
{
    std::vector<cv::Mat> outputs;
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    if (net.empty()) {
        std::cout << "Model yüklenemedi!" << std::endl;   
    }
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    if (outputs.empty()) {
        std::cout << "Çıktılar boş!" << std::endl;
    }

    std::vector<int> classIDs;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Resizing factors
    float xFactor = frame.cols / INPUT_WIDTH;
    float yFactor = frame.rows / INPUT_HEIGHT;
    
    int dimensions = outputs[0].size[1];
    int rows = outputs[0].size[2];

    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    float* data = (float*)outputs[0].data;

    for (int i = 0; i < rows; i++)
    {
        // Output data: x_min, y_min, x_max, y_max, confidence, class_probabilities      
        float* classScores = data + 4;
        cv::Mat scores(1, classNames.size(), CV_32FC1, classScores);
        cv::Point classID;
        double maxClassScore;

        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classID);

        if (maxClassScore >= SCORE_THRESHOLD)
        {   
            // eğer tespit edilen obje araba, kamyon veya motosiklet değilse ignore et
            if (std::find(acceptedClasses.begin(), acceptedClasses.end(), classNames[classID.x]) == acceptedClasses.end())
            {
                continue;
            }

            confidences.push_back(maxClassScore);
            classIDs.push_back(classID.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * xFactor);
            int top = int((y - 0.5 * h) * yFactor);

            int width = int(w * xFactor);
            int height = int(h * yFactor);

            boxes.push_back(cv::Rect(left, top, width, height));

        }
        

        data += dimensions;  // Move to the next detection
    }

    // Non-Maximum Suppression (NMS) işlemi
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);


    // tespitleri döndür
    std::vector<cv::Rect> returnBoxes;
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        returnBoxes.push_back(box);
    }
    return returnBoxes;
}
