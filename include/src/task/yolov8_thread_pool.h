#ifndef YOLOV8_THREAD_POOL_H
#define YOLOV8_THREAD_POOL_H

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

// 引入你的自定义模型头文件
#include "yolov8_custom.h" 

class Yolov8ThreadPool {
public:
    Yolov8ThreadPool();
    ~Yolov8ThreadPool();

    nn_error_e setUp(std::string &model_path, int num_threads);
    nn_error_e submitTask(const cv::Mat &img, int id);
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id);
    nn_error_e getTargetImgResult(cv::Mat &img, int id);
    nn_error_e getTargetResultNonBlock(std::vector<Detection> &objects, int id);
    void stopAll();

private:
    void worker(int id);

    bool stop;
    std::vector<std::thread> threads;
    
    // 多实例存放数组
    std::vector<std::shared_ptr<Yolov8Custom>> Yolov8_instances;
    
    std::queue<std::pair<int, cv::Mat>> tasks;
    std::map<int, std::vector<Detection>> results;
    std::map<int, cv::Mat> img_results;
    
    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable cv_task;
};

#endif // YOLOV8_THREAD_POOL_H