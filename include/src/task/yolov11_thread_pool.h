#ifndef RK3588_DEMO_YOLOV11_THREAD_POOL_H
#define RK3588_DEMO_YOLOV11_THREAD_POOL_H

#include "yolov11_custom.h"

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

class Yolov11ThreadPool
{
private:
    std::queue<std::pair<int, cv::Mat>> tasks;                     // <id, img> to store tasks
    std::vector<std::shared_ptr<Yolov11Custom>> yolov11_instances; // model instances
    std::map<int, std::vector<Detection>> results;                 // <id, objects> to store detection results
    std::map<int, cv::Mat> img_results;                            // <id, img> to store processed images
    std::vector<std::thread> threads;                              // thread pool
    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable cv_task;
    bool stop;

    void worker(int id);

public:
    Yolov11ThreadPool();  // Constructor
    ~Yolov11ThreadPool(); // Destructor

    nn_error_e setUp(std::string &model_path, int num_threads = 12);             // Initialize
    nn_error_e submitTask(const cv::Mat &img, int id);                           // Submit task
    nn_error_e getTargetResult(std::vector<Detection> &objects, int id);         // Get result (detections)
    nn_error_e getTargetImgResult(cv::Mat &img, int id);                         // Get result (image)
    nn_error_e getTargetResultNonBlock(std::vector<Detection> &objects, int id); // Get result (detections) non-blocking
    void stopAll();                                                              // Stop all threads
};

#endif // RK3588_DEMO_YOLOV11_THREAD_POOL_H 