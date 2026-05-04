#ifndef DEEPSORT_ADAPTER_H
#define DEEPSORT_ADAPTER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "deepsort.h"
#include "box.h"

// 跟踪框结构体
struct TrackingBox {
    int id;             // 目标ID
    cv::Rect box;       // 边界框
    float confidence;   // 置信度
};

class DeepSortAdapter {
public:
    DeepSortAdapter(const std::string& model_path, int batch_size = 1, int feature_dim = 512, int max_budget = 100, rknn_core_mask npu_id = RKNN_NPU_CORE_2);
    ~DeepSortAdapter();

    // 使用DeepSORT更新跟踪结果
    std::vector<TrackingBox> update(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, const cv::Mat& frame);

private:
    DeepSort* deepsort;
};

#endif // DEEPSORT_ADAPTER_H 