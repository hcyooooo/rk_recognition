#ifndef _RKNN_DEMO_LPRNET_H_
#define _RKNN_DEMO_LPRNET_H_

#include "rknn_api.h"
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp> // 直接用 OpenCV

#define MODEL_HEIGHT 24
#define MODEL_WIDTH 94
#define OUT_ROWS 68
#define OUT_COLS 18

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_channel;
    int model_width;
    int model_height;
} rknn_app_context_t;

typedef struct {
    std::string plate_name;
} lprnet_result;

// 初始化模型
int init_lprnet_model(const char *model_path, rknn_app_context_t *app_ctx);
// 释放模型
int release_lprnet_model(rknn_app_context_t *app_ctx);
// 极简版推理接口：直接传 OpenCV 的 cv::Mat 进来！
int inference_lprnet_model(rknn_app_context_t *app_ctx, const cv::Mat& src_img, lprnet_result *out_result);

#endif //_RKNN_DEMO_LPRNET_H_