#ifndef _RKNN_DEMO_PPOCR_H_
#define _RKNN_DEMO_PPOCR_H_

#include "rknn_api.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Define standard input size for PPOCR-Rec (usually 48x320)
#define PPOCR_MODEL_HEIGHT 48
#define PPOCR_MODEL_WIDTH 320

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    std::vector<std::string> dict; // To hold the dictionary loaded from ppocr_keys_v1.txt
} ppocr_app_context_t;

typedef struct {
    std::string text;
    float score;
} ppocr_result;

// Initialize the model and load the dictionary
int init_ppocr_model(const char *model_path, const char *dict_path, ppocr_app_context_t *app_ctx);

// Release resources
int release_ppocr_model(ppocr_app_context_t *app_ctx);

// Perform inference
int inference_ppocr_model(ppocr_app_context_t *app_ctx, const cv::Mat& src_img, ppocr_result *out_result);

#endif //_RKNN_DEMO_PPOCR_H_