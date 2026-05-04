#include "ppocr.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

static unsigned char* load_data(const char* filename, int* size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return nullptr;
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(*size);
    if (data) fread(data, 1, *size, fp);
    fclose(fp);
    return data;
}

static bool load_dict(const char* dict_path, std::vector<std::string>& dict) {
    dict.clear();
    dict.push_back("blank"); // 索引 0: CTC blank
    std::ifstream file(dict_path);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        dict.push_back(line);
    }
    dict.push_back(" "); // 索引末尾补充空格
    return true;
}

int init_ppocr_model(const char *model_path, const char *dict_path, ppocr_app_context_t *app_ctx) {
    int ret, model_len = 0;
    unsigned char *model_data = load_data(model_path, &model_len);
    if (model_data == NULL) {
        printf("Load PPOCR model failed!\n");
        return -1;
    }

    rknn_context ctx = 0;
    ret = rknn_init(&ctx, model_data, model_len, 0, NULL);
    free(model_data);
    if (ret < 0) return -1;

    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));

    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    memset(app_ctx->input_attrs, 0, app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
    }

    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    memset(app_ctx->output_attrs, 0, app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(app_ctx->output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    if (app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        app_ctx->model_channel = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_height = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_width = app_ctx->input_attrs[0].dims[3];
    } else {
        app_ctx->model_height = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_width = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_channel = app_ctx->input_attrs[0].dims[3];
    }
    
    app_ctx->rknn_ctx = ctx;

    if (!load_dict(dict_path, app_ctx->dict)) {
        printf("Load PPOCR dict failed!\n");
        return -1;
    }
    return 0;
}

int release_ppocr_model(ppocr_app_context_t *app_ctx) {
    if (app_ctx->input_attrs) free(app_ctx->input_attrs);
    if (app_ctx->output_attrs) free(app_ctx->output_attrs);
    if (app_ctx->rknn_ctx != 0) rknn_destroy(app_ctx->rknn_ctx);
    return 0;
}

int inference_ppocr_model(ppocr_app_context_t *app_ctx, const cv::Mat& src_img, ppocr_result *out_result) {
    if (src_img.empty() || app_ctx->rknn_ctx == 0) return -1;

    // ===================================================================
    // 🌟 完美复刻官方前处理逻辑：比例缩放 + Float32转换 + 归一化 + 补边
    // ===================================================================
    cv::Mat rgb_img;
    cv::cvtColor(src_img, rgb_img, cv::COLOR_BGR2RGB);

    int imgW = app_ctx->model_width;   // 一般为 320
    int imgH = app_ctx->model_height;  // 一般为 48

    float ratio = (float)rgb_img.cols / (float)rgb_img.rows;
    int resized_w = std::ceil(imgH * ratio);
    if (resized_w > imgW) {
        resized_w = imgW;
    }

    cv::Mat img_M;
    cv::resize(rgb_img, img_M, cv::Size(resized_w, imgH));

    // 转 Float32 并归一化 (-127.5)/127.5
    img_M.convertTo(img_M, CV_32FC3);
    img_M = (img_M - 127.5f) / 127.5f;

    // 用常数 0 补边（在浮点归一化后，0 就代表灰底）
    if (resized_w < imgW) {
        cv::copyMakeBorder(img_M, img_M, 0, 0, 0, imgW - resized_w, cv::BORDER_CONSTANT, cv::Scalar(0.0f, 0.0f, 0.0f));
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32; // 🌟 必须传入浮点张量
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = imgW * imgH * app_ctx->model_channel * sizeof(float);
    inputs[0].buf = (void*)img_M.data;

    int ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) return -1;

    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) return -1;

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 0;

    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) return -1;

    // ===================================================================
    // 🌟 完美复刻官方后处理逻辑：利用指针与算法库安全解析
    // ===================================================================
    float* out_data = (float*)outputs[0].buf;
    
    // 官方使用 dims[2] 作为通道数步长，容错率最高
    int out_channel = app_ctx->output_attrs[0].dims[2]; 
    int out_seq_len = imgW / 8; // 例如 320 / 8 = 40
    
    std::string str_res = "";
    float score = 0.f;
    int argmax_idx;
    int last_index = 0;
    int count = 0;

    // 完全复制官方 postprocess.cc 的安全遍历手法
    for (int n = 0; n < out_seq_len; n++) {
        float* start_ptr = &out_data[n * out_channel];
        float* end_ptr = &out_data[(n + 1) * out_channel];

        argmax_idx = int(std::distance(start_ptr, std::max_element(start_ptr, end_ptr)));
        float max_value = *std::max_element(start_ptr, end_ptr);

        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
            score += max_value;
            count += 1;
            if (argmax_idx < (int)app_ctx->dict.size()) {
                str_res += app_ctx->dict[argmax_idx];
            }
        }
        last_index = argmax_idx;
    }

    score /= (count + 1e-6);
    if (count == 0 || std::isnan(score)) {
        score = 0.f;
    }

    out_result->text = str_res;
    out_result->score = score;

    rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
    return 0;
}