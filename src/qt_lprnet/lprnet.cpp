#include "lprnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 车牌字典
const std::vector<std::string> plate_code{
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "I", "O", "-"};

// 替代官方的 file_utils，用标准 C 库自己写个 10 行的读取函数
static unsigned char* load_model_data(const char* filename, int* model_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return nullptr;
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* data = (unsigned char*)malloc(size);
    if (data) fread(data, 1, size, fp);
    fclose(fp);
    *model_size = size;
    return data;
}

int init_lprnet_model(const char *model_path, rknn_app_context_t *app_ctx) {
    int ret, model_len = 0;
    unsigned char *model_data = load_model_data(model_path, &model_len);
    if (model_data == NULL) return -1;

    rknn_context ctx = 0;
    ret = rknn_init(&ctx, model_data, model_len, 0, NULL);
    free(model_data);
    if (ret < 0) return -1;

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    } else {
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    return 0;
}

int release_lprnet_model(rknn_app_context_t *app_ctx) {
    if (app_ctx->input_attrs) free(app_ctx->input_attrs);
    if (app_ctx->output_attrs) free(app_ctx->output_attrs);
    if (app_ctx->rknn_ctx != 0) rknn_destroy(app_ctx->rknn_ctx);
    return 0;
}

// 极简版推理逻辑：直接塞入 cv::Mat
int inference_lprnet_model(rknn_app_context_t *app_ctx, const cv::Mat& src_img, lprnet_result *out_result) {
    int ret;
    rknn_input inputs[1];
    rknn_output outputs[1];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = (void*)src_img.data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) return -1;

    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) return -1;

    outputs[0].want_float = 1;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) return ret;

    std::vector<int> no_repeat_blank_label{};
    float prebs[OUT_COLS];
    int pre_c;
    for (int x = 0; x < OUT_COLS; x++) {
        float *ptr = (float *)outputs[0].buf;
        float preb[OUT_ROWS];
        for (int y = 0; y < OUT_ROWS; y++) {
            preb[y] = ptr[x];
            ptr += OUT_COLS;
        }
        int max_num_index = std::max_element(preb, preb + OUT_ROWS) - preb;
        prebs[x] = max_num_index;
    }

    pre_c = prebs[0];
    if (pre_c != OUT_ROWS - 1) no_repeat_blank_label.push_back(pre_c);
    
    for (int value : prebs) {
        if (value == OUT_ROWS - 1 || value == pre_c) continue;
        no_repeat_blank_label.push_back(value);
        pre_c = value;
    }

    out_result->plate_name.clear();
    for (int hh : no_repeat_blank_label) {
        out_result->plate_name += plate_code[hh];
    }

    rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
    return 0;
}