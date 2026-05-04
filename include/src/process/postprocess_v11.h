#ifndef RK3588_DEMO_POSTPROCESS_V11_H
#define RK3588_DEMO_POSTPROCESS_V11_H

#include <stdint.h>
#include <vector>

namespace yolov11
{
    // Float version
    int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects);
    // Int8 version
    int GetConvDetectionResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects);
}

#endif // RK3588_DEMO_POSTPROCESS_V11_H 