#pragma once

#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_mpi.h>
#include <opencv2/opencv.hpp>
#include <functional>

class MPPEncoder {
public:
    using EncodeFrameCallback = std::function<void(const uint8_t*, size_t, int, bool)>;

    MPPEncoder();
    ~MPPEncoder();

    void setEncodeCallback(EncodeFrameCallback callback);
    void init(int width, int height, int fps);
    void encodeFrame(const cv::Mat& bgr_frame, int frame_count);
    void deinit();

private:
    MppCtx mpp_ctx_;
    MppApi* mpp_mpi_;
    MppBufferGroup mpp_buf_grp_;
    MppFrame mpp_frame_;
    EncodeFrameCallback encode_callback_;

    int width_;
    int height_;
    int align_width_;
    int align_height_;
    int fps_;

    void configureEncoder();
};