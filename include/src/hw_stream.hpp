#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <exception>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavcodec/bsf.h>
}

#include "rk_mpi.h"
#include "mpp_buffer.h"
#include "mpp_frame.h"
#include "mpp_packet.h"

class HwCameraStream {
private:
    std::string stream_url;
    int stream_id;
    std::atomic<bool> is_running;
    std::thread worker_thread;
    
    AVFormatContext* format_ctx = nullptr;
    int video_stream_index = -1;

    MppCtx mpp_ctx = nullptr;
    MppApi* mpp_mpi = nullptr;

    cv::Mat latest_yuv_frame; 
    std::mutex frame_mutex;
    bool has_new_frame = false;

    cv::Rect cam_roi;
    bool has_roi = false;
    
    bool wait_for_keyframe = true;
    bool first_frame_decoded = false;
    
    uint64_t mpp_decode_counter = 0;

    // ==========================================================
    // 🚀【防碎片核武器】：环形内存池，彻底消灭 malloc/free！
    // ==========================================================
    cv::Mat yuv_pool[3];
    int yuv_write_idx = 0;

    cv::Mat bgr_pool[10];
    int bgr_write_idx = 0;

    void process_mpp_frame_to_yuv(MppFrame mpp_frame) {
        mpp_decode_counter++;
        
        if (stream_id != 0 && mpp_decode_counter % 3 != 0) {
            return; 
        }

        int width = mpp_frame_get_width(mpp_frame);
        int height = mpp_frame_get_height(mpp_frame);
        int hor_stride = mpp_frame_get_hor_stride(mpp_frame);
        int ver_stride = mpp_frame_get_ver_stride(mpp_frame);
        
        if (width <= 0 || height <= 0 || hor_stride <= 0 || ver_stride <= 0) return;

        MppBuffer mpp_buf = mpp_frame_get_buffer(mpp_frame);
        if (!mpp_buf) return;

        void* ptr = mpp_buffer_get_ptr(mpp_buf);
        if (!ptr) return;

        // 🚀 使用预分配内存池，永远不分配新内存
        if (yuv_pool[yuv_write_idx].empty() || yuv_pool[yuv_write_idx].cols != width || yuv_pool[yuv_write_idx].rows != height * 3 / 2) {
            yuv_pool[yuv_write_idx] = cv::Mat(height * 3 / 2, width, CV_8UC1);
        }

        uint8_t* src_y = (uint8_t*)ptr;
        uint8_t* src_uv = src_y + hor_stride * ver_stride; 
        uint8_t* dst_y = yuv_pool[yuv_write_idx].data;
        uint8_t* dst_uv = yuv_pool[yuv_write_idx].data + width * height;

        for (int r = 0; r < height; ++r) {
            memcpy(dst_y + r * width, src_y + r * hor_stride, width);
        }
        for (int r = 0; r < height / 2; ++r) {
            memcpy(dst_uv + r * width, src_uv + r * hor_stride, width);
        }

        std::lock_guard<std::mutex> lock(frame_mutex);
        latest_yuv_frame = yuv_pool[yuv_write_idx]; 
        has_new_frame = true;
        yuv_write_idx = (yuv_write_idx + 1) % 3;
    }

    void stream_worker() {
        AVDictionary* options = nullptr;
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
        av_dict_set(&options, "stimeout", "5000000", 0); 
        // 🚀 强制修改：FFmpeg 接收缓冲从 100MB 砍到 2MB，最大延迟 0.5s
        av_dict_set(&options, "buffer_size", "2097152", 0);
        av_dict_set(&options, "max_delay", "500000", 0);            
        av_dict_set(&options, "reorder_queue_size", "1024", 0); 
        av_dict_set(&options, "fflags", "discardcorrupt", 0); 

        if (avformat_open_input(&format_ctx, stream_url.c_str(), nullptr, &options) != 0) return;
        avformat_find_stream_info(format_ctx, nullptr);
        
        for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
            if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index = i;
                break;
            }
        }

        if (video_stream_index == -1) return;

        MppCodingType mpp_type = MPP_VIDEO_CodingAVC; 
        if (format_ctx->streams[video_stream_index]->codecpar->codec_id == AV_CODEC_ID_HEVC) {
            mpp_type = MPP_VIDEO_CodingHEVC; 
        }

        mpp_create(&mpp_ctx, &mpp_mpi);
        mpp_init(mpp_ctx, MPP_CTX_DEC, mpp_type);
        RK_U32 need_split = 1;
        mpp_mpi->control(mpp_ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);

        RK_U32 fast_mode = 0;
        mpp_mpi->control(mpp_ctx, MPP_DEC_SET_PARSER_FAST_MODE, &fast_mode);

        const AVBitStreamFilter *bsf = av_bsf_get_by_name(mpp_type == MPP_VIDEO_CodingHEVC ? "hevc_mp4toannexb" : "h264_mp4toannexb");
        AVBSFContext *bsf_ctx = nullptr;
        if (bsf) {
            av_bsf_alloc(bsf, &bsf_ctx);
            avcodec_parameters_copy(bsf_ctx->par_in, format_ctx->streams[video_stream_index]->codecpar);
            av_bsf_init(bsf_ctx);
        }

        uint8_t *extra_data = nullptr;
        int extra_size = 0;
        
        if (bsf_ctx && bsf_ctx->par_out->extradata_size > 0) {
            extra_data = bsf_ctx->par_out->extradata;
            extra_size = bsf_ctx->par_out->extradata_size;
        } 
        else if (format_ctx->streams[video_stream_index]->codecpar->extradata_size > 0) {
            extra_data = format_ctx->streams[video_stream_index]->codecpar->extradata;
            extra_size = format_ctx->streams[video_stream_index]->codecpar->extradata_size;
        }

        if (extra_size > 0) {
            MppPacket extra_pkt = nullptr;
            mpp_packet_init(&extra_pkt, extra_data, extra_size);
            mpp_packet_set_extra_data(extra_pkt); 
            mpp_mpi->decode_put_packet(mpp_ctx, extra_pkt);
            mpp_packet_deinit(&extra_pkt);
            printf("[Channel %d] 成功向硬件解码器注入标准的 Annex-B SPS/PPS 头 (Size: %d).\n", stream_id, extra_size);
        }

        AVPacket* pkt = av_packet_alloc();
        AVPacket* filtered_pkt = av_packet_alloc();

        auto decode_mpp_packet = [&](AVPacket* p) {
            if (wait_for_keyframe) {
                if (p->flags & AV_PKT_FLAG_KEY) wait_for_keyframe = false;
                else return;
            }

            MppPacket mpp_pkt = nullptr;
            mpp_packet_init(&mpp_pkt, p->data, p->size);
            mpp_packet_set_pts(mpp_pkt, p->pts);
            
            mpp_mpi->decode_put_packet(mpp_ctx, mpp_pkt);
            mpp_packet_deinit(&mpp_pkt);

            while (is_running) {
                MppFrame mpp_frame = nullptr;
                RK_S32 ret = mpp_mpi->decode_get_frame(mpp_ctx, &mpp_frame);
                if (ret != MPP_OK || mpp_frame == nullptr) break;

                if (mpp_frame_get_info_change(mpp_frame)) {
                    mpp_mpi->control(mpp_ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
                } 
                else if (!mpp_frame_get_discard(mpp_frame)) {
                    if (!first_frame_decoded) {
                        if (mpp_frame_get_errinfo(mpp_frame) == 0) {
                            first_frame_decoded = true;
                            if (!has_new_frame) process_mpp_frame_to_yuv(mpp_frame);
                        }
                    } else {
                        if (!has_new_frame) process_mpp_frame_to_yuv(mpp_frame);
                    }
                }
                mpp_frame_deinit(&mpp_frame);
            }
        };

        while (is_running) {
            if (av_read_frame(format_ctx, pkt) >= 0) {
                if (pkt->stream_index == video_stream_index) {
                    if (bsf_ctx) {
                        if (av_bsf_send_packet(bsf_ctx, pkt) == 0) {
                            while (av_bsf_receive_packet(bsf_ctx, filtered_pkt) == 0) {
                                decode_mpp_packet(filtered_pkt);
                                av_packet_unref(filtered_pkt);
                            }
                        }
                    } else {
                        decode_mpp_packet(pkt);
                    }
                }
                av_packet_unref(pkt);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }

        av_packet_free(&pkt);
        av_packet_free(&filtered_pkt);
        if (bsf_ctx) av_bsf_free(&bsf_ctx);

        if (mpp_ctx && mpp_mpi) { 
            mpp_mpi->reset(mpp_ctx); 
            std::this_thread::sleep_for(std::chrono::milliseconds(20)); 
            mpp_destroy(mpp_ctx); 
            mpp_ctx = nullptr; 
            mpp_mpi = nullptr; 
        }

        if (format_ctx) { avformat_close_input(&format_ctx); }
    }

    void stream_worker_safe() noexcept {
        try { stream_worker(); } catch (...) {}
    }

public:
    HwCameraStream(int id, const std::string& url) : stream_id(id), stream_url(url), is_running(false) {}

    ~HwCameraStream() noexcept { stop(); }

    void set_roi(cv::Rect roi) {
        std::lock_guard<std::mutex> lock(frame_mutex);
        cam_roi = roi;
        has_roi = true;
    }

    void start() {
        is_running = true;
        wait_for_keyframe = true;
        first_frame_decoded = false;
        mpp_decode_counter = 0;
        worker_thread = std::thread(&HwCameraStream::stream_worker_safe, this);
    }

    void stop() noexcept {
        is_running = false;
        try { if (worker_thread.joinable()) worker_thread.join(); } catch (...) {}
    }

    bool get_latest_frame(cv::Mat& out_frame) {
        cv::Mat yuv_temp;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (!has_new_frame) return false;
            yuv_temp = latest_yuv_frame; 
            has_new_frame = false;
        }

        int width = yuv_temp.cols;
        int height = yuv_temp.rows * 2 / 3;

        // 🚀 彻底杜绝每次取帧时的 malloc 分配！
        if (bgr_pool[bgr_write_idx].empty() || bgr_pool[bgr_write_idx].cols != width || bgr_pool[bgr_write_idx].rows != height) {
            bgr_pool[bgr_write_idx] = cv::Mat(height, width, CV_8UC3);
        }

        cv::cvtColor(yuv_temp, bgr_pool[bgr_write_idx], cv::COLOR_YUV2BGR_NV12);

        cv::Rect actual_roi(0, 0, width, height);
        if (has_roi) {
            actual_roi = cam_roi;
            actual_roi.x = std::max(0, actual_roi.x);
            actual_roi.y = std::max(0, actual_roi.y);
            if (actual_roi.x + actual_roi.width > width) actual_roi.width = width - actual_roi.x;
            if (actual_roi.y + actual_roi.height > height) actual_roi.height = height - actual_roi.y;
            if (actual_roi.width <= 0 || actual_roi.height <= 0) {
                actual_roi = cv::Rect(0, 0, width, height);
            }
        }
        
        // 🚀 致命修复：干掉 .clone()，利用浅拷贝（零拷贝切片）直接返回数据！
        out_frame = bgr_pool[bgr_write_idx](actual_roi); 
        bgr_write_idx = (bgr_write_idx + 1) % 10;
        return true;
    }
};