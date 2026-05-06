#include "mainwindow.h"
#include <QDebug>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <chrono>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPainter>
#include <QPainterPath>
#include <QFontMetrics>
#include <QPen>
#include <QColor>
#include <QStringList>
#include <queue>
#include <QFileDialog>
#include <QFileInfo>
#include <QGraphicsDropShadowEffect>
#include <QAction>
#include <QMenuBar>
#include <QActionGroup>
#include <algorithm>
#include <cstring>
#include <future>
#include <cmath>

#define TYPE 'S'
#define PCI_MAP_ADDR_CMD _IOWR(TYPE, 2, int)
#define PCI_DMA_WRITE_CMD _IOWR(TYPE, 5, int)
#define PCI_READ_FROM_KERNEL_CMD _IOWR(TYPE, 6, int)
#define PCI_UMAP_ADDR_CMD _IOWR(TYPE, 7, int)

#define DMA_MAX_PACKET_SIZE 4096

typedef struct _DMA_DATA_ {
    unsigned char read_buf[DMA_MAX_PACKET_SIZE];
    unsigned char write_buf[DMA_MAX_PACKET_SIZE];
} DMA_DATA;

typedef struct _DMA_OPERATION_ {
    unsigned int current_len;
    unsigned int offset_addr;
    unsigned int cmd;
    DMA_DATA data;
} DMA_OPERATION;

void rgb565_to_rgb888_local(const uint16_t *image565, uint8_t *image888, size_t num_pixels) {
    for (size_t i = 0; i < num_pixels; ++i) {
        uint16_t pixel = image565[i];
        uint8_t r = ((pixel >> 11) & 0x1F) << 3;
        uint8_t g = ((pixel >> 5) & 0x3F) << 2;
        uint8_t b = (pixel & 0x1F) << 3;
        image888[3 * i] = r;
        image888[3 * i + 1] = g;
        image888[3 * i + 2] = b;
    }
}

struct HardwareRoiInfo {
    bool meta_valid = false;
    int count = 0;
    int seq = 0;
    std::vector<cv::Rect> rois;
};

struct HardwareRoiWord {
    int count = 0;
    int index = 0;
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    int score = 0;
    int flags = 0;
    int seq = 0;
};

static uint16_t readU16Field(const uint8_t* p, int offset, bool little_endian) {
    if (little_endian) {
        return static_cast<uint16_t>(p[offset]) |
               (static_cast<uint16_t>(p[offset + 1]) << 8);
    }
    return (static_cast<uint16_t>(p[offset]) << 8) |
           static_cast<uint16_t>(p[offset + 1]);
}

static bool parseHardwareRoiWord(const uint8_t* word, bool reversed_word,
                                 bool little_endian, HardwareRoiWord& out) {
    const int magic_off = reversed_word ? 14 : 0;
    const uint16_t magic = readU16Field(word, magic_off, little_endian);
    if (magic != 0xA55A) return false;

    out.count = word[reversed_word ? 13 : 2];
    out.index = word[reversed_word ? 12 : 3];
    out.x0 = readU16Field(word, reversed_word ? 10 : 4, little_endian);
    out.y0 = readU16Field(word, reversed_word ? 8 : 6, little_endian);
    out.x1 = readU16Field(word, reversed_word ? 6 : 8, little_endian);
    out.y1 = readU16Field(word, reversed_word ? 4 : 10, little_endian);
    out.score = word[reversed_word ? 3 : 12];
    out.flags = word[reversed_word ? 2 : 13];
    out.seq = readU16Field(word, reversed_word ? 0 : 14, little_endian);

    if (out.count < 0 || out.count > 4) return false;
    if (out.index < 0 || out.index > 3) return false;
    return true;
}

static cv::Rect clampAndExpandHardwareRoi(const HardwareRoiWord& roi,
                                          const cv::Size& bounds) {
    int x0 = std::max(0, std::min(roi.x0, bounds.width - 1));
    int y0 = std::max(0, std::min(roi.y0, bounds.height - 1));
    int x1 = std::max(0, std::min(roi.x1, bounds.width - 1));
    int y1 = std::max(0, std::min(roi.y1, bounds.height - 1));
    if (x1 < x0 || y1 < y0) return cv::Rect();

    cv::Rect rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    const int margin_x = std::max(8, rect.width / 40);
    const int margin_y = std::max(6, rect.height / 40);
    rect.x -= margin_x;
    rect.y -= margin_y;
    rect.width += margin_x * 2;
    rect.height += margin_y * 2;
    return rect & cv::Rect(0, 0, bounds.width, bounds.height);
}

static double rectIouLocal(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    const double inter_area = static_cast<double>(inter.area());
    const double union_area = static_cast<double>(a.area() + b.area()) - inter_area;
    if (union_area <= 0.0) return 0.0;
    return inter_area / union_area;
}

static HardwareRoiInfo parsePcieHardwareRois(const std::vector<uint8_t>& frame565,
                                             const cv::Size& bounds) {
    HardwareRoiInfo info;
    if (frame565.size() < 64 || bounds.width <= 0 || bounds.height <= 0) return info;

    for (int i = 0; i < 4; ++i) {
        const uint8_t* word = frame565.data() + i * 16;
        HardwareRoiWord parsed;
        bool ok = false;

        for (int rev = 0; rev < 2 && !ok; ++rev) {
            for (int le = 0; le < 2 && !ok; ++le) {
                ok = parseHardwareRoiWord(word, rev != 0, le != 0, parsed);
            }
        }
        if (!ok) continue;

        info.meta_valid = true;
        info.count = std::max(info.count, parsed.count);
        info.seq = parsed.seq;
        if (parsed.index >= parsed.count || parsed.score == 0) continue;

        cv::Rect rect = clampAndExpandHardwareRoi(parsed, bounds);
        if (rect.area() <= 64) continue;

        bool merged = false;
        for (cv::Rect& existing : info.rois) {
            if (rectIouLocal(existing, rect) > 0.70) {
                existing = existing | rect;
                merged = true;
                break;
            }
        }
        if (!merged && info.rois.size() < 4) {
            info.rois.push_back(rect);
        }
    }
    return info;
}

static void scrubPcieRoiMetadataPixels(std::vector<uint8_t>& frame565, bool meta_valid) {
    if (!meta_valid || frame565.size() < 128) return;
    memcpy(frame565.data(), frame565.data() + 64, 64);
}

// =========================================================================
// ?? 核心白名单过滤逻辑：验证字符是否为合法车牌字符
// =========================================================================
static bool isValidPlateChar(const QString& str) {
    static const QStringList valid_provinces =
        QStringLiteral("\u4eac \u6caa \u6d25 \u6e1d \u5180 \u664b \u8499 \u8fbd \u5409 \u9ed1 "
                       "\u82cf \u6d59 \u7696 \u95fd \u8d63 \u9c81 \u8c6b \u9102 \u6e58 \u7ca4 "
                       "\u6842 \u743c \u5ddd \u8d35 \u4e91 \u85cf \u9655 \u7518 \u9752 \u5b81 "
                       "\u65b0 \u6e2f \u6fb3 \u53f0 \u4f7f \u9886 \u5b66 \u8b66 \u5e94 \u6025").split(QLatin1Char(' '));

    if (valid_provinces.contains(str)) {
        return true;
    }

    if (str.length() == 1) {
        QChar c = str.at(0);
        if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z')) {
            return true;
        }
    }
    return false;
}

// 清洗 PP-OCR 原始输出结果
QString cleanPPOCRResult(const QString& raw_text) {
    static const QStringList valid_provinces =
        QStringLiteral("\u4eac \u6caa \u6d25 \u6e1d \u5180 \u664b \u8499 \u8fbd \u5409 \u9ed1 "
                       "\u82cf \u6d59 \u7696 \u95fd \u8d63 \u9c81 \u8c6b \u9102 \u6e58 \u7ca4 "
                       "\u6842 \u743c \u5ddd \u8d35 \u4e91 \u85cf \u9655 \u7518 \u9752 \u5b81 "
                       "\u65b0 \u6e2f \u6fb3 \u53f0 \u4f7f \u9886 \u5b66 \u8b66 \u5e94 \u6025").split(QLatin1Char(' '));

    QString clean_text = "";
    for (int i = 0; i < raw_text.length(); i++) {
        QString single_char = raw_text.mid(i, 1).toUpper();
        QChar c = single_char.at(0);
        if (valid_provinces.contains(single_char) || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z')) {
            clean_text += single_char;
        }
    }
    return clean_text;
}

static bool isUsefulRecognitionText(const QString& text) {
    QString normalized = text.trimmed();
    if (normalized.isEmpty()) return false;
    if (normalized.contains(QStringLiteral("\u5931\u8d25")) || normalized.contains(QStringLiteral("\u9519\u8bef")) ||
        normalized.contains(QStringLiteral("\u672a\u5c31\u7eea")) || normalized.contains(QStringLiteral("\u672a\u77e5")) ||
        normalized.contains(QStringLiteral("\u8bc6\u522b\u4e2d")) || normalized.contains(QStringLiteral("\u7b49\u5f85"))) {
        return false;
    }
    return true;
}

static QString chooseVotedPlateText(const QStringList& votes, const QString& fallback) {
    QString best_text;
    int best_count = 0;

    for (int i = votes.size() - 1; i >= 0; --i) {
        const QString candidate = votes.at(i).trimmed();
        if (!isUsefulRecognitionText(candidate)) continue;

        int count = 0;
        for (const QString& vote : votes) {
            if (vote.trimmed() == candidate) count++;
        }

        if (count > best_count) {
            best_count = count;
            best_text = candidate;
        }
    }

    if (isUsefulRecognitionText(best_text)) return best_text;
    return fallback.trimmed();
}

static void appendPlateVote(QStringList& votes, const QString& text, int weight = 1) {
    QString normalized = text.trimmed();
    if (!isUsefulRecognitionText(normalized)) return;

    for (int i = 0; i < weight; ++i) votes.append(normalized);
    while (votes.size() > 12) votes.removeFirst();
}
// =========================================================================

InferenceThread::InferenceThread(QObject *parent)
    : QThread(parent), keep_running(false), current_input_type(INPUT_IMAGE), 
      use_yolo(true), current_rec_model(MODEL_PPOCR),
      yolo11_detector(nullptr), yolo11_thread_pool(nullptr), 
      lprnet_ready(false), ppocr_ready(false) 
{
    std::string yolo_model_path = "./weights/yolov11_rk3568.int.rknn";
    std::string lprnet_model_path = "./weights/lprnet_rk3568.rknn";
    std::string ppocr_model_path = "./weights/ppocr_rk3568.rknn";
    std::string ppocr_dict_path = "./weights/ppocr_keys_v1.txt"; 
    int numThreads = 1;

    yolo11_detector = new Yolov11Custom(); 
    if (yolo11_detector->LoadModel(yolo_model_path.c_str()) == NN_SUCCESS) {
        yolo11_thread_pool = new Yolov11ThreadPool();
        yolo11_thread_pool->setUp(yolo_model_path, numThreads);
    }

    memset(&lprnet_ctx, 0, sizeof(rknn_app_context_t));
    if (init_lprnet_model(lprnet_model_path.c_str(), &lprnet_ctx) == 0) {
        lprnet_ready = true;
    }

    memset(&ppocr_ctx, 0, sizeof(ppocr_app_context_t));
    if (init_ppocr_model(ppocr_model_path.c_str(), ppocr_dict_path.c_str(), &ppocr_ctx) == 0) {
        ppocr_ready = true;
    }
}

InferenceThread::~InferenceThread() {
    stop();
    wait();
    if (yolo11_detector) { delete yolo11_detector; yolo11_detector = nullptr; }
    if (yolo11_thread_pool) {
        yolo11_thread_pool->stopAll();
        delete yolo11_thread_pool;
        yolo11_thread_pool = nullptr;
    }
    if (lprnet_ready) {
        release_lprnet_model(&lprnet_ctx);
        lprnet_ready = false;
    }
    if (ppocr_ready) {
        release_ppocr_model(&ppocr_ctx);
        ppocr_ready = false;
    }
}

void InferenceThread::setInputImage(const QString& path) { file_path = path; current_input_type = INPUT_IMAGE; }
void InferenceThread::setInputVideo(const QString& path) { file_path = path; current_input_type = INPUT_VIDEO; }
void InferenceThread::setInputPCIe() { current_input_type = INPUT_PCIE; }
void InferenceThread::stop() { keep_running = false; }

void InferenceThread::ensureOutputDirectory() {
    QDir dir;
    if (!dir.exists("output/img")) dir.mkpath("output/img");
    if (!dir.exists("output/video")) dir.mkpath("output/video");
}

QImage InferenceThread::matToQImage(const cv::Mat& mat) {
    cv::Mat rgbMat;
    cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
    return QImage((const unsigned char*)(rgbMat.data), rgbMat.cols, rgbMat.rows, rgbMat.step, QImage::Format_RGB888).copy();
}

QString InferenceThread::detectPlateColor(const cv::Mat& plate_img) {
    if (plate_img.empty()) return "Unknown";
    cv::Mat hsv, mask_b, mask_y, mask_g, mask_w, mask_black, mask_neutral_light, mask_dark;
    cv::cvtColor(plate_img, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(100, 50, 50), cv::Scalar(124, 255, 255), mask_b);
    cv::inRange(hsv, cv::Scalar(15, 50, 50), cv::Scalar(34, 255, 255), mask_y);
    cv::inRange(hsv, cv::Scalar(35, 60, 50), cv::Scalar(85, 255, 255), mask_g);
    cv::inRange(hsv, cv::Scalar(0, 0, 150), cv::Scalar(179, 65, 255), mask_w);
    cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(179, 180, 85), mask_black);
    cv::inRange(hsv, cv::Scalar(0, 0, 90), cv::Scalar(179, 75, 255), mask_neutral_light);
    cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(179, 220, 100), mask_dark);

    int b_pts = cv::countNonZero(mask_b);
    int y_pts = cv::countNonZero(mask_y);
    int g_pts = cv::countNonZero(mask_g);
    int w_pts = cv::countNonZero(mask_w);
    int black_pts = cv::countNonZero(mask_black);
    int neutral_light_pts = cv::countNonZero(mask_neutral_light);
    int dark_pts = cv::countNonZero(mask_dark);
    int chroma_max_pts = std::max({b_pts, y_pts, g_pts});
    const double area = static_cast<double>(std::max(1, plate_img.cols * plate_img.rows));
    const double chroma_ratio = chroma_max_pts / area;
    const double white_ratio = w_pts / area;
    const double black_ratio = black_pts / area;
    const double neutral_light_ratio = neutral_light_pts / area;
    const double dark_ratio = dark_pts / area;

    if (neutral_light_ratio >= 0.45 && neutral_light_ratio >= chroma_ratio * 1.50) {
        return QStringLiteral("\u767d\u8272");
    }
    if (dark_ratio >= 0.45 &&
        dark_ratio >= neutral_light_ratio * 0.85 &&
        chroma_ratio < 0.25) {
        return QStringLiteral("\u9ed1\u8272");
    }
    if (chroma_max_pts >= 50 && chroma_ratio >= 0.08) {
        if (chroma_max_pts == b_pts) return QStringLiteral("\u84dd\u8272");
        if (chroma_max_pts == y_pts) return QStringLiteral("\u9ec4\u8272");
        return QStringLiteral("\u7eff\u8272");
    }
    if (white_ratio >= 0.30 || neutral_light_ratio >= 0.40) return QStringLiteral("\u767d\u8272");
    if (black_ratio >= 0.25 || dark_ratio >= 0.40) return QStringLiteral("\u9ed1\u8272");
    return "Unknown";
}

static QString refinePlateColorByText(const QString& plate_text, const QString& detected_color) {
    if (plate_text.contains(QStringLiteral("\u8b66"))) {
        return QStringLiteral("\u767d\u8272");
    }
    if (plate_text.contains(QStringLiteral("\u4f7f")) ||
        plate_text.contains(QStringLiteral("\u9886"))) {
        return QStringLiteral("\u9ed1\u8272");
    }
    return detected_color;
}

static QString buildPlateLabelText(const QString& confidence_text,
                                   const QString& plate_text,
                                   const QString& plate_color) {
    const QString refined_color = refinePlateColorByText(plate_text, plate_color);
    return QString("[%1] %2 (%3)").arg(confidence_text).arg(plate_text).arg(refined_color);
}

QString InferenceThread::recognizeText(const cv::Mat& plate_img) {
    if (plate_img.empty()) return QStringLiteral("\u8bc6\u522b\u5931\u8d25");

    const RecognitionModel rec_model = current_rec_model.load();
    if (rec_model == MODEL_LPRNET) {
        return recognizeWithLprNet(plate_img);
    } 
    else if (rec_model == MODEL_PPOCR) {
        return recognizeWithPPOCR(plate_img);
    }
    else if (rec_model == MODEL_FUSION) {
        QString lpr_text = recognizeWithLprNet(plate_img);
        QString ppocr_text = recognizeWithPPOCR(plate_img);
        QStringList votes;
        appendPlateVote(votes, lpr_text, 1);
        appendPlateVote(votes, ppocr_text, 2);

        if (isUsefulRecognitionText(ppocr_text)) return chooseVotedPlateText(votes, ppocr_text);
        if (isUsefulRecognitionText(lpr_text)) return lpr_text;
        return QStringLiteral("\u8bc6\u522b\u5931\u8d25");
    }
    return QStringLiteral("\u672a\u77e5");
}

QString InferenceThread::recognizeWithLprNet(const cv::Mat& plate_img) {
    if (plate_img.empty()) return QStringLiteral("\u8bc6\u522b\u5931\u8d25");
    if (!lprnet_ready) return QStringLiteral("LPRNet\u672a\u5c31\u7eea");

    cv::Mat resized;
    cv::resize(plate_img, resized, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    lprnet_result result;
    if (inference_lprnet_model(&lprnet_ctx, resized, &result) != 0) return QStringLiteral("\u63a8\u7406\u9519\u8bef");
    return QString::fromStdString(result.plate_name).trimmed();
}

QString InferenceThread::recognizeWithPPOCR(const cv::Mat& plate_img) {
    if (plate_img.empty()) return QStringLiteral("\u8bc6\u522b\u5931\u8d25");
    if (!ppocr_ready) return QStringLiteral("PPOCR\u672a\u5c31\u7eea");

    ppocr_result result;
    if (inference_ppocr_model(&ppocr_ctx, plate_img, &result) != 0) return QStringLiteral("\u63a8\u7406\u9519\u8bef");

    QString raw_text = QString::fromStdString(result.text);
    return cleanPPOCRResult(raw_text);
}

void InferenceThread::drawChineseTextAndBox(cv::Mat& img, const cv::Rect& box, const QString& text, const cv::Scalar& color) {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    QPainter painter(&qimg);
    painter.setRenderHint(QPainter::Antialiasing);
    
    QColor qColor(color[2], color[1], color[0]); 
    painter.setPen(QPen(qColor, 3));
    painter.drawRect(box.x, box.y, box.width, box.height);

    QFont font("Microsoft YaHei");
    font.setPixelSize(16);
    font.setBold(true);
    painter.setFont(font);
    QFontMetrics fm(font);
    
    int w = fm.horizontalAdvance(text), h = fm.height();
    int text_y = box.y;
    if (text_y < h + 12) text_y = h + 12; 

    painter.setBrush(QColor(0, 255, 0));
    painter.setPen(Qt::NoPen);
    painter.drawRect(box.x, text_y - h - 12, w + 20, h + 12);
    
    painter.setBrush(Qt::NoBrush);
    painter.setPen(Qt::NoPen);
    const int text_x = box.x + 10;
    const int baseline_y = text_y - 7;
    QPainterPath textPath;
    textPath.addText(text_x, baseline_y, font, text);
    painter.fillPath(textPath, Qt::black);
    painter.end();
    cv::cvtColor(rgb, img, cv::COLOR_RGB2BGR);
}

void InferenceThread::run() {
    if (!yolo11_detector || !yolo11_thread_pool || (!lprnet_ready && !ppocr_ready)) {
        emit showMessage("Engine Initialization Failed");
        return;
    }
    keep_running = true;
    ensureOutputDirectory();

    if (current_input_type == INPUT_IMAGE) processSingleImage();
    else if (current_input_type == INPUT_PCIE) processPCIeStream();
    else processStream();
    
    emit showMessage("System Ready - Awaiting Stream");
}

void InferenceThread::processPCIeStream() {
    std::cout << "\n[PCIe] 尝试打开设备节点..." << std::endl;
    int fd = open("/dev/pango_pci_driver", O_RDWR);
    if (fd < 0) fd = open("/dev/xdma0_c2h_0", O_RDWR);
    if (fd < 0) {
        emit showMessage("ERROR: PCIe Device Unreachable");
        return;
    }
    std::cout << "[PCIe] 设备节点打开成功!" << std::endl;

    int width = 1280, height = 720;
    std::vector<uint8_t> image_buf_temp(width * height * 2);
    std::vector<uint8_t> image_buf_888(width * height * 3);

    DMA_OPERATION dma_operation; 

    int job_cnt = 0, result_cnt = 0, numThreads = 1;
    struct PendingYoloFrame {
        cv::Mat roi_frame;
        cv::Rect roi;
        int track_id = -1;
        int submit_frame = 0;
        int64 submit_tick = 0;
        double prepare_ms = 0.0;
        double scale_x = 1.0;
        double scale_y = 1.0;
    };
    std::queue<PendingYoloFrame> frame_queue;
    int64 fps_start_time = cv::getTickCount();
    int fps_frame_count = 0;
    double current_fps = 0.0;
    const int max_cached_plate_age = 30;
    const double stable_box_iou_threshold = 0.70;
    const double cache_merge_iou_threshold = 0.30;
    const double ppocr_min_interval_sec = 1.0;
    const int max_roi_tracks = 6;
    const int roi_confirm_hits = 2;
    const int roi_miss_hold = 5;
    const int yolo_refresh_interval = 12;
    const int yolo_retry_interval = 5;
    const int yolo_failure_cooldown = 12;
    const int max_yolo_result_age = 60;
    const cv::Size yolo_fullframe_submit_size(640, 360);
    const float yolo_confidence_threshold = 0.20f;
    const int max_yolo_boxes_per_result = 2;
    int pcie_frame_index = 0;
    int cached_plate_age = max_cached_plate_age + 1;
    double profile_yolo_prepare_sum = 0.0;
    double profile_yolo_async_sum = 0.0;
    double profile_ocr_sum = 0.0;
    int profile_yolo_count = 0;

    struct CachedPlate {
        cv::Rect box;
        QString plate_text;
        QString plate_color;
        QString label;
        int64 last_ocr_tick;
        QString lpr_text;
        QString ppocr_text;
        QStringList plate_votes;
    };
    std::vector<CachedPlate> cached_plates;

    struct OcrJob {
        cv::Mat plate_crop;
        cv::Rect box;
        QString confidence_text;
        RecognitionModel rec_model = MODEL_LPRNET;
        QString fallback_text;
        QString fallback_color;
        QString lpr_text;
        QString ppocr_text;
        QStringList plate_votes;
    };

    struct OcrResult {
        bool valid = false;
        cv::Rect box;
        QString plate_text;
        QString plate_color;
        QString confidence_text;
        int64 last_ocr_tick = 0;
        QString lpr_text;
        QString ppocr_text;
        QStringList plate_votes;
        double ocr_ms = 0.0;
    };

    std::future<OcrResult> ocr_future;
    bool ocr_future_active = false;
    cv::Rect active_ocr_box;
    OcrJob queued_ocr_job;
    bool queued_ocr_valid = false;
    double profile_async_ocr_sum = 0.0;
    int profile_async_ocr_count = 0;

    struct RoiTrack {
        int id = 0;
        cv::Rect roi;
        cv::Rect last_yolo_roi;
        cv::Rect yolo_box;
        int hits = 0;
        int misses = 0;
        int last_seen_frame = 0;
        int last_yolo_frame = -1000;
        int cooldown_until_frame = 0;
        bool pending_yolo = false;
        bool yolo_confirmed = false;
        int yolo_failures = 0;
    };
    std::vector<RoiTrack> roi_tracks;
    int next_roi_track_id = 1;

    auto tickToMs = [](int64 start_tick, int64 end_tick) -> double {
        if (start_tick <= 0 || end_tick < start_tick) return 0.0;
        return (end_tick - start_tick) * 1000.0 / cv::getTickFrequency();
    };

    auto calcBoxIou = [](const cv::Rect& a, const cv::Rect& b) -> double {
        cv::Rect inter = a & b;
        double inter_area = static_cast<double>(inter.area());
        double union_area = static_cast<double>(a.area() + b.area()) - inter_area;
        if (union_area <= 0.0) return 0.0;
        return inter_area / union_area;
    };

    auto secondsBetweenTicks = [&](int64 start_tick, int64 end_tick) -> double {
        if (start_tick <= 0 || end_tick <= start_tick) return ppocr_min_interval_sec;
        return (end_tick - start_tick) / cv::getTickFrequency();
    };

    auto centerClose = [](const cv::Rect& a, const cv::Rect& b) -> bool {
        const int ax = a.x + a.width / 2;
        const int ay = a.y + a.height / 2;
        const int bx = b.x + b.width / 2;
        const int by = b.y + b.height / 2;
        int dx = ax - bx;
        int dy = ay - by;
        if (dx < 0) dx = -dx;
        if (dy < 0) dy = -dy;
        return dx <= std::max(80, std::max(a.width, b.width) / 2) &&
               dy <= std::max(48, std::max(a.height, b.height) / 2);
    };

    auto verticalOverlapRatio = [](const cv::Rect& a, const cv::Rect& b) -> double {
        const int top = std::max(a.y, b.y);
        const int bottom = std::min(a.y + a.height, b.y + b.height);
        const int overlap = std::max(0, bottom - top);
        const int denom = std::max(1, std::min(a.height, b.height));
        return static_cast<double>(overlap) / static_cast<double>(denom);
    };

    auto shouldMergeRois = [&](const cv::Rect& a, const cv::Rect& b) -> bool {
        const double iou = calcBoxIou(a, b);
        if (iou >= 0.08) return true;

        const int a_right = a.x + a.width;
        const int b_right = b.x + b.width;
        const int horizontal_gap = std::max(0, std::max(a.x, b.x) - std::min(a_right, b_right));
        const int max_w = std::max(a.width, b.width);
        const bool same_band = verticalOverlapRatio(a, b) >= 0.35;
        const bool near_in_x = horizontal_gap <= std::max(96, max_w / 3);
        return same_band && near_in_x;
    };

    auto mergeHardwareRois = [&](const std::vector<cv::Rect>& hardware_rois,
                                 const cv::Size& bounds) -> std::vector<cv::Rect> {
        std::vector<cv::Rect> merged;
        const cv::Rect frame_bounds(0, 0, bounds.width, bounds.height);

        for (const cv::Rect& raw : hardware_rois) {
            cv::Rect roi = raw & frame_bounds;
            if (roi.area() <= 0) continue;

            bool changed = true;
            while (changed) {
                changed = false;
                for (int i = 0; i < static_cast<int>(merged.size()); ++i) {
                    if (!shouldMergeRois(merged[i], roi)) continue;
                    cv::Rect union_roi = (merged[i] | roi) & frame_bounds;
                    if (union_roi.area() > bounds.width * bounds.height / 2) continue;
                    roi = union_roi;
                    merged.erase(merged.begin() + i);
                    changed = true;
                    break;
                }
            }
            merged.push_back(roi);
        }

        std::sort(merged.begin(), merged.end(), [](const cv::Rect& a, const cv::Rect& b) {
            return a.area() > b.area();
        });
        if (merged.size() > 4) merged.resize(4);
        return merged;
    };

    auto roiPriorityScore = [](const cv::Rect& roi) -> int {
        if (roi.area() <= 0) return -100000;
        const double aspect = static_cast<double>(roi.width) / static_cast<double>(std::max(1, roi.height));
        int score = 0;
        if (aspect >= 1.6 && aspect <= 8.5) score += 220;
        else if (aspect >= 1.1 && aspect <= 12.0) score += 80;
        else score -= 180;

        if (roi.width >= 80 && roi.height >= 24) score += 120;
        if (roi.width > 760 || roi.height > 360) score -= 180;
        if (roi.area() > 1280 * 720 / 3) score -= 220;
        return score;
    };

    auto smoothRect = [](const cv::Rect& old_rect, const cv::Rect& new_rect) -> cv::Rect {
        if (old_rect.area() <= 0) return new_rect;
        const int x0 = (old_rect.x * 3 + new_rect.x) / 4;
        const int y0 = (old_rect.y * 3 + new_rect.y) / 4;
        const int old_x1 = old_rect.x + old_rect.width;
        const int old_y1 = old_rect.y + old_rect.height;
        const int new_x1 = new_rect.x + new_rect.width;
        const int new_y1 = new_rect.y + new_rect.height;
        const int x1 = (old_x1 * 3 + new_x1) / 4;
        const int y1 = (old_y1 * 3 + new_y1) / 4;
        return cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
    };

    auto matchRoiTrack = [&](const cv::Rect& roi) -> int {
        int best_idx = -1;
        double best_iou = 0.0;
        for (int i = 0; i < static_cast<int>(roi_tracks.size()); ++i) {
            double iou = calcBoxIou(roi, roi_tracks[i].roi);
            if (iou > best_iou) {
                best_iou = iou;
                best_idx = i;
            }
        }
        if (best_idx >= 0 && best_iou >= 0.18) return best_idx;

        for (int i = 0; i < static_cast<int>(roi_tracks.size()); ++i) {
            if (centerClose(roi, roi_tracks[i].roi)) return i;
        }
        return -1;
    };

    auto updateRoiTracks = [&](const std::vector<cv::Rect>& hardware_rois, int frame_index) {
        std::vector<bool> matched(roi_tracks.size(), false);

        for (const cv::Rect& raw_roi : hardware_rois) {
            if (raw_roi.area() <= 0) continue;
            int match_idx = matchRoiTrack(raw_roi);
            if (match_idx >= 0) {
                RoiTrack& track = roi_tracks[match_idx];
                track.roi = smoothRect(track.roi, raw_roi);
                track.hits = std::min(track.hits + 1, 20);
                track.misses = 0;
                track.last_seen_frame = frame_index;
                if (match_idx < static_cast<int>(matched.size())) matched[match_idx] = true;
            } else if (static_cast<int>(roi_tracks.size()) < max_roi_tracks) {
                RoiTrack track;
                track.id = next_roi_track_id++;
                track.roi = raw_roi;
                track.hits = 1;
                track.last_seen_frame = frame_index;
                roi_tracks.push_back(track);
                matched.push_back(true);
            }
        }

        for (int i = 0; i < static_cast<int>(roi_tracks.size()); ++i) {
            if (i < static_cast<int>(matched.size()) && matched[i]) continue;
            roi_tracks[i].misses++;
        }

        roi_tracks.erase(std::remove_if(roi_tracks.begin(), roi_tracks.end(),
            [&](const RoiTrack& track) {
                return !track.pending_yolo && track.misses > roi_miss_hold;
            }), roi_tracks.end());
    };

    auto selectYoloTrack = [&](int frame_index) -> int {
        int best_idx = -1;
        int best_score = -100000;

        for (int i = 0; i < static_cast<int>(roi_tracks.size()); ++i) {
            const RoiTrack& track = roi_tracks[i];
            if (track.pending_yolo) continue;
            if (track.hits < roi_confirm_hits) continue;
            if (track.misses > 1) continue;
            if (frame_index < track.cooldown_until_frame) continue;

            const bool never_run = track.last_yolo_frame < 0;
            const bool retry_unconfirmed = !track.yolo_confirmed &&
                                           (never_run || frame_index - track.last_yolo_frame >= yolo_retry_interval);
            const bool refresh_confirmed = track.yolo_confirmed &&
                                           (frame_index - track.last_yolo_frame >= yolo_refresh_interval);
            const bool roi_moved = track.yolo_confirmed &&
                                   track.last_yolo_roi.area() > 0 &&
                                   calcBoxIou(track.roi, track.last_yolo_roi) < 0.55;

            if (!retry_unconfirmed && !refresh_confirmed && !roi_moved) continue;

            int score = (track.yolo_confirmed ? 0 : 1000) +
                        roiPriorityScore(track.roi) +
                        track.hits * 20 -
                        track.misses * 60 +
                        (frame_index - track.last_yolo_frame);
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        return best_idx;
    };

    auto findRoiTrackById = [&](int track_id) -> int {
        for (int i = 0; i < static_cast<int>(roi_tracks.size()); ++i) {
            if (roi_tracks[i].id == track_id) return i;
        }
        return -1;
    };

    auto pushYoloProfile = [&](double prepare_ms, double async_ms, double ocr_ms) {
        profile_yolo_prepare_sum += prepare_ms;
        profile_yolo_async_sum += async_ms;
        profile_ocr_sum += ocr_ms;
        profile_yolo_count++;
        if (profile_yolo_count >= 30) {
            std::cout << cv::format("[YOLO_PROFILE_AVG] prepare_roi=%.2f ms, blackbox_async=%.2f ms, ocr=%.2f ms",
                                    profile_yolo_prepare_sum / profile_yolo_count,
                                    profile_yolo_async_sum / profile_yolo_count,
                                    profile_ocr_sum / profile_yolo_count)
                      << std::endl;
            profile_yolo_prepare_sum = 0.0;
            profile_yolo_async_sum = 0.0;
            profile_ocr_sum = 0.0;
            profile_yolo_count = 0;
        }
    };

    auto updateFpsOverlay = [&]() -> double {
        fps_frame_count++;
        if (fps_frame_count % 10 == 0) {
            int64 fps_end_time = cv::getTickCount();
            current_fps = 10.0 * cv::getTickFrequency() / (fps_end_time - fps_start_time);
            fps_start_time = fps_end_time;
        }
        return current_fps;
    };

    auto drawFpsOverlay = [&](cv::Mat& draw_frame) {
        double fps_value = updateFpsOverlay();
        if (fps_value > 0.0) {
            cv::putText(draw_frame, cv::format("FPS: %.1f", fps_value), cv::Point(15, 35),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }
    };

    auto drawCachedPlates = [&](cv::Mat& draw_frame) {
        if (cached_plate_age > max_cached_plate_age) return;
        for (const auto& plate : cached_plates) {
            cv::Rect safe_box = plate.box & cv::Rect(0, 0, draw_frame.cols, draw_frame.rows);
            if (safe_box.area() <= 0) continue;
            drawChineseTextAndBox(draw_frame, safe_box, plate.label, cv::Scalar(255, 255, 255));
        }
        cached_plate_age++;
    };

    auto drawHardwareRois = [&](cv::Mat& draw_frame, const std::vector<cv::Rect>& hardware_rois) {
        for (const cv::Rect& roi : hardware_rois) {
            cv::Rect safe_roi = roi & cv::Rect(0, 0, draw_frame.cols, draw_frame.rows);
            if (safe_roi.area() <= 0) continue;
            cv::rectangle(draw_frame, safe_roi, cv::Scalar(0, 200, 255), 1);
        }
    };

    auto mergeCachedPlates = [&](const std::vector<CachedPlate>& updates) {
        for (const CachedPlate& update : updates) {
            int best_idx = -1;
            double best_iou = 0.0;
            for (int i = 0; i < static_cast<int>(cached_plates.size()); ++i) {
                double iou = calcBoxIou(update.box, cached_plates[i].box);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_idx = i;
                }
            }

            if (best_idx >= 0 && best_iou >= cache_merge_iou_threshold) {
                cached_plates[best_idx] = update;
            } else {
                cached_plates.push_back(update);
            }
        }

        while (cached_plates.size() > 6) {
            cached_plates.erase(cached_plates.begin());
        }
    };

    auto buildPlateLabel = [](const QString& confidence_text,
                              const QString& plate_text,
                              const QString& plate_color) -> QString {
        return buildPlateLabelText(confidence_text, plate_text, plate_color);
    };

    auto runOcrJob = [this](OcrJob job) -> OcrResult {
        OcrResult result;
        if (job.plate_crop.empty() || job.box.area() <= 0) return result;

        const int64 ocr_tick0 = cv::getTickCount();
        result.valid = true;
        result.box = job.box;
        result.confidence_text = job.confidence_text;
        result.plate_color = detectPlateColor(job.plate_crop);
        result.lpr_text = job.lpr_text;
        result.ppocr_text = job.ppocr_text;
        result.plate_votes = job.plate_votes;
        result.plate_text = job.fallback_text;

        if (job.rec_model == MODEL_FUSION) {
            QString lpr_candidate = recognizeWithLprNet(job.plate_crop);
            if (isUsefulRecognitionText(lpr_candidate)) {
                result.lpr_text = lpr_candidate;
                appendPlateVote(result.plate_votes, lpr_candidate, 1);
            }

            QString ppocr_candidate = recognizeWithPPOCR(job.plate_crop);
            if (isUsefulRecognitionText(ppocr_candidate)) {
                result.ppocr_text = ppocr_candidate;
                appendPlateVote(result.plate_votes, ppocr_candidate, 2);
            }

            QString fallback = result.plate_text;
            if (isUsefulRecognitionText(result.ppocr_text)) fallback = result.ppocr_text;
            else if (isUsefulRecognitionText(result.lpr_text)) fallback = result.lpr_text;
            result.plate_text = chooseVotedPlateText(result.plate_votes, fallback);
        } else if (job.rec_model == MODEL_PPOCR) {
            QString ppocr_candidate = recognizeWithPPOCR(job.plate_crop);
            if (isUsefulRecognitionText(ppocr_candidate)) {
                result.ppocr_text = ppocr_candidate;
                appendPlateVote(result.plate_votes, ppocr_candidate, 2);
                result.plate_text = chooseVotedPlateText(result.plate_votes, ppocr_candidate);
            }
        } else {
            QString lpr_candidate = recognizeWithLprNet(job.plate_crop);
            if (isUsefulRecognitionText(lpr_candidate)) {
                result.lpr_text = lpr_candidate;
                appendPlateVote(result.plate_votes, lpr_candidate, 1);
                result.plate_text = chooseVotedPlateText(result.plate_votes, lpr_candidate);
            }
        }

        if (!isUsefulRecognitionText(result.plate_text)) result.plate_text = QStringLiteral("\u8bc6\u522b\u5931\u8d25");
        if (result.plate_color.isEmpty()) result.plate_color = job.fallback_color.isEmpty() ? "Unknown" : job.fallback_color;
        result.plate_color = refinePlateColorByText(result.plate_text, result.plate_color);
        result.last_ocr_tick = cv::getTickCount();
        result.ocr_ms = (result.last_ocr_tick - ocr_tick0) * 1000.0 / cv::getTickFrequency();
        return result;
    };

    auto ocrBoxMatches = [&](const cv::Rect& a, const cv::Rect& b) -> bool {
        if (a.area() <= 0 || b.area() <= 0) return false;
        return calcBoxIou(a, b) >= 0.55 || centerClose(a, b);
    };

    auto startOcrJob = [&](const OcrJob& job) {
        if (job.plate_crop.empty() || job.box.area() <= 0) return;
        active_ocr_box = job.box;
        ocr_future = std::async(std::launch::async, runOcrJob, job);
        ocr_future_active = true;
    };

    auto requestOcrJob = [&](const OcrJob& job) {
        if (job.plate_crop.empty() || job.box.area() <= 0) return;
        if (!ocr_future_active) {
            startOcrJob(job);
            return;
        }
        if (ocrBoxMatches(job.box, active_ocr_box)) return;

        queued_ocr_job = job;
        queued_ocr_valid = true;
    };

    auto applyOcrResult = [&](const OcrResult& result) {
        if (!result.valid || result.box.area() <= 0) return;

        const QString label = buildPlateLabel(result.confidence_text, result.plate_text, result.plate_color);
        std::vector<CachedPlate> updates;
        updates.push_back({result.box, result.plate_text, result.plate_color, label, result.last_ocr_tick,
                           result.lpr_text, result.ppocr_text, result.plate_votes});
        mergeCachedPlates(updates);
        cached_plate_age = 0;

        if (isUsefulRecognitionText(result.plate_text)) {
            emit recognitionReady(result.plate_text, result.plate_color, result.confidence_text);
        }

        profile_async_ocr_sum += result.ocr_ms;
        profile_async_ocr_count++;
        if (profile_async_ocr_count >= 10) {
            std::cout << cv::format("[OCR_PROFILE_AVG] async_ocr=%.2f ms",
                                    profile_async_ocr_sum / profile_async_ocr_count)
                      << std::endl;
            profile_async_ocr_sum = 0.0;
            profile_async_ocr_count = 0;
        }
    };

    auto pollOcrFuture = [&]() {
        if (!ocr_future_active) return;
        if (ocr_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) return;

        OcrResult result = ocr_future.get();
        ocr_future_active = false;
        applyOcrResult(result);

        if (queued_ocr_valid) {
            OcrJob next_job = queued_ocr_job;
            queued_ocr_valid = false;
            startOcrJob(next_job);
        }
    };

    while (keep_running || result_cnt < job_cnt) {
        pollOcrFuture();

        bool current_use_yolo = use_yolo;
        bool frame_ready = false;
        cv::Mat frame;
        std::vector<cv::Rect> hardware_rois_for_display;

        if (keep_running) {
            dma_operation.current_len = 2560 / 4;
            dma_operation.offset_addr = 0;
            memset(dma_operation.data.write_buf, 0, DMA_MAX_PACKET_SIZE);
            memset(dma_operation.data.read_buf, 0, DMA_MAX_PACKET_SIZE);

            ioctl(fd, PCI_MAP_ADDR_CMD, &dma_operation);
            for (int i = 0; i < height; i++) {
                memset(dma_operation.data.read_buf, 0, DMA_MAX_PACKET_SIZE);
                ioctl(fd, PCI_DMA_WRITE_CMD, &dma_operation);
                for (volatile int k = 0; k < 2500; k++); 
                ioctl(fd, PCI_READ_FROM_KERNEL_CMD, &dma_operation);
                memcpy(image_buf_temp.data() + i * 2560, dma_operation.data.read_buf, 2560);
            }
            ioctl(fd, PCI_UMAP_ADDR_CMD, &dma_operation);

            HardwareRoiInfo hardware_roi_info =
                parsePcieHardwareRois(image_buf_temp, cv::Size(width, height));
            hardware_rois_for_display = mergeHardwareRois(hardware_roi_info.rois, cv::Size(width, height));
            scrubPcieRoiMetadataPixels(image_buf_temp, hardware_roi_info.meta_valid);

            rgb565_to_rgb888_local((uint16_t*)image_buf_temp.data(), image_buf_888.data(), width * height);
            cv::Mat img(height, width, CV_8UC3, image_buf_888.data());
            cv::cvtColor(img, frame, cv::COLOR_RGB2BGR); 
            frame_ready = true;
            
            if (current_use_yolo) {
                bool yolo_idle = (job_cnt - result_cnt < numThreads);
                if (yolo_idle) {
                    const int64 prepare_tick0 = cv::getTickCount();
                    cv::Mat detect_frame;
                    if (frame.cols > yolo_fullframe_submit_size.width ||
                        frame.rows > yolo_fullframe_submit_size.height) {
                        cv::resize(frame, detect_frame, yolo_fullframe_submit_size, 0, 0, cv::INTER_AREA);
                    } else {
                        detect_frame = frame.clone();
                    }
                    yolo11_thread_pool->submitTask(detect_frame, job_cnt++);
                    const int64 submit_tick = cv::getTickCount();

                    PendingYoloFrame pending_job;
                    pending_job.roi_frame = detect_frame;
                    pending_job.roi = cv::Rect(0, 0, frame.cols, frame.rows);
                    pending_job.track_id = -1;
                    pending_job.submit_frame = pcie_frame_index;
                    pending_job.submit_tick = submit_tick;
                    pending_job.prepare_ms = tickToMs(prepare_tick0, submit_tick);
                    pending_job.scale_x = static_cast<double>(frame.cols) / std::max(1, detect_frame.cols);
                    pending_job.scale_y = static_cast<double>(frame.rows) / std::max(1, detect_frame.rows);
                    frame_queue.push(pending_job);
                }
            } else {
                cv::Mat draw_frame = frame.clone();
                QString plate_color = detectPlateColor(draw_frame);
                QString plate_text = recognizeText(draw_frame);
                plate_color = refinePlateColorByText(plate_text, plate_color);

                cv::Scalar box_color = cv::Scalar(255, 255, 255); 
                QString label_str = QStringLiteral("[\u5168\u56fe\u76f4\u51fa] %1 (%2)").arg(plate_text).arg(plate_color);
                cv::Rect full_img_box(5, 5, draw_frame.cols - 10, draw_frame.rows - 10);
                drawChineseTextAndBox(draw_frame, full_img_box, label_str, box_color);

                drawFpsOverlay(draw_frame);
                drawHardwareRois(draw_frame, hardware_rois_for_display);
                emit frameReady(matToQImage(draw_frame));
            }
            pcie_frame_index++;
        }

        if (result_cnt < job_cnt) {
            std::vector<Detection> objects;
            if (yolo11_thread_pool->getTargetResultNonBlock(objects, result_cnt) == NN_SUCCESS) {
                PendingYoloFrame pending_frame;
                if (!frame_queue.empty()) {
                    pending_frame = frame_queue.front();
                    frame_queue.pop();
                }
                int pending_track_idx = findRoiTrackById(pending_frame.track_id);
                if (pending_track_idx >= 0) {
                    roi_tracks[pending_track_idx].pending_yolo = false;
                }

                std::vector<CachedPlate> new_cached_plates;
                bool plate_detected = false;
                double result_ocr_ms = 0.0;
                const bool result_is_stale =
                    pending_frame.submit_frame > 0 &&
                    pcie_frame_index - pending_frame.submit_frame > max_yolo_result_age;
                if (result_is_stale) {
                    if (pending_track_idx >= 0) {
                        roi_tracks[pending_track_idx].cooldown_until_frame = pcie_frame_index + 2;
                    }
                    const double yolo_async_ms = tickToMs(pending_frame.submit_tick, cv::getTickCount());
                    pushYoloProfile(pending_frame.prepare_ms, yolo_async_ms, result_ocr_ms);
                    result_cnt++;
                    continue;
                }
                std::sort(objects.begin(), objects.end(), [](const Detection& a, const Detection& b) {
                    return a.confidence > b.confidence;
                });
                int accepted_yolo_boxes = 0;

                for (auto &obj : objects) {
                    if (obj.confidence < yolo_confidence_threshold) continue;
                    if (accepted_yolo_boxes >= max_yolo_boxes_per_result) break;
                    if (pending_frame.roi_frame.empty()) continue;
                    cv::Rect roi_bounds(0, 0, pending_frame.roi_frame.cols, pending_frame.roi_frame.rows);
                    cv::Rect roi_box = obj.box & roi_bounds;
                    if (roi_box.area() <= 0) continue;

                    const int mapped_x0 = pending_frame.roi.x +
                        static_cast<int>(std::floor(roi_box.x * pending_frame.scale_x));
                    const int mapped_y0 = pending_frame.roi.y +
                        static_cast<int>(std::floor(roi_box.y * pending_frame.scale_y));
                    const int mapped_x1 = pending_frame.roi.x +
                        static_cast<int>(std::ceil((roi_box.x + roi_box.width) * pending_frame.scale_x));
                    const int mapped_y1 = pending_frame.roi.y +
                        static_cast<int>(std::ceil((roi_box.y + roi_box.height) * pending_frame.scale_y));
                    cv::Rect safe_box(mapped_x0,
                                      mapped_y0,
                                      std::max(1, mapped_x1 - mapped_x0),
                                      std::max(1, mapped_y1 - mapped_y0));
                    safe_box &= cv::Rect(0, 0, width, height);
                    if (safe_box.area() <= 0) continue;

                    accepted_yolo_boxes++;
                    plate_detected = true;
                    int best_cached_idx = -1;
                    double best_iou = 0.0;
                    for (int i = 0; i < static_cast<int>(cached_plates.size()); ++i) {
                        double iou = calcBoxIou(safe_box, cached_plates[i].box);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_cached_idx = i;
                        }
                    }

                    const bool box_stable = best_cached_idx >= 0 && best_iou >= stable_box_iou_threshold;
                    QString plate_text;
                    QString plate_color;
                    int64 last_ocr_tick = 0;
                    QString lpr_text;
                    QString ppocr_text;
                    QStringList plate_votes;

                    if (box_stable) {
                        const CachedPlate& cached = cached_plates[best_cached_idx];
                        plate_text = cached.plate_text;
                        plate_color = cached.plate_color;
                        last_ocr_tick = cached.last_ocr_tick;
                        lpr_text = cached.lpr_text;
                        ppocr_text = cached.ppocr_text;
                        plate_votes = cached.plate_votes;
                    }

                    int64 now_tick = cv::getTickCount();
                    const RecognitionModel rec_model = current_rec_model.load();
                    const bool throttle_ocr = (rec_model == MODEL_PPOCR || rec_model == MODEL_FUSION);
                    const bool ocr_due = !throttle_ocr ||
                                         last_ocr_tick == 0 ||
                                         secondsBetweenTicks(last_ocr_tick, now_tick) >= ppocr_min_interval_sec;
                    const bool text_is_good = isUsefulRecognitionText(plate_text);
                    const bool should_run_ocr = !box_stable || ocr_due || !text_is_good;

                    bool waiting_for_ocr = false;
                    QString confidence_text = QString::number(obj.confidence, 'f', 2);
                    if (should_run_ocr) {
                        cv::Rect plate_roi = roi_box & cv::Rect(0, 0,
                                                                pending_frame.roi_frame.cols,
                                                                pending_frame.roi_frame.rows);
                        cv::Mat plate_crop;
                        if (frame_ready && !frame.empty()) {
                            cv::Rect full_plate_roi = safe_box & cv::Rect(0, 0, frame.cols, frame.rows);
                            if (full_plate_roi.area() > 0) {
                                plate_crop = frame(full_plate_roi).clone();
                            }
                        }
                        if (plate_crop.empty() && plate_roi.area() > 0) {
                            plate_crop = pending_frame.roi_frame(plate_roi).clone();
                        }
                        if (!plate_crop.empty()) {
                            OcrJob ocr_job;
                            ocr_job.plate_crop = plate_crop;
                            ocr_job.box = safe_box;
                            ocr_job.confidence_text = confidence_text;
                            ocr_job.rec_model = rec_model;
                            ocr_job.fallback_text = plate_text;
                            ocr_job.fallback_color = plate_color;
                            ocr_job.lpr_text = lpr_text;
                            ocr_job.ppocr_text = ppocr_text;
                            ocr_job.plate_votes = plate_votes;
                            requestOcrJob(ocr_job);
                            waiting_for_ocr = true;
                        }
                    }

                    if (!isUsefulRecognitionText(plate_text)) {
                        plate_text = waiting_for_ocr ? QStringLiteral("\u8bc6\u522b\u4e2d") : QStringLiteral("\u8bc6\u522b\u5931\u8d25");
                    }
                    plate_color = refinePlateColorByText(plate_text, plate_color);
                    if (plate_color.isEmpty()) {
                        plate_color = "Unknown";
                    }

                    QString label_str = buildPlateLabel(confidence_text, plate_text, plate_color);
                    if (isUsefulRecognitionText(plate_text)) {
                        emit recognitionReady(plate_text, plate_color, confidence_text);
                    }
                    new_cached_plates.push_back({safe_box, plate_text, plate_color, label_str, last_ocr_tick,
                                                 lpr_text, ppocr_text, plate_votes});
                }

                if (plate_detected) {
                    pending_track_idx = findRoiTrackById(pending_frame.track_id);
                    if (pending_track_idx >= 0) {
                        RoiTrack& track = roi_tracks[pending_track_idx];
                        track.yolo_confirmed = true;
                        track.yolo_failures = 0;
                        track.cooldown_until_frame = pcie_frame_index + 2;
                        if (!new_cached_plates.empty()) {
                            track.yolo_box = new_cached_plates.front().box;
                        }
                    }
                    mergeCachedPlates(new_cached_plates);
                    cached_plate_age = 0;
                } else {
                    pending_track_idx = findRoiTrackById(pending_frame.track_id);
                    if (pending_track_idx >= 0) {
                        RoiTrack& track = roi_tracks[pending_track_idx];
                        track.yolo_failures++;
                        if (track.yolo_failures >= 2) {
                            track.yolo_confirmed = false;
                            track.cooldown_until_frame = pcie_frame_index + yolo_failure_cooldown;
                        } else {
                            track.cooldown_until_frame = pcie_frame_index + yolo_retry_interval;
                        }
                    }
                    if (cached_plate_age > max_cached_plate_age) {
                        cached_plates.clear();
                    }
                }

                const double yolo_async_ms = tickToMs(pending_frame.submit_tick, cv::getTickCount());
                pushYoloProfile(pending_frame.prepare_ms, yolo_async_ms, result_ocr_ms);
                result_cnt++;
            } else {
                usleep(1000);
            }
        }

        pollOcrFuture();

        if (frame_ready && current_use_yolo) {
            cv::Mat draw_frame = frame.clone();
            drawHardwareRois(draw_frame, hardware_rois_for_display);
            drawCachedPlates(draw_frame);
            drawFpsOverlay(draw_frame);
            emit frameReady(matToQImage(draw_frame));
        } else if (!frame_ready) {
            usleep(1000);
        }
    }

    if (ocr_future_active) {
        OcrResult result = ocr_future.get();
        ocr_future_active = false;
        applyOcrResult(result);
    }

    close(fd);
    std::cout << "\n[PCIe] 流读取安全关闭" << std::endl;
}

void InferenceThread::processSingleImage() {
    emit showMessage("Analyzing Image Matrix...");
    cv::Mat frame = cv::imread(file_path.toStdString());
    if (frame.empty()) {
        emit showMessage("ERROR: Invalid Image Buffer");
        return;
    }

    int64 t_start = cv::getTickCount();
    bool plate_detected = false;

    if (use_yolo) {
        yolo11_thread_pool->submitTask(frame.clone(), 0);
        std::vector<Detection> objects;
        while (yolo11_thread_pool->getTargetResultNonBlock(objects, 0) != NN_SUCCESS && keep_running) {
            usleep(1000);
        }
        if (!keep_running) return;

        for (auto &obj : objects) {
            if (obj.confidence < 0.15) continue; 
            cv::Rect safe_box = obj.box & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safe_box.area() <= 0) continue;

            plate_detected = true;
            cv::Mat plate_crop = frame(safe_box);
            QString plate_color = detectPlateColor(plate_crop);
            QString plate_text = recognizeText(plate_crop); 
            plate_color = refinePlateColorByText(plate_text, plate_color);

            cv::Scalar box_color = cv::Scalar(255, 255, 255); 
            QString confidence_text = QString::number(obj.confidence, 'f', 2);
            QString label_str = buildPlateLabelText(confidence_text, plate_text, plate_color);
            emit recognitionReady(plate_text, plate_color, confidence_text);
            drawChineseTextAndBox(frame, safe_box, label_str, box_color);
        }
    }

    if (!plate_detected || !use_yolo) {
        QString plate_color = detectPlateColor(frame);
        QString plate_text = recognizeText(frame); 
        plate_color = refinePlateColorByText(plate_text, plate_color);
        cv::Scalar box_color = cv::Scalar(255, 255, 255); 
        QString prefix = use_yolo ? QStringLiteral("[YOLO\u515c\u5e95]") : QStringLiteral("[\u76f4\u51fa]");
        QString label_str = QString("%1 %2 (%3)").arg(prefix).arg(plate_text).arg(plate_color);
        emit recognitionReady(plate_text, plate_color, "-");
        cv::Rect full_img_box(5, 5, frame.cols - 10, frame.rows - 10);
        drawChineseTextAndBox(frame, full_img_box, label_str, box_color);
    }

    int64 t_end = cv::getTickCount();
    double time_ms = (t_end - t_start) * 1000.0 / cv::getTickFrequency();
    cv::putText(frame, cv::format("Cost: %.1f ms", time_ms), cv::Point(15, 150), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

    QString out_file = QString("output/img/image_%1.jpg").arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
    cv::imwrite(out_file.toStdString(), frame);
    
    emit frameReady(matToQImage(frame));
    emit showMessage("Complete. Saved to: " + out_file);
}

void InferenceThread::processStream() {
    cv::VideoCapture cap;
    emit showMessage("Connecting to Local Media...");
    cap.open(file_path.toStdString());

    if (!cap.isOpened()) {
        emit showMessage("ERROR: Media Unreadable");
        return;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    QString out_file = QString("output/video/video_%1.mp4").arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
    cv::VideoWriter writer(out_file.toStdString(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    
    emit showMessage("Stream Active -> " + out_file);

    int job_cnt = 0, result_cnt = 0;
    std::queue<cv::Mat> frame_queue;
    cv::Mat frame;
    int numThreads = 1;

    int64 fps_start_time = cv::getTickCount();
    int fps_frame_count = 0;
    double current_fps = 0.0;

    while (keep_running || result_cnt < job_cnt) {
        bool current_use_yolo = use_yolo; 
        bool should_read = false;
        
        if (current_use_yolo) should_read = (job_cnt - result_cnt < numThreads);
        else should_read = (result_cnt == job_cnt);

        if (keep_running && should_read) {
            cap >> frame;
            if (frame.empty()) break; 
            
            if (current_use_yolo) {
                yolo11_thread_pool->submitTask(frame, job_cnt++);
                frame_queue.push(frame.clone());
            } else {
                cv::Mat draw_frame = frame.clone();
                QString plate_color = detectPlateColor(draw_frame);
                QString plate_text = recognizeText(draw_frame);
                plate_color = refinePlateColorByText(plate_text, plate_color);

                cv::Scalar box_color = cv::Scalar(255, 255, 255); 
                QString label_str = QStringLiteral("[\u5168\u56fe\u76f4\u51fa] %1 (%2)").arg(plate_text).arg(plate_color);
                cv::Rect full_img_box(5, 5, draw_frame.cols - 10, draw_frame.rows - 10);
                drawChineseTextAndBox(draw_frame, full_img_box, label_str, box_color);

                fps_frame_count++;
                if (fps_frame_count % 10 == 0) {
                    int64 fps_end_time = cv::getTickCount();
                    current_fps = 10.0 * cv::getTickFrequency() / (fps_end_time - fps_start_time);
                    fps_start_time = fps_end_time; 
                }
                if (current_fps > 0.0) {
                    cv::putText(draw_frame, cv::format("FPS: %.1f", current_fps), cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                }

                if (writer.isOpened()) writer.write(draw_frame); 
                emit frameReady(matToQImage(draw_frame));
            }
        }

        if (result_cnt < job_cnt) {
            std::vector<Detection> objects;
            if (yolo11_thread_pool->getTargetResultNonBlock(objects, result_cnt) == NN_SUCCESS) {
                cv::Mat draw_frame = frame_queue.front();
                frame_queue.pop();
                
                bool plate_detected = false;
                for (auto &obj : objects) {
                    if (obj.confidence < 0.15) continue; 
                    cv::Rect safe_box = obj.box & cv::Rect(0, 0, draw_frame.cols, draw_frame.rows);
                    if (safe_box.area() <= 0) continue;

                    plate_detected = true;
                    cv::Mat plate_crop = draw_frame(safe_box);
                    QString plate_color = detectPlateColor(plate_crop);
                    QString plate_text = recognizeText(plate_crop); 

                    cv::Scalar box_color = cv::Scalar(255, 255, 255); 
                    QString confidence_text = QString::number(obj.confidence, 'f', 2);
                    QString label_str = QString("[%1] %2 (%3)").arg(confidence_text).arg(plate_text).arg(plate_color);
                    emit recognitionReady(plate_text, plate_color, confidence_text);
                    drawChineseTextAndBox(draw_frame, safe_box, label_str, box_color);
                }

                if (!plate_detected) {
                    QString plate_color = detectPlateColor(draw_frame);
                    QString plate_text = recognizeText(draw_frame); 
                    plate_color = refinePlateColorByText(plate_text, plate_color);
                    cv::Scalar box_color = cv::Scalar(255, 255, 255); 
                    QString label_str = QStringLiteral("[\u5168\u56fe\u515c\u5e95] %1 (%2)").arg(plate_text).arg(plate_color);
                    emit recognitionReady(plate_text, plate_color, "-");
                    cv::Rect full_img_box(5, 5, draw_frame.cols - 10, draw_frame.rows - 10);
                    drawChineseTextAndBox(draw_frame, full_img_box, label_str, box_color);
                }

                fps_frame_count++;
                if (fps_frame_count % 10 == 0) {
                    int64 fps_end_time = cv::getTickCount();
                    current_fps = 10.0 * cv::getTickFrequency() / (fps_end_time - fps_start_time);
                    fps_start_time = fps_end_time; 
                }
                
                if (current_fps > 0.0) {
                    cv::putText(draw_frame, cv::format("FPS: %.1f", current_fps), cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                }

                if (writer.isOpened()) writer.write(draw_frame); 
                emit frameReady(matToQImage(draw_frame));
                result_cnt++;
            } else {
                usleep(1000);
            }
        } else if (!should_read) {
            usleep(1000);
        }
        if (!keep_running && result_cnt == job_cnt) break;
    }
    
    if (writer.isOpened()) writer.release(); 
    cap.release();
    emit showMessage("Stream Offline.");
}

// =========================================================================
// ?? 核心 UI 构建与交互逻辑 (Aurora Glassmorphism 极光毛玻璃拟态风)
// =========================================================================

// 添加磨砂阴影的辅助函数
static void addGlassShadow(QWidget* widget) {
    QGraphicsDropShadowEffect* shadow = new QGraphicsDropShadowEffect(widget);
    shadow->setBlurRadius(30);        
    shadow->setXOffset(0);            
    shadow->setYOffset(10);            
    shadow->setColor(QColor(0, 0, 0, 100)); // 制造深邃的空间悬浮感
    widget->setGraphicsEffect(shadow);
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUI();
    inferenceThread = new InferenceThread(this);

    inferenceThread->setUseYolo(true);
    inferenceThread->setRecognitionModel(MODEL_PPOCR);
    
    connect(inferenceThread, &InferenceThread::frameReady, this, &MainWindow::updateFrame, Qt::QueuedConnection);
    connect(inferenceThread, &InferenceThread::showMessage, this, &MainWindow::updateStatus, Qt::QueuedConnection);
    connect(inferenceThread, &InferenceThread::recognitionReady, this, &MainWindow::updateRecognition, Qt::QueuedConnection);
}

MainWindow::~MainWindow() {}

void MainWindow::setupUI() {
    this->setWindowTitle("license plate recognition");
    this->resize(1400, 820);

    this->setStyleSheet(
        "QMainWindow { background: #ffffff; font-family: 'Segoe UI', 'Microsoft YaHei'; }"
        "QFrame#DisplayPanel, QFrame#SidePanel, QFrame#StatusBar {"
        "    background-color: #ffffff; border: none; border-radius: 0px;"
        "}"
        "QFrame#ResultCard, QFrame#InfoRow {"
        "    background-color: #ffffff; border: 1px solid #d9d9d9; border-radius: 6px;"
        "}"
        "QLabel { color: #000000; background-color: #ffffff; }"
        "QLabel#SectionLabel { font-size: 13px; font-weight: 700; color: #000000; margin-top: 10px; }"
        "QLabel#DisplayLabel { background-color: #ffffff; border: 1px solid #d9d9d9; border-radius: 4px; font-size: 18px; color: #000000; }"
        "QLabel#StatusLabel { color: #000000; font-size: 13px; font-weight: 600; padding-left: 8px; }"
        "QLabel#PlateValue { color: #000000; font-size: 20px; font-weight: 800; padding: 4px 0; }"
        "QLabel#PlateColorValue { color: #000000; font-size: 15px; font-weight: 800; padding: 4px 0; }"
        "QLabel#InfoKey { color: #000000; font-size: 12px; }"
        "QLabel#InfoValue { color: #000000; font-size: 14px; font-weight: 700; }"
        "QPushButton {"
        "    min-height: 42px; border-radius: 6px; padding: 8px 14px; font-size: 14px; font-weight: 700;"
        "    color: #000000; background-color: #ffffff; border: 1px solid #d9d9d9;"
        "}"
        "QPushButton:hover { background-color: #f2f2f2; border-color: #bdbdbd; }"
        "QPushButton:pressed { background-color: #e8e8e8; }"
        "QPushButton#PrimaryButton { background-color: #ffffff; border-color: #bdbdbd; color: #000000; }"
        "QPushButton#PrimaryButton:hover { background-color: #f2f2f2; }"
        "QPushButton#DangerButton { background-color: #ffffff; border-color: #bdbdbd; color: #000000; }"
        "QPushButton#DangerButton:hover { background-color: #f2f2f2; }"
        "QToolButton#EngineSelectButton { background-color: #ffffff; color: #000000; border: 1px solid #d9d9d9; border-radius: 6px; padding: 8px 38px 8px 12px; font-size: 14px; font-weight: 700; text-align: left; min-height: 42px; }"
        "QToolButton#EngineSelectButton:hover { border-color: #bdbdbd; }"
        "QToolButton#EngineSelectButton::menu-indicator { subcontrol-origin: padding; subcontrol-position: right center; right: 14px; width: 0px; height: 0px; image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid #000000; }"
        "QMenuBar { background-color: #ffffff; color: #000000; border-bottom: 1px solid #d9d9d9; font-size: 14px; font-weight: 700; }"
        "QMenuBar::item { padding: 8px 16px; background: transparent; }"
        "QMenuBar::item:selected { background-color: #f2f2f2; color: #000000; }"
        "QMenu { background-color: #ffffff; border: 1px solid #d9d9d9; border-radius: 6px; padding: 4px; }"
        "QMenu::item { color: #000000; background-color: #ffffff; min-height: 36px; padding: 8px 14px; font-size: 14px; font-weight: 700; }"
        "QMenu::item:selected { background-color: #f2f2f2; color: #000000; border-radius: 4px; }"
    );

    menuBar()->setNativeMenuBar(false);
    QMenu *inputMenu = menuBar()->addMenu(QStringLiteral("\u8f93\u5165\u6e90"));
    QAction *actionLocalInput = inputMenu->addAction(QStringLiteral("\u672c\u5730\u8f93\u5165"));
    QAction *actionCameraInput = inputMenu->addAction(QStringLiteral("\u6444\u50cf\u5934\u8f93\u5165"));
    inputMenu->addSeparator();
    QAction *actionStopInput = inputMenu->addAction(QStringLiteral("\u505c\u6b62"));
    connect(actionLocalInput, &QAction::triggered, this, &MainWindow::openLocalInput);
    connect(actionCameraInput, &QAction::triggered, this, &MainWindow::openCameraInput);
    connect(actionStopInput, &QAction::triggered, this, &MainWindow::stopInput);

    engineMenu = menuBar()->addMenu(QStringLiteral("\u8bc6\u522b\u6a21\u578b"));
    QActionGroup *engineActionGroup = new QActionGroup(engineMenu);
    engineActionGroup->setExclusive(true);
    actionLprNet = engineMenu->addAction("LPRNet");
    actionPPOCR = engineMenu->addAction("PPOCRv4");
    actionFusion = engineMenu->addAction(QStringLiteral("\u878d\u5408\u8bc6\u522b"));
    for (QAction *action : { actionLprNet, actionPPOCR, actionFusion }) {
        action->setCheckable(true);
        engineActionGroup->addAction(action);
    }
    actionPPOCR->setChecked(true);
    connect(actionLprNet, &QAction::triggered, this, &MainWindow::selectLprNet);
    connect(actionPPOCR, &QAction::triggered, this, &MainWindow::selectPPOCR);
    connect(actionFusion, &QAction::triggered, this, &MainWindow::selectFusion);

    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setContentsMargins(18, 18, 18, 18);
    mainLayout->setSpacing(14);

    QHBoxLayout *contentLayout = new QHBoxLayout();
    contentLayout->setSpacing(14);

    QFrame *displayFrame = new QFrame();
    displayFrame->setObjectName("DisplayPanel");
    QVBoxLayout *displayLayout = new QVBoxLayout(displayFrame);
    displayLayout->setContentsMargins(0, 0, 0, 0);
    displayLayout->setSpacing(0);

    displayLabel = new QLabel(QStringLiteral("\u7b49\u5f85\u8f93\u5165"));
    displayLabel->setObjectName("DisplayLabel");
    displayLabel->setAlignment(Qt::AlignCenter);
    displayLabel->setMinimumSize(760, 520);
    displayLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    displayLayout->addWidget(displayLabel, 1);

#if 0
    QFrame *sidePanel = new QFrame();
    sidePanel->setObjectName("SidePanel");
    sidePanel->setFixedWidth(360);
    addGlassShadow(sidePanel);
    QVBoxLayout *sideLayout = new QVBoxLayout(sidePanel);
    sideLayout->setContentsMargins(18, 18, 18, 18);
    sideLayout->setSpacing(10);

    /*
    QLabel *resultSection = new QLabel("识别结果");
    resultSection->setObjectName("SectionLabel");
    QFrame *resultCard = new QFrame();
    resultCard->setObjectName("ResultCard");
    QVBoxLayout *resultLayout = new QVBoxLayout(resultCard);
    resultLayout->setContentsMargins(14, 12, 14, 12);
    QHBoxLayout *resultHeaderLayout = new QHBoxLayout();
    resultHeaderLayout->setContentsMargins(0, 0, 0, 2);
    QLabel *resultCaption = new QLabel("最近车牌");
    resultCaption->setObjectName("InfoKey");
    QLabel *colorCaption = new QLabel("颜色");
    colorCaption->setObjectName("InfoKey");
    colorCaption->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    resultHeaderLayout->addWidget(resultCaption);
    resultHeaderLayout->addStretch();
    resultHeaderLayout->addWidget(colorCaption);
    resultLayout->addLayout(resultHeaderLayout);
    for (int i = 0; i < 3; ++i) {
        QHBoxLayout *plateRowLayout = new QHBoxLayout();
        plateRowLayout->setContentsMargins(0, 0, 0, 0);
        plateValueLabels[i] = new QLabel("--");
        plateValueLabels[i]->setObjectName("PlateValue");
        plateValueLabels[i]->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        plateColorLabels[i] = new QLabel("--");
        plateColorLabels[i]->setObjectName("PlateColorValue");
        plateColorLabels[i]->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        plateRowLayout->addWidget(plateValueLabels[i], 1);
        plateRowLayout->addWidget(plateColorLabels[i]);
        resultLayout->addLayout(plateRowLayout);
    }
    */

    sourceValueLabel = new QLabel(QStringLiteral("\u5f85\u673a"));
    engineValueLabel = new QLabel("PPOCRv4");

    auto makeInfoRow = [&](const QString& keyText, QLabel *valueLabel) -> QFrame* {
        QFrame *row = new QFrame();
        row->setObjectName("InfoRow");
        QHBoxLayout *rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(12, 8, 12, 8);
        QLabel *keyLabel = new QLabel(keyText);
        keyLabel->setObjectName("InfoKey");
        valueLabel->setObjectName("InfoValue");
        valueLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        rowLayout->addWidget(keyLabel);
        rowLayout->addStretch();
        rowLayout->addWidget(valueLabel);
        return row;
    };

    /*
    sideLayout->addWidget(resultSection);
    sideLayout->addWidget(resultCard);
    */
    sideLayout->addWidget(makeInfoRow(QStringLiteral("\u8f93\u5165"), sourceValueLabel));
    sideLayout->addWidget(makeInfoRow(QStringLiteral("\u5f15\u64ce"), engineValueLabel));
    sideLayout->addStretch();
#endif

    contentLayout->addWidget(displayFrame, 1);
    // contentLayout->addWidget(sidePanel);

#if 0
    QFrame *bottomBar = new QFrame();
    bottomBar->setObjectName("StatusBar");
    QHBoxLayout *bottomLayout = new QHBoxLayout(bottomBar);
    bottomLayout->setContentsMargins(14, 8, 14, 8);
    QLabel *statusKey = new QLabel(QStringLiteral("\u72b6\u6001"));
    statusKey->setObjectName("InfoKey");
    statusLabel = new QLabel(QStringLiteral("\u7cfb\u7edf\u5f85\u673a\uff0c\u7b49\u5f85\u8f93\u5165"));
    statusLabel->setObjectName("StatusLabel");
    bottomLayout->addWidget(statusKey);
    bottomLayout->addWidget(statusLabel, 1);
#endif

    mainLayout->addLayout(contentLayout, 1);
    // mainLayout->addWidget(bottomBar);

    setCentralWidget(centralWidget);
}

void MainWindow::selectLprNet() {
    if (inferenceThread) inferenceThread->setRecognitionModel(MODEL_LPRNET);
    if (actionLprNet) actionLprNet->setChecked(true);
    if (engineValueLabel) engineValueLabel->setText("LPRNet");
    if (engineMenu) engineMenu->hide();
}

void MainWindow::selectPPOCR() {
    if (inferenceThread) inferenceThread->setRecognitionModel(MODEL_PPOCR);
    if (actionPPOCR) actionPPOCR->setChecked(true);
    if (engineValueLabel) engineValueLabel->setText("PPOCRv4");
    if (engineMenu) engineMenu->hide();
}

void MainWindow::selectFusion() {
    if (inferenceThread) inferenceThread->setRecognitionModel(MODEL_FUSION);
    if (actionFusion) actionFusion->setChecked(true);
    if (engineValueLabel) engineValueLabel->setText(QStringLiteral("\u878d\u5408\u8bc6\u522b"));
    if (engineMenu) engineMenu->hide();
}

void MainWindow::closeEvent(QCloseEvent *event) {
    if (inferenceThread->isRunning()) {
        inferenceThread->stop();
        inferenceThread->wait();
    }
    event->accept();
}

void MainWindow::openLocalInput() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("\u9009\u62e9\u672c\u5730\u8f93\u5165"),
        "",
        "Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mkv);;Images (*.png *.jpg *.jpeg *.bmp);;Videos (*.mp4 *.avi *.mkv)"
    );
    if (fileName.isEmpty()) return;

    if (inferenceThread->isRunning()) {
        inferenceThread->stop();
        inferenceThread->wait();
    }

    inferenceThread->setUseYolo(true);
    QString suffix = QFileInfo(fileName).suffix().toLower();
    if (suffix == "mp4" || suffix == "avi" || suffix == "mkv") {
        inferenceThread->setInputVideo(fileName);
        if (sourceValueLabel) sourceValueLabel->setText(QStringLiteral("\u672c\u5730\u89c6\u9891"));
        updateStatus(QStringLiteral("\u672c\u5730\u89c6\u9891\u5df2\u8f7d\u5165"));
    } else {
        inferenceThread->setInputImage(fileName);
        if (sourceValueLabel) sourceValueLabel->setText(QStringLiteral("\u672c\u5730\u56fe\u7247"));
        updateStatus(QStringLiteral("\u672c\u5730\u56fe\u7247\u5df2\u8f7d\u5165"));
    }

    recentPlateTexts.clear();
    recentPlateColors.clear();
    for (int i = 0; i < 3; ++i) {
        if (plateValueLabels[i]) plateValueLabels[i]->setText("--");
        if (plateColorLabels[i]) plateColorLabels[i]->setText("--");
    }
    inferenceThread->start();
}

void MainWindow::openCameraInput() {
    openPCIe();
}

void MainWindow::openPCIe() {
    if (inferenceThread->isRunning()) { 
        inferenceThread->stop(); 
        inferenceThread->wait(); 
    }
    inferenceThread->setUseYolo(true);
    inferenceThread->setInputPCIe();
    if (sourceValueLabel) sourceValueLabel->setText(QStringLiteral("\u6444\u50cf\u5934"));
    recentPlateTexts.clear();
    recentPlateColors.clear();
    for (int i = 0; i < 3; ++i) {
        if (plateValueLabels[i]) plateValueLabels[i]->setText("--");
        if (plateColorLabels[i]) plateColorLabels[i]->setText("--");
    }
    updateStatus(QStringLiteral("\u6444\u50cf\u5934\u8f93\u5165\u542f\u52a8\u4e2d"));
    inferenceThread->start();
}

void MainWindow::stopInput() {
    if (inferenceThread->isRunning()) {
        inferenceThread->stop();
        inferenceThread->wait();
    }
    if (sourceValueLabel) sourceValueLabel->setText(QStringLiteral("\u5f85\u673a"));
    updateStatus(QStringLiteral("\u7cfb\u7edf\u5f85\u673a\uff0c\u7b49\u5f85\u8f93\u5165"));
}

void MainWindow::updateFrame(QImage image) { 
    displayLabel->setPixmap(QPixmap::fromImage(image).scaled(displayLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation)); 
}

void MainWindow::updateStatus(QString msg) { 
    if (statusLabel) statusLabel->setText(msg);
}

void MainWindow::updateRecognition(QString plate, QString color, QString confidence) {
    Q_UNUSED(confidence);
    QString normalizedPlate = plate.trimmed();
    QString normalizedColor = refinePlateColorByText(normalizedPlate, color.trimmed());
    if (normalizedColor.isEmpty() ||
        normalizedColor.contains(QStringLiteral("\u5931\u8d25")) ||
        normalizedColor.contains(QStringLiteral("\u9519\u8bef")) ||
        normalizedColor.contains(QStringLiteral("\u672a\u5c31\u7eea")) ||
        normalizedColor.contains(QStringLiteral("\u672a\u77e5"))) {
        normalizedColor = "--";
    }

    const bool hasValidPlate = !normalizedPlate.isEmpty() &&
                               normalizedPlate != "--" &&
                               !normalizedPlate.contains(QStringLiteral("\u5931\u8d25")) &&
                               !normalizedPlate.contains(QStringLiteral("\u9519\u8bef")) &&
                               !normalizedPlate.contains(QStringLiteral("\u672a\u5c31\u7eea")) &&
                               !normalizedPlate.contains(QStringLiteral("\u672a\u77e5"));

    if (hasValidPlate) {
        for (int i = recentPlateTexts.size() - 1; i >= 0; --i) {
            if (recentPlateTexts.at(i) == normalizedPlate) {
                recentPlateTexts.removeAt(i);
                if (i < recentPlateColors.size()) recentPlateColors.removeAt(i);
            }
        }
        recentPlateTexts.prepend(normalizedPlate);
        recentPlateColors.prepend(normalizedColor);
        while (recentPlateTexts.size() > 3) recentPlateTexts.removeLast();
        while (recentPlateColors.size() > 3) recentPlateColors.removeLast();
    }

    for (int i = 0; i < 3; ++i) {
        QString plateText = (i < recentPlateTexts.size()) ? recentPlateTexts.at(i) : "--";
        QString colorText = (i < recentPlateColors.size()) ? recentPlateColors.at(i) : "--";
        if (plateValueLabels[i]) plateValueLabels[i]->setText(plateText);
        if (plateColorLabels[i]) plateColorLabels[i]->setText(colorText);
    }
}
