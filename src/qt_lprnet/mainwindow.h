#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QToolButton>
#include <QMenu>
#include <QAction>
#include <QFrame>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QThread>
#include <QImage>
#include <QFileDialog>
#include <QCloseEvent>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QMutex>
#include <QStringList>
#include <queue>
#include <vector>
#include <atomic>       

#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "task/yolov11_custom.h"
#include "task/yolov11_thread_pool.h"

// 引入两种识别模型的头文件
#include "lprnet.h"     
#include "ppocr.h"      

enum InputType {
    INPUT_IMAGE,
    INPUT_VIDEO,
    INPUT_PCIE
};

// 模型选择枚举
enum RecognitionModel {
    MODEL_LPRNET,
    MODEL_PPOCR,
    MODEL_FUSION
};

class InferenceThread : public QThread {
    Q_OBJECT
public:
    InferenceThread(QObject *parent = nullptr);
    ~InferenceThread();

    void setInputImage(const QString& path);
    void setInputVideo(const QString& path);
    void setInputPCIe();
    void stop();
    
    void setUseYolo(bool use) { use_yolo = use; }
    void setRecognitionModel(RecognitionModel model) { current_rec_model = model; } 

signals:
    void frameReady(QImage image);
    void showMessage(QString msg);
    void recognitionReady(QString plate, QString color, QString confidence);

protected:
    void run() override;

private:
    volatile bool keep_running;
    InputType current_input_type;
    QString file_path;
    
    std::atomic<bool> use_yolo; 
    std::atomic<RecognitionModel> current_rec_model; // 当前使用的识别模型

    Yolov11Custom *yolo11_detector;
    Yolov11ThreadPool *yolo11_thread_pool;
    
    rknn_app_context_t lprnet_ctx; 
    bool lprnet_ready;

    ppocr_app_context_t ppocr_ctx; 
    bool ppocr_ready;

    void processStream();
    void processPCIeStream();
    void processSingleImage();
    void ensureOutputDirectory();
    QImage matToQImage(const cv::Mat& mat);
    
    QString detectPlateColor(const cv::Mat& plate_img);
    QString recognizeText(const cv::Mat& plate_img); // 统一识别接口
    QString recognizeWithLprNet(const cv::Mat& plate_img);
    QString recognizeWithPPOCR(const cv::Mat& plate_img);
    void drawChineseTextAndBox(cv::Mat& img, const cv::Rect& box, const QString& text, const cv::Scalar& color);
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void openLocalInput();
    void openCameraInput();
    void stopInput();
    void openPCIe();
    void updateFrame(QImage image);
    void updateStatus(QString msg);
    void updateRecognition(QString plate, QString color, QString confidence);
    void selectLprNet();
    void selectPPOCR();
    void selectFusion();

private:
    QLabel *displayLabel = nullptr;
    QLabel *statusLabel = nullptr;
    QLabel *sourceValueLabel = nullptr;
    QLabel *engineValueLabel = nullptr;
    QLabel *plateValueLabels[3] = { nullptr, nullptr, nullptr };
    QLabel *plateColorLabels[3] = { nullptr, nullptr, nullptr };
    QAction *actionLprNet = nullptr;
    QAction *actionPPOCR = nullptr;
    QAction *actionFusion = nullptr;
    QMenu *engineMenu = nullptr;
    QStringList recentPlateTexts;
    QStringList recentPlateColors;
    
    InferenceThread *inferenceThread;
    void setupUI();
};

#endif // MAINWINDOW_H
