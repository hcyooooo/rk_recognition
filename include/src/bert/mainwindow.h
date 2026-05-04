#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QTextEdit>
#include <QLabel>
#include <QProgressBar>
#include <QComboBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <thread>
#include <memory>
#include "bert_thread_pool.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

signals:
    // 跨线程通信信号
    void sigInitDone();
    void sigSingleResult(int token_id, float score, double time_ms);
    void sigStressProgress(int current, int total);
    void sigStressDone(double fps, double total_time);

private slots:
    // 按钮响应槽函数
    void onRunSingleClicked();
    void onRunStressClicked();
    
    // UI 更新槽函数
    void updateInitStatus();
    void updateSingleResult(int token_id, float score, double time_ms);
    void updateStressProgress(int current, int total);
    void updateStressDone(double fps, double total_time);

private:
    void setupUI();
    void initRKNN();

    // UI 控件
    QComboBox *comboSentences;
    QTextEdit *txtOutput;
    QPushButton *btnRunSingle;
    QPushButton *btnRunStress;
    QLabel *lblStatus;
    QProgressBar *barTop1;
    QProgressBar *barStress;

    // BERT 线程池后端
    std::shared_ptr<BertThreadPool> pool_;
    bool is_initialized_;
};

#endif // MAINWINDOW_H