#include "mainwindow.h"
#include <QApplication>
#include <csignal>
#include <QDebug>

QApplication *appPtr = nullptr;

void handle_sigint(int sig) {
    if (sig == SIGINT && appPtr) {
        qDebug() << "\n[INFO] 捕捉到 Ctrl+C! 正在触发安全退出，保存视频尾帧...";
        appPtr->quit(); 
    }
}

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    appPtr = &a;

    //
    signal(SIGINT, handle_sigint);

    MainWindow w;
    w.show();
    return a.exec();
}