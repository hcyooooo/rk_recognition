#ifndef BERT_THREAD_POOL_H
#define BERT_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <string>
#include "rknn_api.h"

// 与导出时对齐
#define SEQ_LENGTH 128
#define VOCAB_SIZE 30522

// 单个预测的分数
struct TokenScore {
    int id;
    float score;
    bool operator<(const TokenScore& other) const {
        return score > other.score; // 降序排列
    }
};

// 线程池返回的结果
struct BertResult {
    int task_id;
    std::vector<TokenScore> top_k;
};

// 推送给线程池的任务
struct BertTask {
    int task_id;
    int32_t input_ids[SEQ_LENGTH];
    int32_t attention_mask[SEQ_LENGTH];
    int32_t token_type_ids[SEQ_LENGTH];
};

class BertThreadPool {
public:
    // 初始化线程池 (指定模型路径和线程数量)
    BertThreadPool(const std::string& model_path, int num_threads);
    ~BertThreadPool();

    // 提交推理任务 (非阻塞)
    int submitTask(const BertTask& task);
    // 获取推理结果 (可设置阻塞或非阻塞)
    bool getResult(int task_id, BertResult& result, bool blocking = true);

private:
    // 工作线程内部的死循环函数
    void workerFunction(int thread_id);

    int num_threads_;
    std::vector<rknn_context> ctx_list_;

    std::vector<std::thread> workers_;
    std::queue<BertTask> task_queue_;
    std::map<int, BertResult> result_map_;

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::mutex result_mutex_;
    std::condition_variable result_condition_;

    bool stop_;
};

#endif // BERT_THREAD_POOL_H