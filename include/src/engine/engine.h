// 接口定义

#ifndef RK3588_DEMO_ENGINE_H
#define RK3588_DEMO_ENGINE_H

#include "types/error.h"
#include "types/datatype.h"

#include <vector>
#include <memory>

class NNEngine
{
public:
    virtual ~NNEngine(){};
    // 👇 修改：增加 int core_id 参数 👇
    virtual nn_error_e LoadModelFile(const char *model_file, int core_id = 0) = 0;
    virtual const std::vector<tensor_attr_s> &GetInputShapes() = 0;
    virtual const std::vector<tensor_attr_s> &GetOutputShapes() = 0;
    virtual nn_error_e Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outpus, bool want_float) = 0;
    
};

std::shared_ptr<NNEngine> CreateRKNNEngine(); 

#endif // RK3588_DEMO_ENGINE_H