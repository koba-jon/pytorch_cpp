#ifndef LOSS_HPP
#define LOSS_HPP

#include <string>
// For External Library
#include <torch/torch.h>


// -------------------
// class{Loss}
// -------------------
class Loss{
private:
    int judge;
public:
    Loss(){}
    Loss(const std::string loss);
    torch::Tensor operator()(torch::Tensor &input, torch::Tensor &target);
};


#endif