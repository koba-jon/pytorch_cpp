#ifndef LOSS_HPP
#define LOSS_HPP

#include <string>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "losses.hpp"


// -------------------
// class{Loss}
// -------------------
class Loss{
private:
    float Lambda;
    torch::nn::MSELoss l2;
    Losses::SSIMLoss ssim;
public:
    Loss(){}
    Loss(const float Lambda_, torch::Device device);
    torch::Tensor operator()(torch::Tensor &input, torch::Tensor &target);
};


#endif