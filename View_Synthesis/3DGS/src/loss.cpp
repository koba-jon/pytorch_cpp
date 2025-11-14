#include <iostream>
#include <string>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "losses.hpp"


// -----------------------------------
// class{Loss} -> constructor
// -----------------------------------
Loss::Loss(const float Lambda_, torch::Device device){
    this->Lambda = Lambda_;
    this->l2 = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
    this->ssim = Losses::SSIMLoss(3, device);
}


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor &input, torch::Tensor &target){
    return (1.0 - this->Lambda) * this->l2(input, target) + this->Lambda * this->ssim(input, target);
}
