#include <iostream>
#include <string>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "losses.hpp"

// Define Namespace
namespace F = torch::nn::functional;


// -----------------------------------
// class{Loss} -> constructor
// -----------------------------------
Loss::Loss(const std::string loss){
    if (loss == "l1"){
        this->judge = 0;
    }
    else if (loss == "l2"){
        this->judge = 1;
    }
    else if (loss == "ssim"){
        this->judge = 2;
    }
    else{
        std::cerr << "Error : The loss fuction isn't defined right." << std::endl;
        std::exit(1);
    }
}


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor &input, torch::Tensor &target){
    if (this->judge == 0){
        static auto criterion = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kMean));
        return criterion(input, target);
    }
    else if (this->judge == 1){
        static auto criterion = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
        return criterion(input, target);
    }
    static auto criterion = Losses::SSIMLoss(input.size(1), input.device());
    return criterion(input, target);
}


// -----------------------------------
// class{Loss_PixelSnail} -> constructor
// -----------------------------------
Loss_PixelSnail::Loss_PixelSnail(const long int K_){
    this->K = K_;
}


// -----------------------------------
// class{Loss_PixelSnail} -> operator
// -----------------------------------
torch::Tensor Loss_PixelSnail::operator()(torch::Tensor &input, torch::Tensor &target){
    return F::cross_entropy(input.permute({0, 2, 3, 1}).contiguous().view({-1, this->K}), target.view(-1), F::CrossEntropyFuncOptions().reduction(torch::kMean));
}
