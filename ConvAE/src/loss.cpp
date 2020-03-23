#include <iostream>
#include <string>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"


// -----------------------------------
// class{Loss} -> constructor
// -----------------------------------
Loss::Loss(const std::string loss){
    if (loss == "l1"){
        this->flag = 0;
    }
    else if (loss == "l2"){
        this->flag = 1;
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
    if (this->flag == 0){
        static auto criterion = torch::nn::L1Loss(torch::kMean);
        return criterion(input, target);
    }
    static auto criterion = torch::nn::MSELoss(torch::kMean);
    return criterion(input, target);
}