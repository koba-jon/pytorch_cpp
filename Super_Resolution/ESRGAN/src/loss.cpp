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
Loss::Loss(const std::string loss){
    if (loss == "vanilla"){
        this->judge = 0;
        auto criterion = torch::nn::BCEWithLogitsLoss(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean));
        torch::Tensor equal = torch::full({1}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat));
        torch::Tensor label = torch::full({1}, /*value=*/1.0, torch::TensorOptions().dtype(torch::kFloat));
        this->ideal = criterion(equal, label).item<float>();
    }
    else if (loss == "lsgan"){
        this->judge = 1;
        auto criterion = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
        torch::Tensor equal = torch::full({1}, /*value=*/0.5, torch::TensorOptions().dtype(torch::kFloat));
        torch::Tensor label = torch::full({1}, /*value=*/1.0, torch::TensorOptions().dtype(torch::kFloat));
        this->ideal = criterion(equal, label).item<float>();
    }
    else{
        std::cerr << "Error : The loss fuction isn't defined right." << std::endl;
        std::exit(1);
    }
}


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor input, torch::Tensor target){
    if (this->judge == 0){
        static auto criterion = torch::nn::BCEWithLogitsLoss(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean));
        return criterion(input, target);
    }
    static auto criterion = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
    return criterion(input, target);
}


// -----------------------------------
// class{Loss} -> function{ideal_value}
// -----------------------------------
float Loss::ideal_value(){
    return this->ideal;
}
