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
// class{MMDLoss} -> constructor
// -----------------------------------
MMDLoss::MMDLoss(const float var_){
    if (var_ > 0.0){
        this->var = var_;
    }
    else{
        std::cerr << "Error : The variance value isn't defined right." << std::endl;
        std::exit(1);
    }
}


// -------------------------------------------
// class{MMDLoss} -> function{get_kernel_sum}
// -------------------------------------------
torch::Tensor MMDLoss::get_kernel_sum(torch::Tensor &z1, torch::Tensor &z2, const bool exclude_diag, const float eps){
    float C = 2.0 * (float)z1.size(1) * this->var;  // 2.0 * Z * var
    torch::Tensor z11 = z1.unsqueeze(1).repeat({1, z2.size(0), 1});  // z1{N1,Z} ===> {N1,1,Z} ===> z11{N1,N2,Z}
    torch::Tensor z22 = z2.unsqueeze(0).repeat({z1.size(0), 1, 1});  // z2{N2,Z} ===> {1,N2,Z} ===> z22{N1,N2,Z}
    torch::Tensor kernel_matrix = C / (eps + C + (z11 - z22).pow(2.0).sum(/*dim=*/2));  // z11{N1,N2,Z}, z22{N1,N2,Z} ===> kernel_matrix{N1,N2}
    torch::Tensor kernel_sum = kernel_matrix.sum();
    if (exclude_diag){
        kernel_sum = kernel_sum - kernel_matrix.diag().sum();
    }
    return kernel_sum;
}


// -----------------------------------
// class{MMDLoss} -> operator
// -----------------------------------
torch::Tensor MMDLoss::operator()(torch::Tensor &input, torch::Tensor &target){
    float N = (float)input.size(0);
    torch::Tensor k11 = this->get_kernel_sum(input, input, /*exclude_diag=*/true);
    torch::Tensor k22 = this->get_kernel_sum(target, target, /*exclude_diag=*/true);
    torch::Tensor k12 = this->get_kernel_sum(input, target, /*exclude_diag=*/false);
    torch::Tensor out = k11 / (N * (N - 1.0)) + k22 / (N * (N - 1.0)) - k12 * 2.0 / (N * N);
    return out;
}
