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

// -------------------
// class{MMDLoss}
// -------------------
class MMDLoss{
private:
    float var;
public:
    MMDLoss(const float var_=1.0);
    torch::Tensor get_kernel_sum(torch::Tensor &z1, torch::Tensor &z2, const bool exclude_diag=true, const float eps=1e-12);
    torch::Tensor operator()(torch::Tensor &input, torch::Tensor &target);
};


#endif