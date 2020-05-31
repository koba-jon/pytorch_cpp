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
    int flag;
public:
    Loss(){}
    Loss(const std::string loss);
    torch::Tensor operator()(torch::Tensor &input, torch::Tensor &target);
};


// -------------------
// class{SSIMLoss}
// -------------------
class SSIMLoss{
private:
    size_t nc;
    size_t window_size;
    float gauss_std;
    float c1_base;
    float c2_base;
    torch::Tensor window;
public:
    SSIMLoss(){}
    SSIMLoss(const size_t nc_, const torch::Device device, const size_t window_size_=11, const float gauss_std_=1.5, const float c1_base_=0.01, const float c2_base_=0.03);
    torch::Tensor Structural_Similarity(torch::Tensor &image1, torch::Tensor &image2);
    torch::Tensor operator()(torch::Tensor &input, torch::Tensor &target);
};


#endif