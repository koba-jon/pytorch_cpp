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
Loss::Loss(const long int L_){
    this->L = L_;
}


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor input, torch::Tensor target){
    input = input.permute({0, 2, 3, 1}).contiguous().view({-1, this->L});
    target = (target * (this->L - 1) + 0.5).to(torch::kLong).view(-1);
    return F::cross_entropy(input, target, F::CrossEntropyFuncOptions().reduction(torch::kMean));
}
