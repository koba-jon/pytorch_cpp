#include <iostream>
#include <string>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "losses.hpp"


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor sum_logdet, torch::Tensor sum_log_p, size_t n_pixel){
    torch::Tensor loss;
    loss = sum_logdet + sum_log_p;
    loss = (-loss / (std::log(2.0) * n_pixel)).mean();
    return loss;
}
