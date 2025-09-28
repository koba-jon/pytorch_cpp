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
torch::Tensor Loss::operator()(torch::Tensor &zK, torch::Tensor &sum_logdet){
    torch::Tensor log_pz, log_px, loss;
    log_pz = (-0.5 * (zK.pow(2.0) + std::log(2.0 * M_PI))).sum(1);
    log_px = log_pz + sum_logdet;
    loss = -log_px.mean();
    return loss;
}
