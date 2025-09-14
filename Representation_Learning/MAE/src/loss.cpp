#include <iostream>
#include <string>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "losses.hpp"


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor image, torch::Tensor pred, torch::Tensor mask){
    torch::Tensor loss;
    loss = (pred - image) * (pred - image);  // {N,NP,P*P*C}
    loss = loss.mean(/*dim=*/-1);  // {N,NP,P*P*C} ===> {N,NP}
    loss = (loss * mask).sum() / mask.sum();  // {N,NP} ===> {}
    return loss;
}
