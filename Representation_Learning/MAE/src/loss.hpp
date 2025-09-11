#ifndef LOSS_HPP
#define LOSS_HPP

#include <string>
// For External Library
#include <torch/torch.h>


// -------------------
// class{Loss}
// -------------------
class Loss{
public:
    Loss(){}
    torch::Tensor operator()(torch::Tensor image, torch::Tensor pred, torch::Tensor mask);
};


#endif