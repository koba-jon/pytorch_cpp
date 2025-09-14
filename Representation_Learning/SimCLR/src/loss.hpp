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
    torch::Tensor operator()(torch::Tensor z1, torch::Tensor z2, const float tau=0.1);
};


#endif