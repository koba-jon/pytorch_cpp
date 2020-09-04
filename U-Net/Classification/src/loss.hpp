#ifndef LOSS_HPP
#define LOSS_HPP

// For External Library
#include <torch/torch.h>


// -----------------------------------
// class{Loss}
// -----------------------------------
class Loss{
public:
    Loss(){}
    torch::Tensor operator()(torch::Tensor &input, torch::Tensor &target);
};

#endif
