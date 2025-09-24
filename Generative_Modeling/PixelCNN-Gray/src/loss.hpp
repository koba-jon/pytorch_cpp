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
    long int L;
public:
    Loss(){}
    Loss(const long int L_);
    torch::Tensor operator()(torch::Tensor input, torch::Tensor target);
};


#endif