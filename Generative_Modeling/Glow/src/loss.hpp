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
    torch::Tensor operator()(torch::Tensor sum_logdet, torch::Tensor sum_log_p, size_t n_pixel);
};


#endif