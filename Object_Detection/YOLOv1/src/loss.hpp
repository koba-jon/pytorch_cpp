#ifndef LOSS_HPP
#define LOSS_HPP

#include <tuple>
#include <vector>
// For External Library
#include <torch/torch.h>


// -------------------
// class{Loss}
// -------------------
class Loss{
private:
    long int class_num, ng, nb;
    std::tuple<torch::Tensor, torch::Tensor> build_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target);
    torch::Tensor rescale(torch::Tensor &BBs);
    torch::Tensor compute_iou(torch::Tensor &BBs1, torch::Tensor &BBs2);
public:
    Loss(){}
    Loss(const long int class_num_, const long int ng_, const long int nb_);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> operator()(torch::Tensor &input, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target);
};


#endif