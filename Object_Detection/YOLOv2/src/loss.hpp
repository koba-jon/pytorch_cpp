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
    long int class_num, na;
    float thresh;
    torch::Tensor anchors_wh;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> make_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target, const long int ng);
    torch::Tensor rescale(torch::Tensor &BBs);
    torch::Tensor compute_iou(torch::Tensor &BBs1, torch::Tensor &BBs2);
public:
    Loss(){}
    Loss(const std::vector<std::tuple<float, float>> anchors_, const long int class_num_, const float thresh_);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> operator()(torch::Tensor &input, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target);
};


#endif