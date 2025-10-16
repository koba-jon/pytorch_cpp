#ifndef LOSS_HPP
#define LOSS_HPP

#include <tuple>
#include <vector>
#include <array>
#include <utility>
// For External Library
#include <torch/torch.h>


namespace nn = torch::nn;


// -------------------
// class{Loss}
// -------------------
class Loss{
private:
    long int class_num;
    long int reg_max;
    nn::BCEWithLogitsLoss BCE;
    std::vector<float> balance;
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> build_target(std::vector<torch::Tensor> &inputs, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target);
    torch::Tensor bbox_iou(torch::Tensor box1, torch::Tensor box2);
    torch::Tensor distribution_focal_loss(torch::Tensor pred, torch::Tensor target);
public:
    Loss(){}
    Loss(const long int class_num_, const long int reg_max_);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> operator()(std::vector<torch::Tensor> &inputs, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target);
};


#endif