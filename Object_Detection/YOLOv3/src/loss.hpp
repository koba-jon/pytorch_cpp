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
    float ignore_thresh;
    torch::Tensor anchors_wh;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target, const long int ng);
    torch::Tensor rescale(torch::Tensor &BBs);
    torch::Tensor compute_iou(torch::Tensor &BBs1, torch::Tensor &BBs2);
public:
    Loss(){}
    Loss(const std::vector<std::vector<std::tuple<float, float>>> anchors_, const long int class_num_, const float ignore_thresh_);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> operator()(std::vector<torch::Tensor> &inputs, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target, const std::tuple<float, float> image_sizes);
};


#endif