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
    float anchor_thresh;
    torch::Tensor anchors;
    std::vector<float> balance;
    torch::Tensor format_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>> target, torch::Device device);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> build_target(torch::Tensor targets, std::vector<std::array<long int, 2>> grid_shapes, std::vector<torch::Tensor> scaled_anchors, torch::Device device);
    torch::Tensor bbox_iou(torch::Tensor box1, torch::Tensor box2);
public:
    Loss(){}
    Loss(const std::vector<std::vector<std::tuple<float, float>>> anchors_, const long int class_num_, const float anchor_thresh_);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> operator()(std::vector<torch::Tensor> &inputs, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target, const std::tuple<float, float> image_sizes);
};


#endif