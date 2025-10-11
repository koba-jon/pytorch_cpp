#ifndef LOSS_HPP
#define LOSS_HPP

#include <tuple>
#include <vector>
#include <array>
#include <utility>
// For External Library
#include <torch/torch.h>


// -----------------------------
// struct{LossHyperparameters}
// -----------------------------
struct LossHyperparameters{
    float anchor_t = 4.0;
    float label_smoothing = 0.0;
    float class_pos_weight = 1.0;
    float obj_pos_weight = 1.0;
    float focal_gamma = 0.0;
    float focal_alpha = 0.25;
};


// -------------------
// class{Loss}
// -------------------
class Loss{
private:
    long int class_num, na;
    float anchor_thresh;
    torch::Tensor anchors;
    std::vector<float> balance;
    LossHyperparameters hyp;
    bool autobalance;
    bool ssi_initialized;
    size_t ssi;
    float cp, cn;
    float focal_gamma, focal_alpha;
    float class_pos_weight, obj_pos_weight;
    torch::Tensor format_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>> target, torch::Device device);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> build_target(torch::Tensor targets, std::vector<std::array<long int, 2>> grid_shapes, std::vector<torch::Tensor> scaled_anchors, torch::Device device);
    torch::Tensor bbox_iou(torch::Tensor box1, torch::Tensor box2);
    std::pair<float, float> smooth_BCE(float eps);
    torch::Tensor binary_cross_entropy(torch::Tensor input, torch::Tensor target, float pos_weight);
public:
    Loss(){}
    Loss(const std::vector<std::vector<std::tuple<float, float>>> anchors_, const long int class_num_, const float anchor_thresh_, const LossHyperparameters &hyp_=LossHyperparameters(), const bool autobalance_=false);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> operator()(std::vector<torch::Tensor> &inputs, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target, const std::tuple<float, float> image_sizes);
};


#endif