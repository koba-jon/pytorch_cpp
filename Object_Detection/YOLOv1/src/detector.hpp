#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <tuple>
#include <vector>
// For External Library
#include <torch/torch.h>


// -------------------
// class{YOLODetector}
// -------------------
class YOLODetector{
private:
    long int class_num, ng, nb;
    float prob_thresh, nms_thresh;
    torch::Tensor NonMaximumSuppression(torch::Tensor &coord, torch::Tensor &conf);
public:
    YOLODetector(){}
    YOLODetector(const long int class_num_, const long int ng_, const long int nb_, const float prob_thresh_, const float nms_thresh_);
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> get_label_palette();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> operator()(const torch::Tensor pred);
};


#endif