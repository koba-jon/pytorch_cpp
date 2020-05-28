#include <vector>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
// For Original Header
#include "transforms.hpp"


// -------------------------------------------
// namespace{transforms} -> function{apply}
// -------------------------------------------
torch::Tensor transforms::apply(std::vector<transforms::Compose*> &transform, cv::Mat &data_in){
    torch::Tensor data_out;
    transforms::forward<cv::Mat, torch::Tensor>(transform, data_in, data_out, transform.size());
    return data_out.clone();
}


// -------------------------------------------
// namespace{transforms} -> function{forward}
// -------------------------------------------
template <typename T_in, typename T_out>
void transforms::forward(std::vector<transforms::Compose*> &transform_, T_in &data_in, T_out &data_out, const int count){
    auto transform = transform_.at(count - 1);
    if (count > 1){
        auto transform2 = transform_.at(count - 2);
        if (transform2->type() == CV_MAT){
            cv::Mat data_mid;
            transforms::forward<T_in, cv::Mat>(transform_, data_in, data_mid, count - 1);
            transform->forward(data_mid, data_out);
        }
        else{
            torch::Tensor data_mid;
            transforms::forward<T_in, torch::Tensor>(transform_, data_in, data_mid, count - 1);
            transform->forward(data_mid, data_out);
        }
    }
    else{
        transform->forward(data_in, data_out);
    }
    return;
}
template void transforms::forward<cv::Mat, cv::Mat>(std::vector<transforms::Compose*> &transform_, cv::Mat &data_in, cv::Mat &data_out, const int count);
template void transforms::forward<cv::Mat, torch::Tensor>(std::vector<transforms::Compose*> &transform_, cv::Mat &data_in, torch::Tensor &data_out, const int count);
template void transforms::forward<torch::Tensor, cv::Mat>(std::vector<transforms::Compose*> &transform_, torch::Tensor &data_in, cv::Mat &data_out, const int count);
template void transforms::forward<torch::Tensor, torch::Tensor>(std::vector<transforms::Compose*> &transform_, torch::Tensor &data_in, torch::Tensor &data_out, const int count);


// -----------------------------------------------------------------------
// namespace{transforms} -> class{Grayscale}(Compose) -> constructor
// -----------------------------------------------------------------------
transforms::Grayscale::Grayscale(const int channels_){
    this->channels = channels_;
}


// -----------------------------------------------------------------------
// namespace{transforms} -> class{Grayscale}(Compose) -> function{forward}
// -----------------------------------------------------------------------
void transforms::Grayscale::forward(cv::Mat &data_in, cv::Mat &data_out){
    cv::Mat float_mat, float_mat_gray;
    data_in.convertTo(float_mat, CV_32F);  // discrete ===> continuous
    cv::cvtColor(float_mat, float_mat_gray, cv::COLOR_RGB2GRAY);
    float_mat_gray.convertTo(data_out, data_in.depth());  // continuous ===> discrete
    if (this->channels > 1){
        std::vector<cv::Mat> multi;
        for (int i = 0; i < this->channels; i++){
            multi.push_back(data_out.clone());
        }
        cv::merge(multi, data_out);
    }
    return;
}


// -----------------------------------------------------------------------
// namespace{transforms} -> class{Resize}(Compose) -> constructor
// -----------------------------------------------------------------------
transforms::Resize::Resize(const cv::Size size_, const int interpolation_){
    this->size = size_;
    this->interpolation = interpolation_;
}


// -----------------------------------------------------------------------
// namespace{transforms} -> class{Resize}(Compose) -> function{forward}
// -----------------------------------------------------------------------
void transforms::Resize::forward(cv::Mat &data_in, cv::Mat &data_out){
    cv::Mat float_mat, float_mat_resize;
    data_in.convertTo(float_mat, CV_32F);  // discrete ===> continuous
    cv::resize(float_mat, float_mat_resize, this->size, this->interpolation);
    float_mat_resize.convertTo(data_out, data_in.depth());  // continuous ===> discrete
    return;
}


// -----------------------------------------------------------------------
// namespace{transforms} -> class{ToTensor}(Compose) -> function{forward}
// -----------------------------------------------------------------------
void transforms::ToTensor::forward(cv::Mat &data_in, torch::Tensor &data_out){
    cv::Mat float_mat;
    data_in.convertTo(float_mat, CV_32F);  // discrete ===> continuous
    float_mat *= 1.0 / (std::pow(2.0, data_in.elemSize1()*8) - 1.0);  // [0,255] or [0,65535] ===> [0,1]
    torch::Tensor data_out_src = torch::from_blob(float_mat.data, {float_mat.rows, float_mat.cols, float_mat.channels()}, torch::kFloat);  // {0,1,2} = {H,W,C}
    data_out_src = data_out_src.permute({2, 0, 1});  // {0,1,2} = {H,W,C} ===> {0,1,2} = {C,H,W}
    data_out = data_out_src.clone();
    return;
}


// ----------------------------------------------------------------------------
// namespace{transforms} -> class{ToTensorLabel}(Compose) -> function{forward}
// ----------------------------------------------------------------------------
void transforms::ToTensorLabel::forward(cv::Mat &data_in, torch::Tensor &data_out){
    torch::Tensor data_out_src = torch::from_blob(data_in.data, {data_in.rows, data_in.cols, data_in.channels()}, torch::kInt).to(torch::kLong);  // {0,1,2} = {H,W,C}
    data_out_src = data_out_src.permute({2, 0, 1});  // {0,1,2} = {H,W,C} ===> {0,1,2} = {C,H,W}
    data_out_src = torch::squeeze(data_out_src, /*dim=*/0);  // {C,H,W} ===> {H,W}
    data_out = data_out_src.clone();
    return;
}


// -----------------------------------------------------------------------
// namespace{transforms} -> class{Normalize}(Compose) -> constructor
// -----------------------------------------------------------------------
transforms::Normalize::Normalize(const float mean_, const float std_){
    this->mean = mean_;
    this->std = std_;
}


// -----------------------------------------------------------------------
// namespace{transforms} -> class{Normalize}(Compose) -> function{forward}
// -----------------------------------------------------------------------
void transforms::Normalize::forward(torch::Tensor &data_in, torch::Tensor &data_out){
    torch::Tensor data_out_src = (data_in - this->mean) / this->std;
    data_out = data_out_src.clone();
    return;
}
