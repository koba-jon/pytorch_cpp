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
    return data_out.detach().clone();
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
        std::vector<cv::Mat> multi(this->channels);
        for (int i = 0; i < this->channels; i++){
            multi.at(i) = data_out.clone();
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
    cv::resize(float_mat, float_mat_resize, this->size, 0.0, 0.0, this->interpolation);
    float_mat_resize.convertTo(data_out, data_in.depth());  // continuous ===> discrete
    return;
}


// ---------------------------------------------------------------------------
// namespace{transforms} -> class{ConvertIndex}(Compose) -> constructor
// ---------------------------------------------------------------------------
transforms::ConvertIndex::ConvertIndex(const int before_, const int after_){
    this->before = before_;
    this->after = after_;
}


// ---------------------------------------------------------------------------
// namespace{transforms} -> class{ConvertIndex}(Compose) -> function{forward}
// ---------------------------------------------------------------------------
void transforms::ConvertIndex::forward(cv::Mat &data_in, cv::Mat &data_out){
    size_t width = data_in.cols;
    size_t height = data_in.rows;
    data_out = cv::Mat(cv::Size(width, height), CV_32SC1);
    for (size_t j = 0; j < height; j++){
        for (size_t i = 0; i < width; i++){
            if (data_in.at<int>(j, i) == this->before){
                data_out.at<int>(j, i) = this->after;
            }
            else{
                data_out.at<int>(j, i) = data_in.at<int>(j, i);
            }
        }
    }
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
    data_out = data_out_src.detach().clone();
    return;
}


// ----------------------------------------------------------------------------
// namespace{transforms} -> class{ToTensorLabel}(Compose) -> function{forward}
// ----------------------------------------------------------------------------
void transforms::ToTensorLabel::forward(cv::Mat &data_in, torch::Tensor &data_out){
    torch::Tensor data_out_src = torch::from_blob(data_in.data, {data_in.rows, data_in.cols, data_in.channels()}, torch::kInt).to(torch::kLong);  // {0,1,2} = {H,W,C}
    data_out_src = data_out_src.permute({2, 0, 1});  // {0,1,2} = {H,W,C} ===> {0,1,2} = {C,H,W}
    data_out_src = torch::squeeze(data_out_src, /*dim=*/0);  // {C,H,W} ===> {H,W}
    data_out = data_out_src.detach().clone();
    return;
}


// -----------------------------------------------------------------------
// namespace{transforms} -> class{Normalize}(Compose) -> constructor
// -----------------------------------------------------------------------
transforms::Normalize::Normalize(const float mean_, const float std_){
    this->flag = true;
    this->mean = mean_;
    this->std = std_;
}

transforms::Normalize::Normalize(const float mean_, const std::vector<float> std_){
    this->flag = false;
    this->mean_vec = std::vector<float>(std_.size(), mean_);
    this->std_vec = std_;
}

transforms::Normalize::Normalize(const std::vector<float> mean_, const float std_){
    this->flag = false;
    this->mean_vec = mean_;
    this->std_vec = std::vector<float>(mean_.size(), std_);
}

transforms::Normalize::Normalize(const std::vector<float> mean_, const std::vector<float> std_){
    this->flag = false;
    this->mean_vec = mean_;
    this->std_vec = std_;
}


// -----------------------------------------------------------------------
// namespace{transforms} -> class{Normalize}(Compose) -> function{forward}
// -----------------------------------------------------------------------
void transforms::Normalize::forward(torch::Tensor &data_in, torch::Tensor &data_out){

    torch::Tensor data_out_src;

    if (this->flag){
        data_out_src = (data_in - this->mean) / this->std;
    }
    else{
        size_t counter = 0;
        auto data_per_ch = data_in.chunk(data_in.size(0), /*dim=*/0);  // {C,H,W} ===> {1,H,W} + {1,H,W} + ...
        for (auto &tensor : data_per_ch){
            if (counter == 0){
                data_out_src = (tensor - this->mean_vec.at(counter)) / this->std_vec.at(counter);
            }
            else{
                data_out_src = torch::cat({data_out_src, (tensor - this->mean_vec.at(counter)) / this->std_vec.at(counter)}, /*dim=*/0);  // {i,H,W} + {1,H,W} ===> {i+1,H,W}
            }
            counter++;
        }
    }
    
    data_out = data_out_src.detach().clone();
    
    return;
}
