#include <tuple>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
// For Original Header
#include "augmentation.hpp"
#include "transforms.hpp"

// Define Namespace
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;


// ------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> constructor
// ------------------------------------------------------------------
YOLOAugmentationImpl::YOLOAugmentationImpl(const double jitter_, const double flip_rate_, const double scale_rate_, const double blur_rate_, const double brightness_rate_, const double hue_rate_, const double saturation_rate_, const double shift_rate_, const double crop_rate_){
    this->jitter = jitter_;
    this->flip_rate = (flip_rate_ > 0.0 ? flip_rate_ : -1.0);
    this->scale_rate = (scale_rate_ > 0.0 ? scale_rate_ : -1.0);
    this->blur_rate = (blur_rate_ > 0.0 ? blur_rate_ : -1.0);
    this->brightness_rate = (brightness_rate_ > 0.0 ? brightness_rate_ : -1.0);
    this->hue_rate = (hue_rate_ > 0.0 ? hue_rate_ : -1.0);
    this->saturation_rate = (saturation_rate_ > 0.0 ? saturation_rate_ : -1.0);
    this->shift_rate = (shift_rate_ > 0.0 ? shift_rate_ : -1.0);
    this->crop_rate = (crop_rate_ > 0.0 ? crop_rate_ : -1.0);
    this->mt.push_back(std::mt19937(std::rand()));
}


// --------------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{deepcopy}
// --------------------------------------------------------------------------
void YOLOAugmentationImpl::deepcopy(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){
    data_in1.copyTo(data_out1);
    if (std::get<0>(data_in2).numel() > 0){
        data_out2 = {std::get<0>(data_in2).clone(), std::get<1>(data_in2).clone()};
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }
    return;
}


// --------------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{random_flip}
// --------------------------------------------------------------------------
void YOLOAugmentationImpl::random_flip(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){

    size_t i, j, k;
    size_t i_flip;
    size_t width, height, step, elemSize;
    torch::Tensor ids, coords;

    width = data_in1.cols;
    height = data_in1.rows;
    step = data_in1.step;
    elemSize = data_in1.elemSize();

    // (1) Flipping of Image (x-axis)
    data_out1 = cv::Mat(cv::Size(width, height), data_in1.type());
    for (j = 0; j < height; j++){
        for (i = 0; i < width; i++){
            i_flip = width - i - 1;
            for (k = 0; k < elemSize; k++){
                data_out1.data[j * step + i * elemSize + k] = data_in1.data[j * step + i_flip * elemSize + k];
            }
        }
    }

    // (2) Flipping of Bounding Box (x-axis)
    if (std::get<0>(data_in2).numel() > 0){
        ids = std::get<0>(data_in2).clone();  // ids{BB_n}
        coords = std::get<1>(data_in2).clone();  // coords{BB_n,4}
        coords = coords.permute({1, 0}).contiguous();  // coords{4,BB_n}
        coords[0] = 1.0 - coords[0];  // cx = 1.0 - cx
        coords = coords.permute({1, 0}).contiguous();  // coords{BB_n,4}
        data_out2 = {ids.detach().clone(), coords.detach().clone()};  // {BB_n} (ids), {BB_n,4} (coordinates)
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }

    return;

}


// --------------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{random_scale}
// --------------------------------------------------------------------------
void YOLOAugmentationImpl::random_scale(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){

    size_t width, height;
    std::uniform_real_distribution<double> urand(0.8, 1.2);
    size_t thread_num = omp_get_thread_num();

    // (1) Scaling of Image (x-axis)
    width = (size_t)((double)data_in1.cols * urand(this->mt.at(thread_num)));
    height = data_in1.rows;
    cv::resize(data_in1, data_out1, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
    
    // (2) Scaling of Bounding Box (x-axis)
    if (std::get<0>(data_in2).numel() > 0){
        data_out2 = {std::get<0>(data_in2).clone(), std::get<1>(data_in2).clone()};
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }

    return;

}


// --------------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{random_blur}
// --------------------------------------------------------------------------
void YOLOAugmentationImpl::random_blur(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){

    size_t ksize;
    std::uniform_int_distribution<int> urand(2, 5);
    size_t thread_num = omp_get_thread_num();

    // (1) Blurring of Image
    ksize = urand(this->mt.at(thread_num));
    cv::blur(data_in1, data_out1, cv::Size(ksize, ksize));

    // (2) Blurring of Bounding Box
    if (std::get<0>(data_in2).numel() > 0){
        data_out2 = {std::get<0>(data_in2).clone(), std::get<1>(data_in2).clone()};
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }

    return;

}


// --------------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{random_brightness}
// --------------------------------------------------------------------------
void YOLOAugmentationImpl::random_brightness(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){

    cv::Mat data_mid1, HSV, V;
    std::vector<cv::Mat> HSV_vec;
    std::uniform_real_distribution<double> urand(0.5, 1.5);
    size_t thread_num = omp_get_thread_num();

    // (1) Change Brightness of Image
    data_mid1 = cv::max(cv::min(data_in1, 1.0), 0.0);  // [*,*] ===> [0,1]
    cv::cvtColor(data_mid1, HSV, cv::COLOR_RGB2HSV);  // R[0,1], G[0,1], B[0,1] ===> H[0,360], S[0,1], V[0,1]
    cv::split(HSV, HSV_vec);
    V = HSV_vec.at(2) * (float)urand(this->mt.at(thread_num));  // Change Brightness
    HSV_vec.at(2) = cv::max(cv::min(V, 1.0), 0.0);  // V[*,*] ===> V[0,1]
    cv::merge(HSV_vec, HSV);
    cv::cvtColor(HSV, data_out1, cv::COLOR_HSV2RGB);  // H[0,360], S[0,1], V[0,1] ===> R[0,1], G[0,1], B[0,1]

    // (2) Change Brightness of Bounding Box
    if (std::get<0>(data_in2).numel() > 0){
        data_out2 = {std::get<0>(data_in2).clone(), std::get<1>(data_in2).clone()};
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }

    return;

}


// --------------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{random_hue}
// --------------------------------------------------------------------------
void YOLOAugmentationImpl::random_hue(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){

    cv::Mat data_mid1, HSV, H;
    std::vector<cv::Mat> HSV_vec;
    std::uniform_real_distribution<double> urand(0.8, 1.2);
    size_t thread_num = omp_get_thread_num();

    // (1) Change Hue of Image
    data_mid1 = cv::max(cv::min(data_in1, 1.0), 0.0);  // [*,*] ===> [0,1]
    cv::cvtColor(data_mid1, HSV, cv::COLOR_RGB2HSV);  // R[0,1], G[0,1], B[0,1] ===> H[0,360], S[0,1], V[0,1]
    cv::split(HSV, HSV_vec);
    H = HSV_vec.at(0) * (float)urand(this->mt.at(thread_num));  // Change Hue
    HSV_vec.at(0) = cv::max(cv::min(H, 360.0), 0.0);  // H[*,*] ===> H[0,360]
    cv::merge(HSV_vec, HSV);
    cv::cvtColor(HSV, data_out1, cv::COLOR_HSV2RGB);  // H[0,360], S[0,1], V[0,1] ===> R[0,1], G[0,1], B[0,1]

    // (2) Change Hue of Bounding Box
    if (std::get<0>(data_in2).numel() > 0){
        data_out2 = {std::get<0>(data_in2).clone(), std::get<1>(data_in2).clone()};
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }

    return;

}


// --------------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{random_saturation}
// --------------------------------------------------------------------------
void YOLOAugmentationImpl::random_saturation(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){

    cv::Mat data_mid1, HSV, S;
    std::vector<cv::Mat> HSV_vec;
    std::uniform_real_distribution<double> urand(0.5, 1.5);
    size_t thread_num = omp_get_thread_num();

    // (1) Change Saturation of Image
    data_mid1 = cv::max(cv::min(data_in1, 1.0), 0.0);  // [*,*] ===> [0,1]
    cv::cvtColor(data_mid1, HSV, cv::COLOR_RGB2HSV);  // R[0,1], G[0,1], B[0,1] ===> H[0,360], S[0,1], V[0,1]
    cv::split(HSV, HSV_vec);
    S = HSV_vec.at(1) * (float)urand(this->mt.at(thread_num));  // Change Saturation
    HSV_vec.at(1) = cv::max(cv::min(S, 1.0), 0.0);  // S[*,*] ===> S[0,1]
    cv::merge(HSV_vec, HSV);
    cv::cvtColor(HSV, data_out1, cv::COLOR_HSV2RGB);  // H[0,360], S[0,1], V[0,1] ===> R[0,1], G[0,1], B[0,1]

    // (2) Change Saturation of Bounding Box
    if (std::get<0>(data_in2).numel() > 0){
        data_out2 = {std::get<0>(data_in2).clone(), std::get<1>(data_in2).clone()};
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }

    return;

}


// --------------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{random_shift}
// --------------------------------------------------------------------------
void YOLOAugmentationImpl::random_shift(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){

    int i, j, k, i_in, j_in;
    int dx, dy;
    int width, height, channels;
    float dx_f, dy_f;
    float *data_in1_ptr, *data_out1_ptr;
    std::uniform_real_distribution<double> urand1(-1.0, 1.0);
    std::uniform_real_distribution<double> urand2(0.0, 1.0);
    torch::Tensor x_min, y_min, x_max, y_max;
    torch::Tensor x_min_out, y_min_out, x_max_out, y_max_out;
    torch::Tensor cx, cy, w, h;
    torch::Tensor cx_mid, cy_mid, cx_out, cy_out, w_out, h_out;
    torch::Tensor mask_x, mask_y, mask;
    torch::Tensor ids, coords, ids_out, coords_out;
    size_t thread_num = omp_get_thread_num();

    width = data_in1.cols;
    height = data_in1.rows;
    channels = data_in1.channels();
    dx = (int)(urand1(this->mt.at(thread_num)) * (double)width * this->jitter);
    dy = (int)(urand1(this->mt.at(thread_num)) * (double)height * this->jitter);

    // (1) Shifting of Image
    data_out1 = cv::Mat(cv::Size(width, height), data_in1.type());
    data_in1_ptr = (float*)data_in1.data;
    data_out1_ptr = (float*)data_out1.data;
    for (j = 0; j < height; j++){
        for (i = 0; i < width; i++){
            i_in = i - dx;
            j_in = j - dy;
            if ((i_in < 0) || (width <= i_in) || (j_in < 0) || (height <= j_in)){
                for (k = 0; k < channels; k++){
                    data_out1_ptr[j * width * channels + i * channels + k] = (float)urand2(this->mt.at(thread_num));
                }
            }
            else{
                for (k = 0; k < channels; k++){
                    data_out1_ptr[j * width * channels + i * channels + k] = data_in1_ptr[j_in * width * channels + i_in * channels + k];
                }
            }
        }
    }

    // (2) Shifting of Bounding Box
    if (std::get<0>(data_in2).numel() > 0){
        ids = std::get<0>(data_in2).clone();  // ids{BB_n}
        coords = std::get<1>(data_in2).clone();  // coords{BB_n,4}
        coords = coords.permute({1, 0}).contiguous();  // coords{4,BB_n}
        /*****************************************************************/
        cx = coords[0];  // center of x{BB_n} = [0.0,1.0)
        cy = coords[1];  // center of y{BB_n} = [0.0,1.0)
        w = coords[2];  // width{BB_n} = [0.0,1.0)
        h = coords[3];  // height{BB_n} = [0.0,1.0)
        /*****************************************************************/
        dx_f = (float)dx / (float)width;  // normalized float dx
        dy_f = (float)dy / (float)height;  // normalized float dy
        /*****************************************************************/
        cx_mid = cx + dx_f;  // cx_mid{BB_n}
        cy_mid = cy + dy_f;  // cy_mid{BB_n}
        /*****************************************************************/
        mask_x = (0.0 <= cx_mid) * (cx_mid < 1.0);  // mask_x{BB_n}
        mask_y = (0.0 <= cy_mid) * (cy_mid < 1.0);  // mask_y{BB_n}
        mask = mask_x * mask_y;  // mask{BB_n}
        /*****************************************************************/
        x_min = cx - 0.5 * w;  // x_min{BB_n}
        y_min = cy - 0.5 * h;  // y_min{BB_n}
        x_max = cx + 0.5 * w;  // x_max{BB_n}
        y_max = cy + 0.5 * h;  // y_max{BB_n}
        /*****************************************************************/
        x_min_out = x_min.masked_select(/*mask=*/mask);  // x_min{BB_n} ===> x_min_out{object}
        y_min_out = y_min.masked_select(/*mask=*/mask);  // y_min{BB_n} ===> y_min_out{object}
        x_max_out = x_max.masked_select(/*mask=*/mask);  // x_max{BB_n} ===> x_max_out{object}
        y_max_out = y_max.masked_select(/*mask=*/mask);  // y_max{BB_n} ===> y_max_out{object}
        /*****************************************************************/
        if (x_min_out.numel() > 0){
            x_min_out = (x_min_out + dx_f).clamp(/*min=*/0.0, /*max=*/1.0);  // x_min_out{object}
            y_min_out = (y_min_out + dy_f).clamp(/*min=*/0.0, /*max=*/1.0);  // y_min_out{object}
            x_max_out = (x_max_out + dx_f).clamp(/*min=*/0.0, /*max=*/1.0);  // x_max_out{object}
            y_max_out = (y_max_out + dy_f).clamp(/*min=*/0.0, /*max=*/1.0);  // y_max_out{object}
            /*****************************************************************/
            cx_out = (x_min_out + x_max_out) * 0.5;  // cx_out{object}
            cy_out = (y_min_out + y_max_out) * 0.5;  // cy_out{object}
            w_out = x_max_out - x_min_out;  // w_out{object}
            h_out = y_max_out - y_min_out;  // h_out{object}
            /*****************************************************************/
            ids_out = ids.masked_select(/*mask=*/mask).contiguous();  // ids{BB_n} ===> ids_out{object}
            coords_out = torch::cat({cx_out.unsqueeze(1), cy_out.unsqueeze(1), w_out.unsqueeze(1), h_out.unsqueeze(1)}, /*dim=*/1).contiguous();  // coords_out{object,4}
            data_out2 = {ids_out.detach().clone(), coords_out.detach().clone()};  // {object} (ids), {object,4} (coordinates)
        }
        else{
            data_out2 = {torch::Tensor(), torch::Tensor()};
        }
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }

    return;

}


// --------------------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{random_crop}
// --------------------------------------------------------------------------
void YOLOAugmentationImpl::random_crop(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){

    int i, j, k, i_in, j_in;
    int dx, dy;
    int width, height;
    int step_in, step_out, elemSize;
    int width_out, height_out;
    float dx_f, dy_f, scale_x, scale_y;
    std::uniform_real_distribution<double> urand1(0.6, 1.0);
    std::uniform_real_distribution<double> urand2(0.0, 1.0);
    torch::Tensor x_min, y_min, x_max, y_max;
    torch::Tensor x_min_out, y_min_out, x_max_out, y_max_out;
    torch::Tensor cx, cy, w, h;
    torch::Tensor cx_mid, cy_mid, cx_out, cy_out, w_out, h_out;
    torch::Tensor mask_x, mask_y, mask;
    torch::Tensor ids, coords, ids_out, coords_out;
    size_t thread_num = omp_get_thread_num();

    width = data_in1.cols;
    height = data_in1.rows;
    step_in = data_in1.step;
    elemSize = data_in1.elemSize();
    width_out = (int)(urand1(this->mt.at(thread_num)) * (double)width);
    height_out = (int)(urand1(this->mt.at(thread_num)) * (double)height);
    dx = (int)(urand2(this->mt.at(thread_num)) * (double)(width - width_out));
    dy = (int)(urand2(this->mt.at(thread_num)) * (double)(height - height_out));

    // (1) Cropping of Image
    data_out1 = cv::Mat(cv::Size(width_out, height_out), data_in1.type());
    step_out = data_out1.step;
    for (j = 0; j < height_out; j++){
        for (i = 0; i < width_out; i++){
            i_in = i + dx;
            j_in = j + dy;
            for (k = 0; k < elemSize; k++){
                data_out1.data[j * step_out + i * elemSize + k] = data_in1.data[j_in * step_in + i_in * elemSize + k];
            }
        }
    }

    // (2) Cropping of Bounding Box
    if (std::get<0>(data_in2).numel() > 0){
        ids = std::get<0>(data_in2).clone();  // ids{BB_n}
        coords = std::get<1>(data_in2).clone();  // coords{BB_n,4}
        coords = coords.permute({1, 0}).contiguous();  // coords{4,BB_n}
        /*****************************************************************/
        cx = coords[0];  // center of x{BB_n} = [0.0,1.0)
        cy = coords[1];  // center of y{BB_n} = [0.0,1.0)
        w = coords[2];  // width{BB_n} = [0.0,1.0)
        h = coords[3];  // height{BB_n} = [0.0,1.0)
        /*****************************************************************/
        dx_f = (float)dx / (float)width;  // normalized float dx
        dy_f = (float)dy / (float)height;  // normalized float dy
        scale_x = (float)width_out / (float)width;  // scale factor of x-axis
        scale_y = (float)height_out / (float)height;  // scale factor of y-axis
        /*****************************************************************/
        cx_mid = (cx - dx_f) / scale_x;  // cx_mid{BB_n}
        cy_mid = (cy - dy_f) / scale_y;  // cy_mid{BB_n}
        /*****************************************************************/
        mask_x = (0.0 <= cx_mid) * (cx_mid < 1.0);  // mask_x{BB_n}
        mask_y = (0.0 <= cy_mid) * (cy_mid < 1.0);  // mask_y{BB_n}
        mask = mask_x * mask_y;  // mask{BB_n}
        /*****************************************************************/
        x_min = cx - 0.5 * w;  // x_min{BB_n}
        y_min = cy - 0.5 * h;  // y_min{BB_n}
        x_max = cx + 0.5 * w;  // x_max{BB_n}
        y_max = cy + 0.5 * h;  // y_max{BB_n}
        /*****************************************************************/
        x_min_out = x_min.masked_select(/*mask=*/mask);  // x_min{BB_n} ===> x_min_out{object}
        y_min_out = y_min.masked_select(/*mask=*/mask);  // y_min{BB_n} ===> y_min_out{object}
        x_max_out = x_max.masked_select(/*mask=*/mask);  // x_max{BB_n} ===> x_max_out{object}
        y_max_out = y_max.masked_select(/*mask=*/mask);  // y_max{BB_n} ===> y_max_out{object}
        /*****************************************************************/
        if (x_min_out.numel() > 0){
            x_min_out = ((x_min_out - dx_f) / scale_x).clamp(/*min=*/0.0, /*max=*/1.0);  // x_min_out{object}
            y_min_out = ((y_min_out - dy_f) / scale_y).clamp(/*min=*/0.0, /*max=*/1.0);  // y_min_out{object}
            x_max_out = ((x_max_out - dx_f) / scale_x).clamp(/*min=*/0.0, /*max=*/1.0);  // x_max_out{object}
            y_max_out = ((y_max_out - dy_f) / scale_y).clamp(/*min=*/0.0, /*max=*/1.0);  // y_max_out{object}
            /*****************************************************************/
            cx_out = (x_min_out + x_max_out) * 0.5;  // cx_out{object}
            cy_out = (y_min_out + y_max_out) * 0.5;  // cy_out{object}
            w_out = x_max_out - x_min_out;  // w_out{object}
            h_out = y_max_out - y_min_out;  // h_out{object}
            /*****************************************************************/
            ids_out = ids.masked_select(/*mask=*/mask).contiguous();  // ids{BB_n} ===> ids_out{object}
            coords_out = torch::cat({cx_out.unsqueeze(1), cy_out.unsqueeze(1), w_out.unsqueeze(1), h_out.unsqueeze(1)}, /*dim=*/1).contiguous();  // coords_out{object,4}
            data_out2 = {ids_out.detach().clone(), coords_out.detach().clone()};  // {object} (ids), {object,4} (coordinates)
        }
        else{
            data_out2 = {torch::Tensor(), torch::Tensor()};
        }
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }
    
    return;

}


// -----------------------------------------------------------------
// class{YOLOAugmentationImpl}(transforms::ComposeImpl) -> function{forward}
// -----------------------------------------------------------------
void YOLOAugmentationImpl::forward(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){

    // --------------------------------------
    // 1. Parallel Processing Settings
    // --------------------------------------

    // (1) Get Global Parameter
    size_t threads = omp_get_num_threads();
    while (threads > this->mt.size()){
        #pragma omp critical
        {
            this->mt.push_back(std::mt19937(std::rand()));
        }
    }

    // (2) Get Local Parameter
    size_t thread_num = omp_get_thread_num();

    // --------------------------------------
    // 2. Data Augmentation
    // --------------------------------------

    cv::Mat data_mid1;
    std::tuple<torch::Tensor, torch::Tensor> data_mid2;
    std::uniform_real_distribution<double> urand(0.0, 1.0);

    // Copy
    data_in1.convertTo(data_mid1, CV_32F);  // discrete ===> continuous
    data_mid1 *= 1.0 / (std::pow(2.0, data_in1.elemSize1()*8) - 1.0);  // [0,255] or [0,65535] ===> [0,1]
    if (std::get<0>(data_in2).numel() > 0){
        data_mid2 = {std::get<0>(data_in2).clone(), std::get<1>(data_in2).clone()};
    }
    else{
        data_mid2 = {torch::Tensor(), torch::Tensor()};
    }

    // (1) Flipping
    if (urand(this->mt.at(thread_num)) <= this->flip_rate){
        this->random_flip(data_mid1, data_mid2, data_out1, data_out2);
        this->deepcopy(data_out1, data_out2, data_mid1, data_mid2);
    }

    // (2) Scaling
    if (urand(this->mt.at(thread_num)) <= this->scale_rate){
        this->random_scale(data_mid1, data_mid2, data_out1, data_out2);
        this->deepcopy(data_out1, data_out2, data_mid1, data_mid2);
    }

    // (3) Blurring
    if (urand(this->mt.at(thread_num)) <= this->blur_rate){
        this->random_blur(data_mid1, data_mid2, data_out1, data_out2);
        this->deepcopy(data_out1, data_out2, data_mid1, data_mid2);
    }

    // (4) Change Brightness
    if (urand(this->mt.at(thread_num)) <= this->brightness_rate){
        this->random_brightness(data_mid1, data_mid2, data_out1, data_out2);
        this->deepcopy(data_out1, data_out2, data_mid1, data_mid2);
    }

    // (5) Change Hue
    if (urand(this->mt.at(thread_num)) <= this->hue_rate){
        this->random_hue(data_mid1, data_mid2, data_out1, data_out2);
        this->deepcopy(data_out1, data_out2, data_mid1, data_mid2);
    }

    // (6) Change Saturation
    if (urand(this->mt.at(thread_num)) <= this->saturation_rate){
        this->random_saturation(data_mid1, data_mid2, data_out1, data_out2);
        this->deepcopy(data_out1, data_out2, data_mid1, data_mid2);
    }

    // (7) Shifting
    if (urand(this->mt.at(thread_num)) <= this->shift_rate){
        this->random_shift(data_mid1, data_mid2, data_out1, data_out2);
        this->deepcopy(data_out1, data_out2, data_mid1, data_mid2);
    }

    // (8) Cropping
    if (urand(this->mt.at(thread_num)) <= this->crop_rate){
        this->random_crop(data_mid1, data_mid2, data_out1, data_out2);
        this->deepcopy(data_out1, data_out2, data_mid1, data_mid2);
    }

    // Copy
    data_mid1 *= std::pow(2.0, data_in1.elemSize1()*8) - 1.0;  // [0,1] ===> [0,255] or [0,65535]
    data_mid1.convertTo(data_out1, data_in1.depth());  // continuous ===> discrete
    if (std::get<0>(data_mid2).numel() > 0){
        data_out2 = {std::get<0>(data_mid2).clone(), std::get<1>(data_mid2).clone()};
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }

    return;

}


// ------------------------------------------------------------------
// class{YOLOBatchAugmentation} -> constructor
// ------------------------------------------------------------------
YOLOBatchAugmentation::YOLOBatchAugmentation(const double mosaic_rate_, const double mixup_rate_){
    this->mosaic_rate = mosaic_rate_;
    this->mixup_rate = mixup_rate_;
    this->mt = std::mt19937(std::rand());
}


// --------------------------------------------------------------------------
// class{YOLOBatchAugmentation} -> function{random_mosaic}
// --------------------------------------------------------------------------
std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>> YOLOBatchAugmentation::random_mosaic(torch::Tensor data_in1, std::vector<std::tuple<torch::Tensor, torch::Tensor>> data_in2){

    torch::Device device = data_in1.device();
    long int mini_batch_size = data_in1.size(0);
    std::uniform_int_distribution<size_t> irand(0, mini_batch_size - 1);
    size_t idx;
    torch::Tensor top_left, top_right, bottom_left, bottom_right, top, bottom, data_mid, data_out1;
    torch::Tensor ids, coords, idss, coordss;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> data_out2;
    
    data_out1 = torch::empty({0, data_in1.size(1), data_in1.size(2), data_in1.size(3)}, torch::kFloat).to(device);  // {0,C,H,W}
    data_out2 = std::vector<std::tuple<torch::Tensor, torch::Tensor>>(mini_batch_size);
    for (long int i = 0; i < mini_batch_size; i++){

        idss = torch::empty({0}, torch::kLong);
        coordss = torch::empty({0, 4}, torch::kFloat);

        // (1) Top left
        top_left = data_in1.index({Slice(i, i + 1)}); // {1,C,H,W}
        if (std::get<0>(data_in2[i]).numel() > 0){
            ids = std::get<0>(data_in2[i]).clone();  // {BB_n}
            coords = std::get<1>(data_in2[i]).clone();  // {BB_n,4}
            coords.index_put_({Slice(), 0}, coords.index({Slice(), 0}) * 0.5);
            coords.index_put_({Slice(), 1}, coords.index({Slice(), 1}) * 0.5);
            coords.index_put_({Slice(), 2}, coords.index({Slice(), 2}) * 0.5);
            coords.index_put_({Slice(), 3}, coords.index({Slice(), 3}) * 0.5);
            idss = torch::cat({idss, ids}, 0);
            coordss = torch::cat({coordss, coords}, 0);
        }

        // (2) Top right
        idx = irand(this->mt);
        top_right = data_in1.index({Slice(idx, idx + 1)}); // {1,C,H,W}
        if (std::get<0>(data_in2[idx]).numel() > 0){
            ids = std::get<0>(data_in2[idx]).clone();  // {BB_n}
            coords = std::get<1>(data_in2[idx]).clone();  // {BB_n,4}
            coords.index_put_({Slice(), 0}, coords.index({Slice(), 0}) * 0.5 + 0.5);
            coords.index_put_({Slice(), 1}, coords.index({Slice(), 1}) * 0.5);
            coords.index_put_({Slice(), 2}, coords.index({Slice(), 2}) * 0.5);
            coords.index_put_({Slice(), 3}, coords.index({Slice(), 3}) * 0.5);
            idss = torch::cat({idss, ids}, 0);
            coordss = torch::cat({coordss, coords}, 0);
        }

        // (3) Bottom left
        idx = irand(this->mt);
        bottom_left = data_in1.index({Slice(idx, idx + 1)}); // {1,C,H,W}
        if (std::get<0>(data_in2[idx]).numel() > 0){
            ids = std::get<0>(data_in2[idx]).clone();  // {BB_n}
            coords = std::get<1>(data_in2[idx]).clone();  // {BB_n,4}
            coords.index_put_({Slice(), 0}, coords.index({Slice(), 0}) * 0.5);
            coords.index_put_({Slice(), 1}, coords.index({Slice(), 1}) * 0.5 + 0.5);
            coords.index_put_({Slice(), 2}, coords.index({Slice(), 2}) * 0.5);
            coords.index_put_({Slice(), 3}, coords.index({Slice(), 3}) * 0.5);
            idss = torch::cat({idss, ids}, 0);
            coordss = torch::cat({coordss, coords}, 0);
        }

        // (4) Bottom right
        idx = irand(this->mt);
        bottom_right = data_in1.index({Slice(idx, idx + 1)}); // {1,C,H,W}
        if (std::get<0>(data_in2[idx]).numel() > 0){
            ids = std::get<0>(data_in2[idx]).clone();  // {BB_n}
            coords = std::get<1>(data_in2[idx]).clone();  // {BB_n,4}
            coords.index_put_({Slice(), 0}, coords.index({Slice(), 0}) * 0.5 + 0.5);
            coords.index_put_({Slice(), 1}, coords.index({Slice(), 1}) * 0.5 + 0.5);
            coords.index_put_({Slice(), 2}, coords.index({Slice(), 2}) * 0.5);
            coords.index_put_({Slice(), 3}, coords.index({Slice(), 3}) * 0.5);
            idss = torch::cat({idss, ids}, 0);
            coordss = torch::cat({coordss, coords}, 0);
        }

        top = torch::cat({top_left, top_right}, 3); // {1,C,H,2W}
        bottom = torch::cat({bottom_left, bottom_right}, 3);  // {1,C,H,2W}
        data_mid = torch::cat({top, bottom}, 2);  // {1,C,2H,2W}
        data_out1 = torch::cat({data_out1, F::interpolate(data_mid, F::InterpolateFuncOptions().size(std::vector<long int>({data_in1.size(2), data_in1.size(3)})).mode(torch::kBilinear).align_corners(false))}, 0);  // {i,C,H,W}
        data_out2[i] = {idss, coordss};

    }

    return {data_out1, data_out2};

}


// --------------------------------------------------------------------------
// class{YOLOBatchAugmentation} -> function{random_mixup}
// --------------------------------------------------------------------------
std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>> YOLOBatchAugmentation::random_mixup(torch::Tensor data_in1, std::vector<std::tuple<torch::Tensor, torch::Tensor>> data_in2){

    torch::Device device = data_in1.device();
    long int mini_batch_size = data_in1.size(0);
    std::uniform_int_distribution<size_t> irand(0, mini_batch_size - 1);
    torch::Tensor data_out1, image1, image2;
    torch::Tensor ids, coords, idss, coordss;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> data_out2;
    size_t j;

    data_out1 = torch::empty({0, data_in1.size(1), data_in1.size(2), data_in1.size(3)}, torch::kFloat).to(device);  // {0,C,H,W}
    data_out2 = std::vector<std::tuple<torch::Tensor, torch::Tensor>>(mini_batch_size);
    for (long int i = 0; i < mini_batch_size; i++){

        idss = torch::empty({0}, torch::kLong);
        coordss = torch::empty({0, 4}, torch::kFloat);

        // (1) image 1
        image1 = data_in1.index({Slice(i, i + 1)});  // {1,C,H,W}
        if (std::get<0>(data_in2[i]).numel() > 0){
            std::tie(ids, coords) = data_in2[i];
            idss = torch::cat({idss, ids}, 0);
            coordss = torch::cat({coordss, coords}, 0);
        }

        // (2) image 2
        j = irand(this->mt);
        image2 = data_in1.index({Slice(j, j + 1)});  // {1,C,H,W}
        if (std::get<0>(data_in2[j]).numel() > 0){
            std::tie(ids, coords) = data_in2[j];
            idss = torch::cat({idss, ids}, 0);
            coordss = torch::cat({coordss, coords}, 0);
        }

        data_out1 = torch::cat({data_out1, image1 * 0.5 + image2 * 0.5}, 0);  // {i,C,H,W}
        data_out2[i] = {idss, coordss};

    }

    return {data_out1, data_out2};

}



// -----------------------------------------------------------------
// class{YOLOBatchAugmentation} -> function{forward}
// -----------------------------------------------------------------
std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>> YOLOBatchAugmentation::forward(torch::Tensor data_in1, std::vector<std::tuple<torch::Tensor, torch::Tensor>> data_in2){

    torch::Tensor data_out1;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> data_out2;
    std::uniform_real_distribution<double> urand(0.0, 1.0);

    data_out1 = data_in1;
    data_out2 = data_in2;

    // (1) Mosaic
    if (urand(this->mt) <= this->mosaic_rate){
        std::tie(data_out1, data_out2) = this->random_mosaic(data_out1, data_out2);
    }

    // (2) Mixup
    if (urand(this->mt) <= this->mixup_rate){
        std::tie(data_out1, data_out2) = this->random_mixup(data_out1, data_out2);
    }

    return {data_out1, data_out2};

}
