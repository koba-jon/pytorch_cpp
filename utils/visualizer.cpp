#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <png++/png.hpp>
// For Original Header
#include "visualizer.hpp"

// Define Namespace
namespace fs = std::filesystem;


// ----------------------------------------------------------
// namespace{visualizer} -> function{save_image}
// ----------------------------------------------------------
void visualizer::save_image(const torch::Tensor image, const std::string path, const std::pair<float, float> range, const size_t cols, const size_t padding, const size_t bits){

    // (0) Initialization and Declaration
    size_t i, j, k, l;
    size_t i_dev, j_dev;
    size_t width, height, channels, mini_batch_size;
    size_t width_out, height_out;
    size_t ncol, nrow;
    int mtype_in, mtype_out;
    cv::Mat float_mat, normal_mat, bit_mat, RGB, BGR;
    cv::Mat sample, output;
    std::vector<cv::Mat> samples;
    torch::Tensor tensor_sq, tensor_per;

    // (1) Get Tensor Size
    mini_batch_size = image.size(0);
    channels = image.size(1);
    height = image.size(2);
    width = image.size(3);

    // (2) Judge the number of channels and bits
    mtype_in = CV_32FC1;
    if (channels == 1){
        if (bits == 8){
            mtype_out = CV_8UC1;
        }
        else if (bits == 16){
            mtype_out = CV_16UC1;
        }
        else{
            std::cerr << "Error : Bits of the image to be saved is inappropriate." << std::endl;
            std::exit(1);
        }
    }
    else if (channels == 3){
        if (bits == 8){
            mtype_out = CV_8UC3;
        }
        else if (bits == 16){
            mtype_out = CV_16UC3;
        }
        else{
            std::cerr << "Error : Bits of the image to be saved is inappropriate." << std::endl;
            std::exit(1);
        }
    }
    else{
        std::cerr << "Error : Channels of the image to be saved is inappropriate." << std::endl;
        std::exit(1);
    }

    // (3) Add images to the array
    i = 0;
    samples = std::vector<cv::Mat>(mini_batch_size);
    auto mini_batch = image.clamp(/*min=*/range.first, /*max=*/range.second).contiguous().to(torch::kCPU).chunk(mini_batch_size, /*dim=*/0);  // {N,C,H,W} ===> {1,C,H,W} + {1,C,H,W} + ...
    for (auto &tensor : mini_batch){
        tensor_sq = torch::squeeze(tensor, /*dim=*/0);  // {1,C,H,W} ===> {C,H,W}
        tensor_per = tensor_sq.permute({1, 2, 0});  // {C,H,W} ===> {H,W,C}
        if (channels == 3){
            auto tensor_vec = tensor_per.chunk(channels, /*dim=*/2);  // {H,W,3} ===> {H,W,1} + {H,W,1} + {H,W,1}
            std::vector<cv::Mat> mv;
            for (auto &tensor_channel : tensor_vec){
                mv.push_back(cv::Mat(cv::Size(width, height), mtype_in, tensor_channel.data_ptr<float>()));  // torch::Tensor ===> cv::Mat
            }
            cv::merge(mv, float_mat);  // {H,W,1} + {H,W,1} + {H,W,1} ===> {H,W,3}
        }
        else{
            float_mat = cv::Mat(cv::Size(width, height), mtype_in, tensor_per.data_ptr<float>());  // torch::Tensor ===> cv::Mat
        }
        normal_mat = (float_mat - range.first) / (float)(range.second - range.first);  // [range.first, range.second] ===> [0,1]
        bit_mat = normal_mat * (std::pow(2.0, bits) - 1.0);  // [0,1] ===> [0,255] or [0,65535]
        bit_mat.convertTo(sample, mtype_out);  // {32F} ===> {8U} or {16U}
        if (channels == 3){
            RGB = sample;
            cv::cvtColor(RGB, BGR, cv::COLOR_RGB2BGR);  // {R,G,B} ===> {B,G,R}
            sample = BGR;
        }
        sample.copyTo(samples.at(i));
        i++;
    }

    // (4) Output Image Information
    ncol = (mini_batch_size < cols) ? mini_batch_size : cols;
    width_out = width * ncol + padding * (ncol + 1);
    nrow = 1 + (mini_batch_size - 1) / ncol;
    height_out =  height * nrow + padding * (nrow + 1);

    // (5) Value Substitution for Output Image
    output = cv::Mat(cv::Size(width_out, height_out), mtype_out, cv::Scalar::all(0));
    for (l = 0; l < mini_batch_size; l++){
        sample = samples.at(l);
        i_dev = (l % ncol) * width + padding * (l % ncol + 1);
        j_dev = (l / ncol) * height + padding * (l / ncol + 1);
        for (j = 0; j < height; j++){
            for (i = 0; i < width; i++){
                for (k = 0; k < sample.elemSize(); k++){
                    output.data[(j + j_dev) * output.step + (i + i_dev) * output.elemSize() + k] = sample.data[j * sample.step + i * sample.elemSize() + k];
                }
            }
        }
    }

    // (6) Image Output
    cv::imwrite(path, output);

    // End Processing
    return;

}


// ----------------------------------------------------------
// namespace{visualizer} -> function{save_label}
// ----------------------------------------------------------
void visualizer::save_label(const torch::Tensor label, const std::string path, const std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette, const size_t cols, const size_t padding){

    // (0) Initialization and Declaration
    size_t i, j, k;
    size_t i_dev, j_dev;
    size_t width, height, mini_batch_size;
    size_t width_out, height_out;
    size_t ncol, nrow;
    unsigned char R, G, B;
    cv::Mat sample;
    std::vector<cv::Mat> samples;
    torch::Tensor tensor_sq, tensor_per;
    png::image<png::index_pixel> output;
    png::palette pal;
    std::tuple<unsigned char, unsigned char, unsigned char> pal_one;

    // (1) Get Tensor Size
    mini_batch_size = label.size(0);
    height = label.size(2);
    width = label.size(3);

    // (2) Add images to the array
    i = 0;
    samples = std::vector<cv::Mat>(mini_batch_size);
    auto mini_batch = label.contiguous().to(torch::kCPU).chunk(mini_batch_size, /*dim=*/0);  // {N,1,H,W} ===> {1,1,H,W} + {1,1,H,W} + ...
    for (auto &tensor : mini_batch){
        tensor_sq = torch::squeeze(tensor, /*dim=*/0);  // {1,1,H,W} ===> {1,H,W}
        tensor_per = tensor_sq.permute({1, 2, 0});  // {1,H,W} ===> {H,W,1}
        sample = cv::Mat(cv::Size(width, height), CV_32SC1, tensor_per.to(torch::kInt).data_ptr<int>());  // torch::Tensor ===> cv::Mat
        sample.copyTo(samples.at(i));
        i++;
    }

    // (3) Output Image Information
    ncol = (mini_batch_size < cols) ? mini_batch_size : cols;
    width_out = width * ncol + padding * (ncol + 1);
    nrow = 1 + (mini_batch_size - 1) / ncol;
    height_out =  height * nrow + padding * (nrow + 1);

    // (4) Palette and Value Initialization for Index Image
    output = png::image<png::index_pixel>(width_out, height_out);
    pal = png::palette(label_palette.size());
    for (i = 0; i < pal.size(); i++){
        pal_one = label_palette.at(i);
        R = std::get<0>(pal_one);
        G = std::get<1>(pal_one);
        B = std::get<2>(pal_one);
        pal[i] = png::color(R, G, B);
    }
    output.set_palette(pal);
    for (j = 0; j < height_out; j++){
        for (i = 0; i < width_out; i++){
            output[j][i] = 0;
        }
    }

    // (5) Value Substitution for Output Image
    for (k = 0; k < mini_batch_size; k++){
        sample = samples.at(k);
        i_dev = (k % ncol) * width + padding * (k % ncol + 1);
        j_dev = (k / ncol) * height + padding * (k / ncol + 1);
        for (j = 0; j < height; j++){
            for (i = 0; i < width; i++){
                output[j+j_dev][i+i_dev] = sample.at<int>(j, i);
            }
        }
    }

    // (6) Image Output
    output.write(path);

    // End Processing
    return;

}


// ----------------------------------------------------------
// namespace{visualizer} -> class{graph} -> constructor
// ----------------------------------------------------------
visualizer::graph::graph(const std::string dir_, const std::string gname_, const std::vector<std::string> label_){
    this->flag = false;
    this->dir = dir_;
    this->data_dir = this->dir + "/data";
    this->gname = gname_;
    this->graph_fname= this->dir + '/' + this->gname + ".png";
    this->data_fname= this->data_dir + '/' + this->gname + ".dat";
    this->label = label_;
    fs::create_directories(this->dir);
    fs::create_directories(this->data_dir);
}


// ----------------------------------------------------------
// namespace{visualizer} -> class{graph} -> function{plot}
// ----------------------------------------------------------
void visualizer::graph::plot(const float base, const std::vector<float> value){

    // (1) Value Output
    std::ofstream ofs(this->data_fname, std::ios::app);
    ofs << base << std::flush;
    for (auto &v : value){
        ofs << ' ' << v << std::flush;
    }
    ofs << std::endl;
    ofs.close();

    // (2) Graph Output
    if (this->flag){
        FILE* gp;
        gp = popen("gnuplot -persist", "w");
        fprintf(gp, "set terminal png\n");
        fprintf(gp, "set output '%s'\n", this->graph_fname.c_str());
        fprintf(gp, "plot ");
        for (size_t i = 0; i < this->label.size() - 1; i++){
            fprintf(gp, "'%s' using 1:%zu ti '%s' with lines,", this->data_fname.c_str(), i + 2, this->label.at(i).c_str());
        }
        fprintf(gp, "'%s' using 1:%zu ti '%s' with lines\n", this->data_fname.c_str(), this->label.size() + 1, this->label.at(this->label.size() - 1).c_str());
        pclose(gp);
    }

    // (3) Setting for after the Second Time
    this->flag = true;

    // End Processing
    return;

}