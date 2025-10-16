#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <typeinfo>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;


// ---------------------------------------------------------
// struct{ConvBlockImpl}(nn::Module) -> constructor
// ---------------------------------------------------------
ConvBlockImpl::ConvBlockImpl(const size_t in_nc, const size_t out_nc, const size_t kernel, const size_t stride, const size_t padding, const bool BN, const bool SiLU, const bool bias){
    this->sq->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, kernel).stride(stride).padding(padding).bias(bias)));
    if (BN) {
        this->sq->push_back(nn::BatchNorm2d(out_nc));
    }
    if (SiLU) {
        this->sq->push_back(nn::SiLU());
    }
    register_module("ConvBlock", this->sq);
}


// ---------------------------------------------------------------
// struct{ConvBlockImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------------
torch::Tensor ConvBlockImpl::forward(torch::Tensor x){
    return this->sq->forward(x);
}


// ---------------------------------------------------------
// struct{BottleneckImpl}(nn::Module) -> constructor
// ---------------------------------------------------------
BottleneckImpl::BottleneckImpl(const size_t in_nc, const size_t out_nc, const bool shortcut, const double expansion){
    size_t hidden_nc;
    hidden_nc = std::max<size_t>(1, std::round(out_nc * expansion));
    this->cv1 = register_module("cv1", ConvBlock(in_nc, hidden_nc, 1, 1, 0));
    this->cv2 = register_module("cv2", ConvBlock(hidden_nc, out_nc, 3, 1, 1));
    this->residual = (shortcut && (in_nc == out_nc));
}


// ---------------------------------------------------------------
// struct{BottleneckImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------------
torch::Tensor BottleneckImpl::forward(torch::Tensor x){
    torch::Tensor out;
    out = this->cv2->forward(this->cv1->forward(x));
    if (this->residual){
        out = out + x;
    }
    return out;
}


// ---------------------------------------------------------
// struct{C3Impl}(nn::Module) -> constructor
// ---------------------------------------------------------
C3Impl::C3Impl(const size_t in_nc, const size_t out_nc, const size_t n, const bool shortcut){

    size_t hidden_nc;

    hidden_nc = std::max<size_t>(1, out_nc / 2);
    this->cv1 = register_module("cv1", ConvBlock(in_nc, hidden_nc, 1, 1, 0));
    this->cv2 = register_module("cv2", ConvBlock(in_nc, hidden_nc, 1, 1, 0));
    this->cv3 = register_module("cv3", ConvBlock(hidden_nc * 2, out_nc, 1, 1, 0));

    for (size_t i = 0; i < n; i++){
        this->m->push_back(Bottleneck(hidden_nc, hidden_nc, shortcut, 0.5));
    }
    register_module("m", this->m);

}


// ---------------------------------------------------------------
// struct{C3Impl}(nn::Module) -> function{forward}
// ---------------------------------------------------------------
torch::Tensor C3Impl::forward(torch::Tensor x){
    torch::Tensor y1, y2, out;
    y1 = this->m->forward(this->cv1->forward(x));
    y2 = this->cv2->forward(x);
    out = torch::cat({y1, y2}, 1);
    out = this->cv3->forward(out);
    return out;
}


// ---------------------------------------------------------
// struct{SPPFImpl}(nn::Module) -> constructor
// ---------------------------------------------------------
SPPFImpl::SPPFImpl(const size_t in_nc, const size_t out_nc, const size_t kernel){
    size_t hidden_nc;
    hidden_nc = std::max<size_t>(1, out_nc / 2);
    this->cv1 = register_module("cv1", ConvBlock(in_nc, hidden_nc, 1, 1, 0));
    this->cv2 = register_module("cv2", ConvBlock(hidden_nc * 4, out_nc, 1, 1, 0));
    this->maxpool = register_module("maxpool", nn::MaxPool2d(nn::MaxPool2dOptions(kernel).stride(1).padding(kernel / 2)));
}


// ---------------------------------------------------------------
// struct{SPPFImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------------
torch::Tensor SPPFImpl::forward(torch::Tensor x){
    torch::Tensor y, y1, y2, y3, out;
    y = this->cv1->forward(x);
    y1 = this->maxpool->forward(y);
    y2 = this->maxpool->forward(y1);
    y3 = this->maxpool->forward(y2);
    out = torch::cat({y, y1, y2, y3}, 1);
    out = this->cv2->forward(out);
    return out;
}


// ----------------------------------------------------------------------
// struct{YOLOv5Impl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
YOLOv5Impl::YOLOv5Impl(po::variables_map &vm){

    size_t nc = vm["nc"].as<size_t>();  // the number of image channels
    size_t na = vm["na"].as<size_t>();  // the number of anchor
    size_t class_num = vm["class_num"].as<size_t>();  // total classes
    long int final_features = (long int)(na * (class_num + 5));  // anchors * (total classes + 5=len[t_x, t_y, t_w, t_h, confidence])
    std::string model = vm["model"].as<std::string>();  // total classes
    double depth, width;

    if (model == "yolov5n"){
        depth = 0.33;
        width = 0.25;
    }
    else if (model == "yolov5s"){
        depth = 0.33;
        width = 0.50;
    }
    else if (model == "yolov5m"){
        depth = 0.67;
        width = 0.75;
    }
    else if (model == "yolov5l"){
        depth = 1.0;
        width = 1.0;
    }
    else if (model == "yolov5x"){
        depth = 1.33;
        width = 1.25;
    }
    else{
        std::cerr << "Error : The name of model is " << model << '.' << std::endl;
        std::cerr << "Error : Please choose yolov5n, yolov5s, yolov5m, yolov5l or yolov5x." << std::endl;
        std::exit(1);
    }
    
    this->conv_0 = register_module("conv_0", ConvBlock(nc, mul(64, width), 6, 2, 2));
    this->conv_1 = register_module("conv_1", ConvBlock(mul(64, width), mul(128, width), 3, 2, 1));
    this->c3_2 = register_module("c3_2", C3(mul(128, width), mul(128, width), mul(3, depth)));
    this->conv_3 = register_module("conv_3", ConvBlock(mul(128, width), mul(256, width), 3, 2, 1));
    this->c3_4 = register_module("c3_4", C3(mul(256, width), mul(256, width), mul(6, depth)));
    this->conv_5 = register_module("conv_5", ConvBlock(mul(256, width), mul(512, width), 3, 2, 1));
    this->c3_6 = register_module("c3_6", C3(mul(512, width), mul(512, width), mul(9, depth)));
    this->conv_7 = register_module("conv_7", ConvBlock(mul(512, width), mul(1024, width), 3, 2, 1));
    this->c3_8 = register_module("c3_8", C3(mul(1024, width), mul(1024, width), mul(3, depth)));
    this->sppf_9 = register_module("sppf_9", SPPF(mul(1024, width), mul(1024, width), 5));

    this->head_conv_10 = register_module("head_conv_10", ConvBlock(mul(1024, width), mul(512, width), 1, 1, 0));
    this->head_c3_13 = register_module("head_c3_13", C3(mul(512, width) + mul(512, width), mul(512, width), mul(3, depth), false));

    this->head_conv_14 = register_module("head_conv_14", ConvBlock(mul(512, width), mul(256, width), 1, 1, 0));
    this->head_c3_17 = register_module("head_c3_17", C3(mul(256, width) + mul(256, width), mul(256, width), mul(3, depth), false));

    this->head_conv_18 = register_module("head_conv_18", ConvBlock(mul(256, width), mul(256, width), 3, 2, 1));
    this->head_c3_20 = register_module("head_c3_20", C3(mul(256, width) + mul(256, width), mul(512, width), mul(3, depth), false));

    this->head_conv_21 = register_module("head_conv_21", ConvBlock(mul(512, width), mul(512, width), 3, 2, 1));
    this->head_c3_23 = register_module("head_c3_23", C3(mul(512, width) + mul(512, width), mul(1024, width), mul(3, depth), false));

    this->detect_small = register_module("detect_small", nn::Conv2d(nn::Conv2dOptions(mul(256, width), final_features, 1).stride(1).padding(0).bias(true)));
    this->detect_medium = register_module("detect_medium", nn::Conv2d(nn::Conv2dOptions(mul(512, width), final_features, 1).stride(1).padding(0).bias(true)));
    this->detect_large = register_module("detect_large", nn::Conv2d(nn::Conv2dOptions(mul(1024, width), final_features, 1).stride(1).padding(0).bias(true)));

}


// ---------------------------------------------------------
// struct{YOLOv5Impl}(nn::Module) -> function{mul}
// ---------------------------------------------------------
size_t YOLOv5Impl::mul(const double base, const double scale){
    return std::max<size_t>(1, (size_t)(std::round(base * scale)));
}

// ---------------------------------------------------------
// struct{YOLOv5Impl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
std::vector<torch::Tensor> YOLOv5Impl::forward(torch::Tensor x){

    torch::Tensor x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23;
    torch::Tensor small, medium, large;
    std::vector<torch::Tensor> out;

    x0 = this->conv_0->forward(x);
    x1 = this->conv_1->forward(x0);
    x2 = this->c3_2->forward(x1);
    x3 = this->conv_3->forward(x2);
    x4 = this->c3_4->forward(x3);
    x5 = this->conv_5->forward(x4);
    x6 = this->c3_6->forward(x5);
    x7 = this->conv_7->forward(x6);
    x8 = this->c3_8->forward(x7);
    x9 = this->sppf_9->forward(x8);

    x10 = this->head_conv_10->forward(x9);
    x11 = UpSampling(x10, {x6.size(2), x6.size(3)});
    x12 = torch::cat({x11, x6}, 1);
    x13 = this->head_c3_13->forward(x12);

    x14 = this->head_conv_14->forward(x13);
    x15 = UpSampling(x14, {x4.size(2), x4.size(3)});
    x16 = torch::cat({x15, x4}, 1);
    x17 = this->head_c3_17->forward(x16);

    x18 = this->head_conv_18->forward(x17);
    x19 = torch::cat({x18, x14}, 1);
    x20 = this->head_c3_20->forward(x19);

    x21 = this->head_conv_21->forward(x20);
    x22 = torch::cat({x21, x10}, 1);
    x23 = this->head_c3_23->forward(x22);

    small = this->detect_small->forward(x17);
    small = small.permute({0, 2, 3, 1}).contiguous();  // {N,A*(5+CN),G,G} ===> {N,G,G,A*(5+CN)}
    medium = this->detect_medium->forward(x20);
    medium = medium.permute({0, 2, 3, 1}).contiguous();  // {N,A*(5+CN),G,G} ===> {N,G,G,A*(5+CN)}
    large = this->detect_large->forward(x23);
    large = large.permute({0, 2, 3, 1}).contiguous();  // {N,A*(5+CN),G,G} ===> {N,G,G,A*(5+CN)}

    out.push_back(small);
    out.push_back(medium);
    out.push_back(large);

    return out;

}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl))) {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::kaiming_normal_(*w, 0.0, /*mode=*/torch::kFanIn, /*nonlinearity*/torch::kLeakyReLU);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::BatchNorm2d)) || (typeid(m) == typeid(nn::BatchNorm2dImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::constant_(*w, /*weight=*/1.0);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}


// ----------------------------
// function{UpSampling}
// ----------------------------
torch::Tensor UpSampling(torch::Tensor x, const std::vector<long int> shape){
    torch::Tensor out;
    out = F::interpolate(x, F::InterpolateFuncOptions().size(shape).mode(torch::kNearest));
    return out;
}

