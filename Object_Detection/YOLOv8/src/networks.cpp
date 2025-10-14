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
// struct{C2fImpl}(nn::Module) -> constructor
// ---------------------------------------------------------
C2fImpl::C2fImpl(const size_t in_nc, const size_t out_nc, const size_t n, const bool shortcut){

    this->hidden_nc = std::max<size_t>(1, out_nc / 2);
    this->cv1 = register_module("cv1", ConvBlock(in_nc, this->hidden_nc * 2, 1, 1, 0));
    this->cv2 = register_module("cv2", ConvBlock(this->hidden_nc * (2 + n), out_nc, 1, 1, 0));

    for (size_t i = 0; i < n; i++){
        Bottleneck bottleneck = register_module("m_" + std::to_string(i), Bottleneck(this->hidden_nc, this->hidden_nc, shortcut, 1.0));
        this->m.push_back(bottleneck);
    }

}


// ---------------------------------------------------------------
// struct{C2fImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------------
torch::Tensor C2fImpl::forward(torch::Tensor x){

    torch::Tensor y, current, out;
    std::vector<torch::Tensor> concat_tensors;

    y = this->cv1->forward(x);
    concat_tensors = y.split(this->hidden_nc, 1);  // {N,2D} ===> ({N,D},{N,D})
    for (size_t i = 0; i < this->m.size(); i++){
        current = this->m[i]->forward(concat_tensors.back());
        concat_tensors.push_back(current);
    }
    out = torch::cat(concat_tensors, 1);
    out = this->cv2->forward(out);
    
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
// struct{YOLOv8Impl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
YOLOv8Impl::YOLOv8Impl(po::variables_map &vm){

    size_t nc = vm["nc"].as<size_t>();  // the number of image channels
    size_t class_num = vm["class_num"].as<size_t>();  // total classes
    std::string model = vm["model"].as<std::string>();  // total classes
    double depth, width;

    if (model == "yolov8n"){
        depth = 0.33;
        width = 0.25;
    }
    else if (model == "yolov8s"){
        depth = 0.33;
        width = 0.50;
    }
    else if (model == "yolov8m"){
        depth = 0.67;
        width = 0.75;
    }
    else if (model == "yolov8l"){
        depth = 1.0;
        width = 1.0;
    }
    else if (model == "yolov8x"){
        depth = 1.0;
        width = 1.25;
    }
    else{
        std::cerr << "Error : The name of model is " << model << '.' << std::endl;
        std::cerr << "Error : Please choose yolov8n, yolov8s, yolov8m, yolov8l or yolov8x." << std::endl;
        std::exit(1);
    }
    
    this->conv_0 = register_module("conv_0", ConvBlock(nc, mul(64, width), 3, 2, 1));
    this->conv_1 = register_module("conv_1", ConvBlock(mul(64, width), mul(128, width), 3, 2, 1));
    this->c2f_2 = register_module("c2f_2", C2f(mul(128, width), mul(128, width), mul(3, depth)));
    this->conv_3 = register_module("conv_3", ConvBlock(mul(128, width), mul(256, width), 3, 2, 1));
    this->c2f_4 = register_module("c2f_4", C2f(mul(256, width), mul(256, width), mul(6, depth)));
    this->conv_5 = register_module("conv_5", ConvBlock(mul(256, width), mul(512, width), 3, 2, 1));
    this->c2f_6 = register_module("c2f_6", C2f(mul(512, width), mul(512, width), mul(6, depth)));
    this->conv_7 = register_module("conv_7", ConvBlock(mul(512, width), mul(1024, width), 3, 2, 1));
    this->c2f_8 = register_module("c2f_8", C2f(mul(1024, width), mul(1024, width), mul(3, depth)));
    this->sppf_9 = register_module("sppf_9", SPPF(mul(1024, width), mul(1024, width), 5));

    this->head_c2f_12 = register_module("head_c2f_12", C2f(mul(1024, width) + mul(512, width), mul(512, width), mul(3, depth), false));

    this->head_c2f_15 = register_module("head_c2f_15", C2f(mul(512, width) + mul(256, width), mul(256, width), mul(3, depth), false));

    this->head_conv_16 = register_module("head_conv_16", ConvBlock(mul(256, width), mul(256, width), 3, 2, 1));
    this->head_c2f_18 = register_module("head_c2f_18", C2f(mul(256, width) + mul(512, width), mul(512, width), mul(3, depth), false));

    this->head_conv_19 = register_module("head_conv_19", ConvBlock(mul(512, width), mul(512, width), 3, 2, 1));
    this->head_c2f_21 = register_module("head_c2f_21", C2f(mul(512, width) + mul(1024, width), mul(1024, width), mul(3, depth), false));

    this->detect_small_coord = register_module("detect_small_coord", nn::Conv2d(nn::Conv2dOptions(mul(256, width), 4, 1).stride(1).padding(0).bias(true)));
    this->detect_small_obj = register_module("detect_small_obj", nn::Conv2d(nn::Conv2dOptions(mul(256, width), 1, 1).stride(1).padding(0).bias(true)));
    this->detect_small_class = register_module("detect_small_class", nn::Conv2d(nn::Conv2dOptions(mul(256, width), class_num, 1).stride(1).padding(0).bias(true)));

    this->detect_medium_coord = register_module("detect_medium_coord", nn::Conv2d(nn::Conv2dOptions(mul(512, width), 4, 1).stride(1).padding(0).bias(true)));
    this->detect_medium_obj = register_module("detect_medium_obj", nn::Conv2d(nn::Conv2dOptions(mul(512, width), 1, 1).stride(1).padding(0).bias(true)));
    this->detect_medium_class = register_module("detect_medium_class", nn::Conv2d(nn::Conv2dOptions(mul(512, width), class_num, 1).stride(1).padding(0).bias(true)));

    this->detect_large_coord = register_module("detect_large_coord", nn::Conv2d(nn::Conv2dOptions(mul(1024, width), 4, 1).stride(1).padding(0).bias(true)));
    this->detect_large_obj = register_module("detect_large_obj", nn::Conv2d(nn::Conv2dOptions(mul(1024, width), 1, 1).stride(1).padding(0).bias(true)));
    this->detect_large_class = register_module("detect_large_class", nn::Conv2d(nn::Conv2dOptions(mul(1024, width), class_num, 1).stride(1).padding(0).bias(true)));

}


// ---------------------------------------------------------
// struct{YOLOv8Impl}(nn::Module) -> function{mul}
// ---------------------------------------------------------
size_t YOLOv8Impl::mul(const double base, const double scale){
    return std::max<size_t>(1, (size_t)(std::round(base * scale)));
}

// ---------------------------------------------------------
// struct{YOLOv8Impl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
std::vector<torch::Tensor> YOLOv8Impl::forward(torch::Tensor x){

    torch::Tensor x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21;
    torch::Tensor small_coord, small_obj, small_class, small, medium_coord, medium_obj, medium_class, medium, large_coord, large_obj, large_class, large;
    std::vector<torch::Tensor> out;

    x0 = this->conv_0->forward(x);
    x1 = this->conv_1->forward(x0);
    x2 = this->c2f_2->forward(x1);
    x3 = this->conv_3->forward(x2);
    x4 = this->c2f_4->forward(x3);
    x5 = this->conv_5->forward(x4);
    x6 = this->c2f_6->forward(x5);
    x7 = this->conv_7->forward(x6);
    x8 = this->c2f_8->forward(x7);
    x9 = this->sppf_9->forward(x8);

    x10 = UpSampling(x9, {x6.size(2), x6.size(3)});
    x11 = torch::cat({x10, x6}, 1);
    x12 = this->head_c2f_12->forward(x11);

    x13 = UpSampling(x12, {x4.size(2), x4.size(3)});
    x14 = torch::cat({x13, x4}, 1);
    x15 = this->head_c2f_15->forward(x14);

    x16 = this->head_conv_16->forward(x15);
    x17 = torch::cat({x16, x12}, 1);
    x18 = this->head_c2f_18->forward(x17);

    x19 = this->head_conv_19->forward(x18);
    x20 = torch::cat({x19, x9}, 1);
    x21 = this->head_c2f_21->forward(x20);

    small_coord = this->detect_small_coord->forward(x15);  // {N,4,G,G}
    small_obj = this->detect_small_obj->forward(x15);  // {N,1,G,G}
    small_class = this->detect_small_class->forward(x15);  // {N,CN,G,G}
    small = torch::cat({small_coord, small_obj, small_class}, 1).permute({0, 2, 3, 1}).contiguous();  // {N,5+CN,G,G} ===> {N,G,G,5+CN}
    
    medium_coord = this->detect_medium_coord->forward(x18);  // {N,4,G,G}
    medium_obj = this->detect_medium_obj->forward(x18);  // {N,1,G,G}
    medium_class = this->detect_medium_class->forward(x18);  // {N,CN,G,G}
    medium = torch::cat({medium_coord, medium_obj, medium_class}, 1).permute({0, 2, 3, 1}).contiguous();  // {N,5+CN,G,G} ===> {N,G,G,5+CN}

    large_coord = this->detect_large_coord->forward(x21);  // {N,4,G,G}
    large_obj = this->detect_large_obj->forward(x21);  // {N,1,G,G}
    large_class = this->detect_large_class->forward(x21);  // {N,CN,G,G}
    large = torch::cat({large_coord, large_obj, large_class}, 1).permute({0, 2, 3, 1}).contiguous();  // {N,5+CN,G,G} ===> {N,G,G,5+CN}

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

