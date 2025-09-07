#include <algorithm>
#include <utility>
#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;


// ----------------------------------------------------------------------
// struct{ResNetBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ResNetBlockImpl::ResNetBlockImpl(const size_t dim){
    this->model = nn::Sequential(
        nn::ReflectionPad2d(nn::ReflectionPad2dOptions({1, 1, 1, 1})),
        nn::Conv2d(nn::Conv2dOptions(dim, dim, 3).stride(1).padding(0).bias(true)),
        nn::InstanceNorm2d(nn::InstanceNorm2dOptions(dim).affine(false).track_running_stats(false)),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::ReflectionPad2d(nn::ReflectionPad2dOptions({1, 1, 1, 1})),
        nn::Conv2d(nn::Conv2dOptions(dim, dim, 3).stride(1).padding(0).bias(true)),
        nn::InstanceNorm2d(nn::InstanceNorm2dOptions(dim).affine(false).track_running_stats(false))
    );
    register_module("ResNet_Block", this->model);
}


// ----------------------------------------------------------------------
// struct{ResNetBlockImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor ResNetBlockImpl::forward(torch::Tensor x){
    torch::Tensor out = x + this->model->forward(x);
    return out;
}


// ----------------------------------------------------------------------
// struct{ResNet_GeneratorImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ResNet_GeneratorImpl::ResNet_GeneratorImpl(size_t input_nc, size_t output_nc, size_t ngf, size_t n_blocks){

    constexpr size_t n_downsampling = 2;
    size_t mul;

    this->model = nn::Sequential(
        nn::ReflectionPad2d(nn::ReflectionPad2dOptions({3, 3, 3, 3})),
        nn::Conv2d(nn::Conv2dOptions(input_nc, ngf, 7).stride(1).padding(0).bias(true)),
        nn::InstanceNorm2d(nn::InstanceNorm2dOptions(ngf).affine(false).track_running_stats(false)),
        nn::ReLU(nn::ReLUOptions().inplace(true))
    );

    for (size_t i = 0; i < n_downsampling; i++){
        mul = std::pow(n_downsampling, i);
        this->model->push_back(nn::Conv2d(nn::Conv2dOptions(ngf * mul, ngf * mul * 2, 3).stride(2).padding(1).bias(true)));
        this->model->push_back(nn::InstanceNorm2d(nn::InstanceNorm2dOptions(ngf * mul * 2).affine(false).track_running_stats(false)));
        this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }

    mul = std::pow(2, n_downsampling);
    for (size_t i = 0; i < n_blocks; i++){
        this->model->push_back(ResNetBlock(ngf * mul));
    }

    for (size_t i = 0; i < n_downsampling; i++){
        mul = std::pow(2, n_downsampling - i);
        this->model->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(ngf * mul, ngf * mul / 2, 3).stride(2).padding(1).output_padding(1).bias(true)));
        this->model->push_back(nn::InstanceNorm2d(nn::InstanceNorm2dOptions(ngf * mul / 2).affine(false).track_running_stats(false)));
        this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    this->model->push_back(nn::ReflectionPad2d(nn::ReflectionPad2dOptions({3, 3, 3, 3})));
    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(ngf, output_nc, 7).stride(1).padding(0).bias(true)));
    this->model->push_back(nn::Tanh());
    
    register_module("ResNet", this->model);

}


// ----------------------------------------------------------------------
// struct{ResNet_GeneratorImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor ResNet_GeneratorImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);  // {IC,256,256} ===> {OC,256,256}
    return out;
}



// ----------------------------------------------------------------------
// struct{PatchGAN_DiscriminatorImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
// A : making the final output a feature map with a certain size and making a true or false decision at each pixel
// B : making the input image a patch and making a true or false decision at the output of each patch
// This structure is A type. (A is equivalent to B)
// ----------------------------------------------------------------------
PatchGAN_DiscriminatorImpl::PatchGAN_DiscriminatorImpl(size_t input_nc, size_t ndf, size_t n_layers){
    
    size_t mul = 1;
    size_t mul_pre = 1;

    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(input_nc, ndf, 4).stride(2).padding(1).bias(true)));  // {IC+OC,256,256} ===> {F,128,128}
    this->model->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));

    for (size_t n = 1; n < n_layers; n++){  // {F,128,128} ===> {4F,32,32}
        mul_pre = mul;
        mul = std::min((size_t)std::pow(2, n), (size_t)8);
        this->model->push_back(nn::Conv2d(nn::Conv2dOptions(ndf * mul_pre, ndf * mul, 4).stride(2).padding(1).bias(false)));
        this->model->push_back(nn::InstanceNorm2d(ndf * mul));
        this->model->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
    }

    mul_pre = mul;
    mul = std::min((size_t)std::pow(2, n_layers), (size_t)8);
    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(ndf * mul_pre, ndf * mul, 4).stride(1).padding(1).bias(false)));  // {4F,32,32} ===> {8F,31,31}
    this->model->push_back(nn::InstanceNorm2d(ndf * mul));
    this->model->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(ndf * mul, 1, 4).stride(1).padding(1).bias(true)));  // {8F,31,31} ===> {1,30,30}

    register_module("PatchGAN", this->model);

}


// ----------------------------------------------------------------------
// struct{PatchGAN_DiscriminatorImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor PatchGAN_DiscriminatorImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);  // {IC+OC,256,256} ===> {1,30,30}
    return out;
}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl)) || (typeid(m) == typeid(nn::ConvTranspose2d)) || (typeid(m) == typeid(nn::ConvTranspose2dImpl))) {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::InstanceNorm2d)) || (typeid(m) == typeid(nn::InstanceNorm2dImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/1.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}
