#include <algorithm>
#include <utility>
#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
using namespace torch;


// ----------------------------------------------------------------------
// struct{UNet_GeneratorImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
UNet_GeneratorImpl::UNet_GeneratorImpl(po::variables_map &vm){
    
    size_t feature = vm["ngf"].as<size_t>();
    size_t num_downs = (size_t)(std::log2(vm["size"].as<size_t>()));
    bool use_dropout = !vm["no_dropout"].as<bool>();

    UNetBlockImpl blocks, fake;
    blocks = UNetBlockImpl({feature*8, feature*8}, vm["nz"].as<size_t>(), /*submodule_=*/fake, /*outermost_=*/false, /*innermost=*/true);
    for (size_t i = 0; i < num_downs - 5; i++){
        blocks = UNetBlockImpl({feature*8, feature*8}, feature*8, /*submodule_=*/blocks, /*outermost_=*/false, /*innermost=*/false, /*use_dropout=*/use_dropout);
    }
    blocks = UNetBlockImpl({feature*4, feature*4}, feature*8, /*submodule_=*/blocks);
    blocks = UNetBlockImpl({feature*2, feature*2}, feature*4, /*submodule_=*/blocks);
    blocks = UNetBlockImpl({feature, feature}, feature*2, /*submodule_=*/blocks);
    blocks = UNetBlockImpl({vm["input_nc"].as<size_t>(), vm["output_nc"].as<size_t>()}, feature, /*submodule_=*/blocks, /*outermost_=*/true);
    
    this->model->push_back(blocks);
    register_module("U-Net", this->model);

}


// ----------------------------------------------------------------------
// struct{UNet_GeneratorImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor UNet_GeneratorImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);  // {IC,256,256} ===> {OC,256,256}
    return out;
}


// ----------------------------------------------------------------------
// struct{UNetBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
UNetBlockImpl::UNetBlockImpl(const std::pair<size_t, size_t> outside_nc, const size_t inside_nc, UNetBlockImpl &submodule, bool outermost_, bool innermost, bool use_dropout){

    this->outermost = outermost_;

    if (this->outermost){  // {IC,256,256} ===> {F,128,128} ===> ... ===> {2F,128,128} ===> {OC,256,256}
        DownSampling(this->model, outside_nc.first, inside_nc, /*BN=*/false, /*LReLU=*/false);                // {IC,256,256} ===> {F,128,128}
        this->model->push_back(submodule);
        UpSampling(this->model, inside_nc*2, outside_nc.second, /*BN=*/false, /*ReLU=*/true, /*bias=*/true);  // {2F,128,128} ===> {OC,256,256}
        this->model->push_back(nn::Tanh());
    }
    else if (innermost){   // {8F,2,2} ===> {Z,1,1} ===> {8F,2,2}
        DownSampling(this->model, outside_nc.first, inside_nc, /*BN=*/false, /*LReLU=*/false);  // {8F,2,2} ===> {Z,1,1}
        UpSampling(this->model, inside_nc, outside_nc.second, /*BN=*/true, /*ReLU=*/true);      // {Z,1,1} ===> {8F,2,2}
    }
    else{                  // {NF,H,W} ===> {NF,H/2,W/2} ===> ... ===> {2NF,H/2,W/2} ===> {NF,H,W}
        DownSampling(this->model, outside_nc.first, inside_nc, /*BN=*/true, /*LReLU=*/true);  // {NF,H,W} ===> {NF,H/2,W/2}
        this->model->push_back(submodule);
        UpSampling(this->model, inside_nc*2, outside_nc.second, /*BN=*/true, /*ReLU=*/true);  // {2NF,H/2,W/2} ===> {NF,H,W}
        if (use_dropout){
            this->model->push_back(nn::Dropout(0.5));
        }
    }
    register_module("U-Net_Block", this->model);

}


// ----------------------------------------------------------------------
// struct{UNetBlockImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor UNetBlockImpl::forward(torch::Tensor x){
    torch::Tensor out;
    if (this->outermost){
        out = this->model->forward(x);
    }
    else{
        out = this->model->forward(x);
        out = torch::cat({x, out}, /*dim=*/1);
    }
    return out;
}


// ----------------------------------------------------------------------
// struct{PatchGAN_DiscriminatorImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
// A : making the final output a feature map with a certain size and making a true or false decision at each pixel
// B : making the input image a patch and making a true or false decision at the output of each patch
// This structure is A type. (A is equivalent to B)
// ----------------------------------------------------------------------
PatchGAN_DiscriminatorImpl::PatchGAN_DiscriminatorImpl(po::variables_map &vm){
    
    size_t feature = vm["ndf"].as<size_t>();

    size_t mul = 1;
    size_t mul_pre = 1;
    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(vm["input_nc"].as<size_t>()+vm["output_nc"].as<size_t>(), feature, 4).stride(2).padding(1).bias(true)));  // {IC+OC,256,256} ===> {F,128,128}
    this->model->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
    for (size_t n = 1; n < vm["n_layers"].as<size_t>(); n++){  // {F,128,128} ===> {4F,32,32}
        mul_pre = mul;
        mul = std::min((size_t)std::pow(2, n), (size_t)8);
        this->model->push_back(nn::Conv2d(nn::Conv2dOptions(feature*mul_pre, feature*mul, 4).stride(2).padding(1).bias(false)));
        this->model->push_back(nn::BatchNorm2d(feature*mul));
        this->model->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
    }

    mul_pre = mul;
    mul = std::min((size_t)std::pow(2, vm["n_layers"].as<size_t>()), (size_t)8);
    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(feature*mul_pre, feature*mul, 4).stride(1).padding(1).bias(false)));  // {4F,32,32} ===> {8F,31,31}
    this->model->push_back(nn::BatchNorm2d(feature*mul));
    this->model->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(feature*mul, 1, 4).stride(1).padding(1).bias(true)));  // {8F,31,31} ===> {1,30,30}

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
    else if ((typeid(m) == typeid(nn::BatchNorm2d)) || (typeid(m) == typeid(nn::BatchNorm2dImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/1.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}


// ----------------------------
// function{DownSampling}
// ----------------------------
void DownSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool LReLU, const bool bias){
    if (LReLU){
        sq->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
    }
    sq->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 4).stride(2).padding(1).bias(bias)));
    if (BN){
        sq->push_back(nn::BatchNorm2d(out_nc));
    }
    return;
}


// ----------------------------
// function{UpSampling}
// ----------------------------
void UpSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias){
    if (ReLU){
        sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    sq->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(in_nc, out_nc, 4).stride(2).padding(1).bias(bias)));
    if (BN){
        sq->push_back(nn::BatchNorm2d(out_nc));
    }
    return;
}
