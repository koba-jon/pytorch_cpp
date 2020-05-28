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
// struct{UNetBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
UNetBlockImpl::UNetBlockImpl(const std::pair<size_t, size_t> outside_nc, const size_t inside_nc, UNetBlockImpl &submodule, bool outermost_, bool innermost, bool use_dropout){

    this->outermost = outermost_;

    if (this->outermost){  // {IC,256,256} ===> {F,128,128} ===> ... ===> {2F,128,128} ===> {OC,256,256}
        DownSampling(this->model, outside_nc.first, inside_nc, /*BN=*/false, /*ReLU=*/false);  // {IC,256,256} ===> {F,128,128}
        this->model->push_back(submodule);
        UpSampling(this->model, inside_nc*2, outside_nc.second, /*BN=*/false, /*ReLU=*/true);  // {2F,128,128} ===> {OC,256,256}
        this->model->push_back(nn::Tanh());
    }
    else if (innermost){   // {8F,2,2} ===> {Z,1,1} ===> {8F,2,2}
        DownSampling(this->model, outside_nc.first, inside_nc, /*BN=*/false, /*ReLU=*/false);  // {8F,2,2} ===> {Z,1,1}
        UpSampling(this->model, inside_nc, outside_nc.second, /*BN=*/true, /*ReLU=*/true);     // {Z,1,1} ===> {8F,2,2}
    }
    else{                  // {NF,H,W} ===> {NF,H/2,W/2} ===> ... ===> {2NF,H/2,W/2} ===> {NF,H,W}
        DownSampling(this->model, outside_nc.first, inside_nc, /*BN=*/true, /*ReLU=*/true);    // {NF,H,W} ===> {NF,H/2,W/2}
        this->model->push_back(submodule);
        UpSampling(this->model, inside_nc*2, outside_nc.second, /*BN=*/true, /*ReLU=*/true);   // {2NF,H/2,W/2} ===> {NF,H,W}
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
// struct{UNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
UNetImpl::UNetImpl(po::variables_map &vm){
    
    size_t feature = vm["nf"].as<size_t>();
    size_t num_downs = (size_t)(std::log2(vm["size"].as<size_t>()));
    bool use_dropout = !vm["no_dropout"].as<bool>();

    this->blocks = UNetBlockImpl({feature*8, feature*8}, vm["nz"].as<size_t>(), /*submodule_=*/this->fake, /*outermost_=*/false, /*innermost=*/true);
    for (size_t i = 0; i < num_downs - 5; i++){
        this->blocks = UNetBlockImpl({feature*8, feature*8}, feature*8, /*submodule_=*/this->blocks, /*outermost_=*/false, /*innermost=*/false, /*use_dropout=*/use_dropout);
    }
    this->blocks = UNetBlockImpl({feature*4, feature*4}, feature*8, /*submodule_=*/this->blocks);
    this->blocks = UNetBlockImpl({feature*2, feature*2}, feature*4, /*submodule_=*/this->blocks);
    this->blocks = UNetBlockImpl({feature, feature}, feature*2, /*submodule_=*/this->blocks);
    this->blocks = UNetBlockImpl({vm["nc"].as<size_t>(), vm["class_num"].as<size_t>()}, feature, /*submodule_=*/this->blocks, /*outermost_=*/true);
    
    this->model->push_back(this->blocks);
    register_module("U-Net", this->model);

}


// ----------------------------------------------------------------------
// struct{UNetImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor UNetImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);  // {IC,256,256} ===> {OC,256,256}
    out = torch::log_softmax(out, /*dim=*/1);
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
void DownSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias){
    if (ReLU){
        sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
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