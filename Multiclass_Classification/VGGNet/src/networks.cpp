#include <iostream>
#include <vector>
#include <typeinfo>
#include <cstdlib>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;

#define MAX_POOLING -1


// ----------------------------------------------------------------------
// struct{MC_VGGNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MC_VGGNetImpl::MC_VGGNetImpl(po::variables_map &vm){

    std::vector<long int> cfg;
    size_t n_layers = vm["n_layers"].as<size_t>();
    if (n_layers == 11){
        cfg = {64, MAX_POOLING, 128, MAX_POOLING, 256, 256, MAX_POOLING, 512, 512, MAX_POOLING, 512, 512, MAX_POOLING};
    }
    else if (n_layers == 13){
        cfg = {64, 64, MAX_POOLING, 128, 128, MAX_POOLING, 256, 256, MAX_POOLING, 512, 512, MAX_POOLING, 512, 512, MAX_POOLING};
    }
    else if (n_layers == 16){
        cfg = {64, 64, MAX_POOLING, 128, 128, MAX_POOLING, 256, 256, 256, MAX_POOLING, 512, 512, 512, MAX_POOLING, 512, 512, 512, MAX_POOLING};
    }
    else if (n_layers == 19){
        cfg = {64, 64, MAX_POOLING, 128, 128, MAX_POOLING, 256, 256, 256, 256, MAX_POOLING, 512, 512, 512, 512, MAX_POOLING, 512, 512, 512, 512, MAX_POOLING};
    }
    else{
        std::cerr << "Error : The number of layers is " << n_layers << '.' << std::endl;
        std::cerr << "Error : Please choose 11, 13, 16 or 19." << std::endl;
        std::exit(1);
    }

    this->features = make_layers(vm["nc"].as<size_t>(), cfg, vm["BN"].as<bool>());  // {C,224,224} ===> {512,7,7}
    register_module("features", this->features);

    this->avgpool = nn::Sequential(nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({7, 7})));  // {512,X,X} ===> {512,7,7}
    register_module("avgpool", this->avgpool);

    this->classifier = nn::Sequential(
        nn::Linear(/*in_channels=*/512*7*7, /*out_channels=*/4096),                      // {512*7*7} ===> {4096}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Dropout(0.5),
        nn::Linear(/*in_channels=*/4096, /*out_channels=*/4096),                         // {4096} ===> {4096}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Dropout(0.5),
        nn::Linear(/*in_channels=*/4096, /*out_channels=*/vm["class_num"].as<size_t>())  // {4096} ===> {CN}
    );
    register_module("classifier", this->classifier);

}


// ---------------------------------------------------------
// struct{MC_VGGNetImpl}(nn::Module) -> function{init}
// ---------------------------------------------------------
void MC_VGGNetImpl::init(){
    this->apply(weights_init);
    return;
}


// ---------------------------------------------------------
// struct{MC_VGGNetImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor MC_VGGNetImpl::forward(torch::Tensor x){
    torch::Tensor feature, out;
    feature = this->features->forward(x);           // {C,224,224} ===> {512,7,7}
    feature = this->avgpool->forward(feature);      // {512,X,X} ===> {512,7,7}
    feature = feature.view({feature.size(0), -1});  // {512,7,7} ===> {512*7*7}
    out = this->classifier->forward(feature);       // {512*7*7} ===> {CN}
    out = F::log_softmax(out, /*dim=*/1);
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
        if (w != nullptr) nn::init::kaiming_normal_(*w, /*a=*/0, /*mode=*/torch::kFanOut, /*nonlinearity=*/torch::kReLU);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
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
// function{make_layers}
// ----------------------------
nn::Sequential make_layers(const size_t nc, const std::vector<long int> cfg, const bool BN){
    nn::Sequential sq;
    long int in_channels = (long int)nc;
    for (auto v : cfg){
        if (v == MAX_POOLING){
            sq->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));
        }
        else{
            sq->push_back(nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/in_channels, /*out_channels=*/v, /*kernel_size=*/3).stride(1).padding(1)));
            if (BN){
                sq->push_back(nn::BatchNorm2d(v));
            }
            sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
            in_channels = v;
        }
    }
    return sq;
}
