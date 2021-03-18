#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;


// ----------------------------------------------------------------------
// struct{MC_DiscriminatorImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MC_DiscriminatorImpl::MC_DiscriminatorImpl(po::variables_map &vm){

    bool BN = vm["BN"].as<bool>();
    size_t feature = vm["nf"].as<size_t>();
    size_t class_num = vm["class_num"].as<size_t>();
    size_t num_downs = (size_t)(std::log2(vm["size"].as<size_t>()));

    // 1th Blocks  {C,256,256} ===> {8F,2,2}
    Convolution(this->features, /*in_nc=*/vm["nc"].as<size_t>(), /*out_nc=*/feature, /*ksize=*/4, /*stride=*/2, /*pad=*/1, /*BN=*/false, /*LReLU=*/true);  // {C,256,256} ===> {F,128,128}
    Convolution(this->features, /*in_nc=*/feature, /*out_nc=*/feature*2, /*ksize=*/4, /*stride=*/2, /*pad=*/1, /*BN=*/BN, /*LReLU=*/true);                 // {F,128,128} ===> {2F,64,64}
    Convolution(this->features, /*in_nc=*/feature*2, /*out_nc=*/feature*4, /*ksize=*/4, /*stride=*/2, /*pad=*/1, /*BN=*/BN, /*LReLU=*/true);               // {2F,64,64}  ===> {4F,32,32}
    Convolution(this->features, /*in_nc=*/feature*4, /*out_nc=*/feature*8, /*ksize=*/4, /*stride=*/2, /*pad=*/1, /*BN=*/BN, /*LReLU=*/true);               // {4F,32,32}  ===> {8F,16,16}
    for (size_t i = 0; i < num_downs - 5; i++){
        Convolution(this->features, /*in_nc=*/feature*8, /*out_nc=*/feature*8, /*ksize=*/4, /*stride=*/2, /*pad=*/1, /*BN=*/BN, /*LReLU=*/true);           // {8F,16,16}  ===> {8F,2,2}
    }
    register_module("features", this->features);

    // 2th Blocks  {8F,X,X} ===> {8F,2,2}
    this->avgpool = nn::Sequential(nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({2, 2})));                                                           // {8F,X,X}    ===> {8F,2,2}
    register_module("avgpool", this->avgpool);

    // 3th Blocks  {8F,2,2} ===> {CN,1,1}
    Convolution(this->classifier, /*in_nc=*/feature*8, /*out_nc=*/class_num, /*ksize=*/2, /*stride=*/1, /*pad=*/0, /*BN=*/false, /*LReLU=*/false);         // {8F,2,2}    ===> {CN,1,1}
    register_module("classifier", this->classifier);

}


// -----------------------------------------------------------
// struct{MC_DiscriminatorImpl}(nn::Module) -> function{init}
// -----------------------------------------------------------
void MC_DiscriminatorImpl::init(){
    this->apply(weights_init);
    return;
}


// --------------------------------------------------------------
// struct{MC_DiscriminatorImpl}(nn::Module) -> function{forward}
// --------------------------------------------------------------
torch::Tensor MC_DiscriminatorImpl::forward(torch::Tensor x){
    torch::Tensor feature, out;
    feature = this->features->forward(x);       // {C,256,256} ===> {512,2,2}
    feature = this->avgpool->forward(feature);  // {512,X,X} ===> {512,2,2}
    out = this->classifier->forward(feature);   // {512,2,2} ===> {CN,1,1}
    out = out.view({out.size(0), -1});          // {CN,1,1} ===> {CN}
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
// function{Convolution}
// ----------------------------
void Convolution(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const size_t ksize, const size_t stride, const size_t pad, const bool BN, const bool LReLU, const bool bias){
    sq->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, ksize).stride(stride).padding(pad).bias(bias)));
    if (BN){
        sq->push_back(nn::BatchNorm2d(out_nc));
    }
    if (LReLU){
        sq->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
    }
    return;
}
