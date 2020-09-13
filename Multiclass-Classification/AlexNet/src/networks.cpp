#include <typeinfo>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;


// ----------------------------------------------------------------------
// struct{MC_AlexNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MC_AlexNetImpl::MC_AlexNetImpl(po::variables_map &vm){

    this->features = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/vm["nc"].as<size_t>(), /*out_channels=*/96, /*kernel_size=*/11).stride(4)),  // {C,227,227} ===> {96,55,55}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::LocalResponseNorm(nn::LocalResponseNormOptions(/*size=*/5).alpha(0.0001).beta(0.75).k(2.0)),
        nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2)),                                                         // {96,55,55} ===> {96,27,27}
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/96, /*out_channels=*/256, /*kernel_size=*/5).stride(1).padding(2)),          // {96,27,27} ===> {256,27,27}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::LocalResponseNorm(nn::LocalResponseNormOptions(/*size=*/5).alpha(0.0001).beta(0.75).k(2.0)),
        nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2)),                                                         // {256,27,27} ===> {256,13,13}
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/256, /*out_channels=*/384, /*kernel_size=*/3).stride(1).padding(1)),         // {256,13,13} ===> {384,13,13}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/384, /*out_channels=*/384, /*kernel_size=*/3).stride(1).padding(1)),         // {384,13,13} ===> {384,13,13}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/384, /*out_channels=*/256, /*kernel_size=*/3).stride(1).padding(1)),         // {384,13,13} ===> {256,13,13}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2))                                                          // {256,13,13} ===> {256,6,6}
    );
    register_module("features", this->features);

    this->avgpool = nn::Sequential(nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({6, 6})));  // {256,X,X} ===> {256,6,6}
    register_module("avgpool", this->avgpool);

    this->classifier = nn::Sequential(
        nn::Dropout(0.5),
        nn::Linear(/*in_channels=*/256*6*6, /*out_channels=*/4096),                      // {256*6*6} ===> {4096}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Dropout(0.5),
        nn::Linear(/*in_channels=*/4096, /*out_channels=*/4096),                         // {4096} ===> {4096}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Linear(/*in_channels=*/4096, /*out_channels=*/vm["class_num"].as<size_t>())  // {4096} ===> {CN}
    );
    register_module("classifier", this->classifier);

}


// ---------------------------------------------------------
// struct{MC_AlexNetImpl}(nn::Module) -> function{init}
// ---------------------------------------------------------
void MC_AlexNetImpl::init(){
    this->apply(weights_init);
    nn::init::constant_(*(this->features[4]->named_parameters(false).find("bias")), /*bias=*/1.0);
    nn::init::constant_(*(this->features[10]->named_parameters(false).find("bias")), /*bias=*/1.0);
    nn::init::constant_(*(this->features[12]->named_parameters(false).find("bias")), /*bias=*/1.0);
    return;
}


// ---------------------------------------------------------
// struct{MC_AlexNetImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor MC_AlexNetImpl::forward(torch::Tensor x){
    torch::Tensor feature, out;
    feature = this->features->forward(x);           // {C,227,227} ===> {256,6,6}
    feature = this->avgpool->forward(feature);      // {256,X,X} ===> {256,6,6}
    feature = feature.view({feature.size(0), -1});  // {256,6,6} ===> {256*6*6}
    out = this->classifier->forward(feature);       // {256*6*6} ===> {CN}
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
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}
