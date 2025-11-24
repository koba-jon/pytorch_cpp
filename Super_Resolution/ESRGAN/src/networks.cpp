#include <iostream>
#include <vector>
#include <typeinfo>
#include <cstdlib>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

#include <cmath>
#include <string>
#include <typeinfo>

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;

#define MAX_POOLING -1


// ----------------------------------------------------------------------
// struct{MC_VGGNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MC_VGGNetImpl::MC_VGGNetImpl(){

    std::vector<long int> cfg;
    cfg = {64, 64, MAX_POOLING, 128, 128, MAX_POOLING, 256, 256, 256, 256, MAX_POOLING, 512, 512, 512, 512, MAX_POOLING, 512, 512, 512, 512, MAX_POOLING};

    this->features = make_layers(3, cfg, true);  // {3,224,224} ===> {512,7,7}
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
        nn::Linear(/*in_channels=*/4096, /*out_channels=*/1000)  // {4096} ===> {CN}
    );
    register_module("classifier", this->classifier);

}


// ---------------------------------------------------------
// struct{MC_VGGNetImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor MC_VGGNetImpl::forward(torch::Tensor x){
    torch::Tensor feature, out;
    x = x.expand({x.size(0), 3, x.size(2), x.size(3)}) * 0.5 + 0.5;
    feature = this->features->forward(x);  // {3,224,224} ===> {512,7,7}
    return feature;
}


// -----------------------------------
// struct{DenseBlockImpl} -> constructor
// -----------------------------------
DenseBlockImpl::DenseBlockImpl(const size_t in_nc, const size_t growth_){

    this->growth = growth_;
    this->convs.reserve(5);
    size_t channels = in_nc;
    for (size_t i = 0; i < 4; i++){
        auto conv = nn::Conv2d(nn::Conv2dOptions(channels, growth_, /*kernel_size=*/3).stride(1).padding(1));
        this->convs.push_back(register_module("conv" + std::to_string(i), conv));
        channels += growth_;
    }
    auto conv_last = nn::Conv2d(nn::Conv2dOptions(channels, in_nc, /*kernel_size=*/3).stride(1).padding(1));
    this->convs.push_back(register_module("conv4", conv_last));
}


// -----------------------------------
// struct{DenseBlockImpl} -> function{forward}
// -----------------------------------
torch::Tensor DenseBlockImpl::forward(torch::Tensor x){
    std::vector<torch::Tensor> inputs;
    inputs.reserve(this->convs.size() + 1);
    inputs.push_back(x);

    for (size_t i = 0; i < this->convs.size(); i++){
        auto stacked = torch::cat(inputs, 1);
        auto out = this->convs[i]->forward(stacked);
        if (i + 1 < this->convs.size()){
            out = F::leaky_relu(out, F::LeakyReLUFuncOptions().negative_slope(0.2));
        }
        inputs.push_back(out);
    }

    return x + inputs.back() * 0.2;
}


// -----------------------------------
// struct{RRDBImpl} -> constructor
// -----------------------------------
RRDBImpl::RRDBImpl(const size_t in_nc, const size_t growth){
    this->blocks = nn::Sequential(
        DenseBlock(in_nc, growth),
        DenseBlock(in_nc, growth),
        DenseBlock(in_nc, growth)
    );
    register_module("blocks", this->blocks);
}


// -----------------------------------
// struct{RRDBImpl} -> function{forward}
// -----------------------------------
torch::Tensor RRDBImpl::forward(torch::Tensor x){
    auto out = this->blocks->forward(x);
    return x + out * 0.2;
}


// -----------------------------------
// struct{UpsampleBlockImpl} -> constructor
// -----------------------------------
UpsampleBlockImpl::UpsampleBlockImpl(const size_t in_nc){
    this->block = nn::Sequential(
        nn::Upsample(nn::UpsampleOptions().scale_factor(std::vector<double>({2.0, 2.0})).mode(torch::kNearest)),
        nn::Conv2d(nn::Conv2dOptions(in_nc, in_nc, /*kernel_size=*/3).stride(1).padding(1)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2))
    );
    register_module("block", this->block);
}


// -----------------------------------
// struct{UpsampleBlockImpl} -> function{forward}
// -----------------------------------
torch::Tensor UpsampleBlockImpl::forward(torch::Tensor x){
    return this->block->forward(x);
}


// -----------------------------------
// struct{ESRGAN_GeneratorImpl} -> constructor
// -----------------------------------
ESRGAN_GeneratorImpl::ESRGAN_GeneratorImpl(po::variables_map &vm){

    const size_t nc = vm["nc"].as<size_t>();
    const size_t ngf = vm["ngf"].as<size_t>();
    const size_t n_rrdb = vm["n_rrdb"].as<size_t>();
    const size_t growth = vm["growth_channels"].as<size_t>();
    const size_t upscale = vm["upscale"].as<size_t>();

    // (1) Head
    this->head = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(nc, ngf, /*kernel_size=*/3).stride(1).padding(1)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2))
    );
    register_module("head", this->head);

    // (2) Body
    this->body = nn::Sequential();
    for (size_t i = 0; i < n_rrdb; i++){
        this->body->push_back(RRDB(ngf, growth));
    }
    this->body->push_back(nn::Conv2d(nn::Conv2dOptions(ngf, ngf, /*kernel_size=*/3).stride(1).padding(1)));
    register_module("body", this->body);

    // (3) Tail
    this->tail = nn::Sequential();
    size_t n_upsample = size_t(std::round(std::log2((double)upscale)));
    for (size_t i = 0; i < n_upsample; i++){
        this->tail->push_back(UpsampleBlock(ngf));
    }
    this->tail->push_back(nn::Conv2d(nn::Conv2dOptions(ngf, ngf, /*kernel_size=*/3).stride(1).padding(1)));
    this->tail->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)));
    this->tail->push_back(nn::Conv2d(nn::Conv2dOptions(ngf, nc, /*kernel_size=*/3).stride(1).padding(1)));
    this->tail->push_back(nn::Tanh());
    register_module("tail", this->tail);

}


// -----------------------------------
// struct{ESRGAN_GeneratorImpl} -> function{forward}
// -----------------------------------
torch::Tensor ESRGAN_GeneratorImpl::forward(torch::Tensor x){
    auto head = this->head->forward(x);
    auto trunk = this->body->forward(head);
    auto out = head + trunk;
    out = this->tail->forward(out);
    return out;
}


// ---------------------------------------------------------
// struct{ConvBlock} -> constructor
// ---------------------------------------------------------
ConvBlockImpl::ConvBlockImpl(const size_t in_nc, const size_t out_nc, const int stride, const bool BN){
    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, /*kernel_size=*/3).stride(stride).padding(1)));
    if (BN){
        this->model->push_back(nn::BatchNorm2d(out_nc));
    }
    this->model->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)));
    register_module("model", this->model);
}


// ---------------------------------------------------------
// struct{ConvBlock} -> function{forward}
// ---------------------------------------------------------
torch::Tensor ConvBlockImpl::forward(torch::Tensor x){
    return this->model->forward(x);
}


// ------------------------------------------------
// struct{ESRGAN_DiscriminatorImpl} -> constructor
// ------------------------------------------------
ESRGAN_DiscriminatorImpl::ESRGAN_DiscriminatorImpl(po::variables_map &vm){

    const size_t input_nc = vm["nc"].as<size_t>();
    const size_t ndf = vm["ndf"].as<size_t>();

    this->model = nn::Sequential(
        ConvBlock(input_nc, ndf, /*stride=*/1, /*batch_norm=*/false),
        ConvBlock(ndf, ndf, /*stride=*/2),
        ConvBlock(ndf, ndf * 2, /*stride=*/1),
        ConvBlock(ndf * 2, ndf * 2, /*stride=*/2),
        ConvBlock(ndf * 2, ndf * 4, /*stride=*/1),
        ConvBlock(ndf * 4, ndf * 4, /*stride=*/2),
        ConvBlock(ndf * 4, ndf * 8, /*stride=*/1),
        ConvBlock(ndf * 8, ndf * 8, /*stride=*/2),
        nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({6, 6})),
        nn::Flatten(),
        nn::Linear(/*in_features=*/ndf * 8 * 6 * 6, /*out_features=*/1024),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        nn::Linear(/*in_features=*/1024, /*out_features=*/1)
    );

    register_module("model", this->model);
}


// -----------------------------------
// struct{ESRGAN_DiscriminatorImpl} -> function{forward}
// -----------------------------------
torch::Tensor ESRGAN_DiscriminatorImpl::forward(torch::Tensor x){
    return this->model->forward(x);
}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl))) {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
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
