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


// ----------------------------------------------------------------------
// struct{MC_ResNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MC_ResNetImpl::MC_ResNetImpl(po::variables_map &vm){

    bool basic_block;
    std::vector<long int> cfg;
    size_t n_layers = vm["n_layers"].as<size_t>();
    if (n_layers == 18){
        basic_block = true;
        cfg = {2, 2, 2, 2};
    }
    else if (n_layers == 34){
        basic_block = true;
        cfg = {3, 4, 6, 3};
    }
    else if (n_layers == 50){
        basic_block = false;
        cfg = {3, 4, 6, 3};
    }
    else if (n_layers == 101){
        basic_block = false;
        cfg = {3, 4, 23, 3};
    }
    else if (n_layers == 152){
        basic_block = false;
        cfg = {3, 8, 36, 3};
    }
    else{
        std::cerr << "Error : The number of layers is " << n_layers << '.' << std::endl;
        std::cerr << "Error : Please choose 18, 34, 50, 101 or 152." << std::endl;
        std::exit(1);
    }
    size_t feature = vm["nf"].as<size_t>();
    this->inplanes = vm["nf"].as<size_t>();

    // First Downsampling
    this->first = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/vm["nc"].as<size_t>(), /*out_channels=*/this->inplanes, /*kernel_size=*/7).stride(2).padding(3).bias(false)),  // {C,224,224} ===> {F,112,112}
        nn::BatchNorm2d(this->inplanes),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1))                                                                                 // {F,112,112} ===> {F,56,56}
    );
    register_module("first", this->first);

    // After the Second Time
    size_t expansion;
    if (basic_block){
        BasicBlockImpl block;
        expansion = block.expansion;
        this->layer1 = this->make_layers(block, feature, /*num_blocks=*/cfg.at(0), /*stride=*/1);    // {F,56,56} ===> {F*E,56,56}
        this->layer2 = this->make_layers(block, feature*2, /*num_blocks=*/cfg.at(1), /*stride=*/2);  // {F*E,56,56} ===> {2F*E,28,28}
        this->layer3 = this->make_layers(block, feature*4, /*num_blocks=*/cfg.at(2), /*stride=*/2);  // {2F*E,28,28} ===> {4F*E,14,14}
        this->layer4 = this->make_layers(block, feature*8, /*num_blocks=*/cfg.at(3), /*stride=*/2);  // {4F*E,14,14} ===> {8F*E,7,7}
    }
    else{
        BottleneckImpl block;
        expansion = block.expansion;
        this->layer1 = this->make_layers(block, feature, /*num_blocks=*/cfg.at(0), /*stride=*/1);    // {F,56,56} ===> {F*E,56,56}
        this->layer2 = this->make_layers(block, feature*2, /*num_blocks=*/cfg.at(1), /*stride=*/2);  // {F*E,56,56} ===> {2F*E,28,28}
        this->layer3 = this->make_layers(block, feature*4, /*num_blocks=*/cfg.at(2), /*stride=*/2);  // {2F*E,28,28} ===> {4F*E,14,14}
        this->layer4 = this->make_layers(block, feature*8, /*num_blocks=*/cfg.at(3), /*stride=*/2);  // {4F*E,14,14} ===> {8F*E,7,7}
    }
    register_module("layer1", this->layer1);
    register_module("layer2", this->layer2);
    register_module("layer3", this->layer3);
    register_module("layer4", this->layer4);

    // Final Downsampling
    this->avgpool = nn::Sequential(nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({1, 1})));  // {8F*E,7,7} ===> {8F*E,1,1}
    register_module("avgpool", this->avgpool);

    // Classification
    this->classifier = nn::Sequential(nn::Linear(/*in_channels=*/feature*8*expansion, /*out_channels=*/vm["class_num"].as<size_t>()));  // {8F*E} ===> {CN}
    register_module("classifier", this->classifier);

}


// ---------------------------------------------------------
// struct{MC_ResNetImpl}(nn::Module) -> function{init}
// ---------------------------------------------------------
void MC_ResNetImpl::init(){
    this->apply(weights_init);
    return;
}


// -----------------------------------------------------------
// struct{MC_ResNetImpl}(nn::Module) -> function{make_layers}
// -----------------------------------------------------------
template <typename T>
nn::Sequential MC_ResNetImpl::make_layers(T &block, const size_t planes, const size_t num_blocks, const size_t stride){
    nn::Sequential layers = nn::Sequential(T(this->inplanes, planes, stride));
    this->inplanes = planes * block.expansion;
    for (size_t i = 1; i < num_blocks; i++){
        layers->push_back(T(this->inplanes, planes, /*stride=*/1));
    }
    return layers;
}


// ---------------------------------------------------------
// struct{MC_ResNetImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor MC_ResNetImpl::forward(torch::Tensor x){
    torch::Tensor feature, out;
    feature = this->first->forward(x);              // {C,224,224} ===> {F,56,56}
    feature = this->layer1->forward(feature);       // {F,56,56} ===> {F*E,56,56}
    feature = this->layer2->forward(feature);       // {F*E,56,56} ===> {2F*E,28,28}
    feature = this->layer3->forward(feature);       // {2F*E,28,28} ===> {4F*E,14,14}
    feature = this->layer4->forward(feature);       // {4F*E,14,14} ===> {8F*E,7,7}
    feature = this->avgpool->forward(feature);      // {8F*E,7,7} ===> {8F*E,1,1}
    feature = feature.view({feature.size(0), -1});  // {8F*E,1,1} ===> {8F*E}
    out = this->classifier->forward(feature);       // {8F*E} ===> {CN}
    out = F::log_softmax(out, /*dim=*/1);
    return out;
}


// ----------------------------------------------------------------------
// struct{BasicBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
BasicBlockImpl::BasicBlockImpl(const size_t inplanes, const size_t planes, const size_t stride){

    this->layerA = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/planes, /*kernel_size=*/3).stride(stride).padding(1).bias(false)),
        nn::BatchNorm2d(planes),
        nn::ReLU(nn::ReLUOptions().inplace(true))
    );
    register_module("layerA", this->layerA);

    this->layerB = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/planes, /*out_channels=*/this->expansion*planes, /*kernel_size=*/3).stride(1).padding(1).bias(false)),
        nn::BatchNorm2d(this->expansion*planes)
    );
    register_module("layerB", this->layerB);

    this->down = false;
    if ((stride != 1) || (inplanes != (this->expansion*planes))){
        this->down = true;
        this->downsample = nn::Sequential(
            nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/this->expansion*planes, /*kernel_size=*/1).stride(stride).padding(0).bias(false)),
            nn::BatchNorm2d(this->expansion*planes)
        );
        register_module("downsample", this->downsample);
    }

    this->last = nn::Sequential(
        nn::ReLU(nn::ReLUOptions().inplace(true))
    );
    register_module("last", this->last);

}


// ---------------------------------------------------------
// struct{BasicBlockImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor BasicBlockImpl::forward(torch::Tensor x){
    torch::Tensor out;
    out = this->layerA->forward(x);
    out = this->layerB->forward(out);
    out += (this->down ? this->downsample->forward(x) : x);
    out = this->last->forward(out);
    return out;
}


// ----------------------------------------------------------------------
// struct{BottleneckImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
BottleneckImpl::BottleneckImpl(const size_t inplanes, const size_t planes, const size_t stride){

    this->layerA = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/planes, /*kernel_size=*/1).stride(1).padding(0).bias(false)),
        nn::BatchNorm2d(planes),
        nn::ReLU(nn::ReLUOptions().inplace(true))
    );
    register_module("layerA", this->layerA);

    this->layerB = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/planes, /*out_channels=*/planes, /*kernel_size=*/3).stride(stride).padding(1).bias(false)),
        nn::BatchNorm2d(planes),
        nn::ReLU(nn::ReLUOptions().inplace(true))
    );
    register_module("layerB", this->layerB);

    this->layerC = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/planes, /*out_channels=*/this->expansion*planes, /*kernel_size=*/1).stride(1).padding(0).bias(false)),
        nn::BatchNorm2d(this->expansion*planes)
    );
    register_module("layerC", this->layerC);

    this->down = false;
    if ((stride != 1) || (inplanes != (this->expansion*planes))){
        this->down = true;
        this->downsample = nn::Sequential(
            nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/this->expansion*planes, /*kernel_size=*/1).stride(stride).padding(0).bias(false)),
            nn::BatchNorm2d(this->expansion*planes)
        );
        register_module("downsample", this->downsample);
    }

    this->last = nn::Sequential(
        nn::ReLU(nn::ReLUOptions().inplace(true))
    );
    register_module("last", this->last);

}


// ---------------------------------------------------------
// struct{BottleneckImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor BottleneckImpl::forward(torch::Tensor x){
    torch::Tensor out;
    out = this->layerA->forward(x);
    out = this->layerB->forward(out);
    out = this->layerC->forward(out);
    out += (this->down ? this->downsample->forward(x) : x);
    out = this->last->forward(out);
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
    else if ((typeid(m) == typeid(nn::BatchNorm2d)) || (typeid(m) == typeid(nn::BatchNorm2dImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::constant_(*w, /*weight=*/1.0);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}

