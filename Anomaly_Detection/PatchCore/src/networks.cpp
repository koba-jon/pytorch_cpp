#include <utility>
#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;
using torch::indexing::Slice;


// ----------------------------------------------------------------------
// struct{MC_ResNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MC_ResNetImpl::MC_ResNetImpl(po::variables_map &vm){

    bool basic_block;
    std::vector<long int> cfg;
    size_t feature;
    size_t expansion;
    std::string n_layers = vm["n_layers"].as<std::string>();
    if (n_layers == "18"){
        basic_block = true;
        cfg = {2, 2, 2, 2};
        feature = 64;
        expansion = 1;
    }
    else if (n_layers == "w50"){
        basic_block = false;
        cfg = {3, 4, 6, 3};
        feature = 64 * 2;
        expansion = 2;
    }
    else{
        std::cerr << "Error : The number of layers is " << n_layers << '.' << std::endl;
        std::cerr << "Error : Please choose 18 or w50." << std::endl;
        std::exit(1);
    }
    this->inplanes = 64;

    // First Downsampling
    this->first = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/3, /*out_channels=*/this->inplanes, /*kernel_size=*/7).stride(2).padding(3).bias(false)),  // {C,224,224} ===> {F,112,112}
        nn::BatchNorm2d(this->inplanes),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1))                                                                                 // {F,112,112} ===> {F,56,56}
    );
    register_module("first", this->first);

    // After the Second Time
    if (basic_block){
        BasicBlockImpl block;
        this->layer1 = this->make_layers(block, expansion, /*planes=*/feature, /*num_blocks=*/cfg.at(0), /*stride=*/1);    // {F,56,56} ===> {F*E,56,56}
        this->layer2 = this->make_layers(block, expansion, /*planes=*/feature*2, /*num_blocks=*/cfg.at(1), /*stride=*/2);  // {F*E,56,56} ===> {2F*E,28,28}
        this->layer3 = this->make_layers(block, expansion, /*planes=*/feature*4, /*num_blocks=*/cfg.at(2), /*stride=*/2);  // {2F*E,28,28} ===> {4F*E,14,14}
        this->layer4 = this->make_layers(block, expansion, /*planes=*/feature*8, /*num_blocks=*/cfg.at(3), /*stride=*/2);  // {4F*E,14,14} ===> {8F*E,7,7}
    }
    else{
        BottleneckImpl block;
        this->layer1 = this->make_layers(block, expansion, /*planes=*/feature, /*num_blocks=*/cfg.at(0), /*stride=*/1);    // {F,56,56} ===> {F*E,56,56}
        this->layer2 = this->make_layers(block, expansion, /*planes=*/feature*2, /*num_blocks=*/cfg.at(1), /*stride=*/2);  // {F*E,56,56} ===> {2F*E,28,28}
        this->layer3 = this->make_layers(block, expansion, /*planes=*/feature*4, /*num_blocks=*/cfg.at(2), /*stride=*/2);  // {2F*E,28,28} ===> {4F*E,14,14}
        this->layer4 = this->make_layers(block, expansion, /*planes=*/feature*8, /*num_blocks=*/cfg.at(3), /*stride=*/2);  // {4F*E,14,14} ===> {8F*E,7,7}
    }
    register_module("layer1", this->layer1);
    register_module("layer2", this->layer2);
    register_module("layer3", this->layer3);
    register_module("layer4", this->layer4);

    // Final Downsampling
    this->avgpool = nn::Sequential(nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({1, 1})));  // {8F*E,7,7} ===> {8F*E,1,1}
    register_module("avgpool", this->avgpool);

    // Classification
    this->classifier = nn::Sequential(nn::Linear(/*in_channels=*/feature*8*expansion, /*out_channels=*/1000));  // {8F*E} ===> {CN}
    register_module("classifier", this->classifier);

}


// -----------------------------------------------------------
// struct{MC_ResNetImpl}(nn::Module) -> function{make_layers}
// -----------------------------------------------------------
template <typename T>
nn::Sequential MC_ResNetImpl::make_layers(T &block, const size_t expansion, const size_t planes, const size_t num_blocks, const size_t stride){
    nn::Sequential layers = nn::Sequential(T(expansion, this->inplanes, planes, stride));
    this->inplanes = planes * expansion;
    for (size_t i = 1; i < num_blocks; i++){
        layers->push_back(T(expansion, this->inplanes, planes, /*stride=*/1));
    }
    return layers;
}


// ---------------------------------------------------------
// struct{MC_ResNetImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor MC_ResNetImpl::forward(torch::Tensor x){

    std::vector<long int> size;
    torch::Tensor feature, feature_l1, feature_l2, feature_l3, out;

    feature = this->first->forward(x);               // {C,224,224} ===> {F,56,56}
    feature_l1 = this->layer1->forward(feature);     // {F,56,56} ===> {F*E,56,56}
    feature_l2 = this->layer2->forward(feature_l1);  // {F*E,56,56} ===> {2F*E,28,28}
    feature_l3 = this->layer3->forward(feature_l2);  // {2F*E,28,28} ===> {4F*E,14,14}

    size = {feature_l2.size(2), feature_l2.size(3)};
    /*********************************************************/
    feature_l2 = F::avg_pool2d(feature_l2, F::AvgPool2dFuncOptions(3).stride(1).padding(1));
    feature_l2 = F::interpolate(feature_l2, F::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false));
    /*********************************************************/
    feature_l3 = F::avg_pool2d(feature_l3, F::AvgPool2dFuncOptions(3).stride(1).padding(1));
    feature_l3 = F::interpolate(feature_l3, F::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false));
    /*********************************************************/
    out = torch::cat({feature_l2, feature_l3}, 1);

    return out;

}


// --------------------------------------------------------------
// struct{MC_ResNetImpl}(nn::Module) -> function{get_total_dim}
// --------------------------------------------------------------
size_t MC_ResNetImpl::get_total_dim(torch::Device device){

    torch::Tensor x;
    size_t total_dim;

    x = torch::ones({1, 3, 224, 224}).to(device);
    total_dim = this->forward(x).size(1);

    return total_dim;

}


// ----------------------------------------------------------------------
// struct{BasicBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
BasicBlockImpl::BasicBlockImpl(const size_t expansion, const size_t inplanes, const size_t planes, const size_t stride){

    this->layerA = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/planes, /*kernel_size=*/3).stride(stride).padding(1).bias(false)),
        nn::BatchNorm2d(planes),
        nn::ReLU(nn::ReLUOptions().inplace(true))
    );
    register_module("layerA", this->layerA);

    this->layerB = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/planes, /*out_channels=*/expansion*planes, /*kernel_size=*/3).stride(1).padding(1).bias(false)),
        nn::BatchNorm2d(expansion*planes)
    );
    register_module("layerB", this->layerB);

    this->down = false;
    if ((stride != 1) || (inplanes != (expansion*planes))){
        this->down = true;
        this->downsample = nn::Sequential(
            nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/expansion*planes, /*kernel_size=*/1).stride(stride).padding(0).bias(false)),
            nn::BatchNorm2d(expansion*planes)
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
BottleneckImpl::BottleneckImpl(const size_t expansion, const size_t inplanes, const size_t planes, const size_t stride){

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
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/planes, /*out_channels=*/expansion*planes, /*kernel_size=*/1).stride(1).padding(0).bias(false)),
        nn::BatchNorm2d(expansion*planes)
    );
    register_module("layerC", this->layerC);

    this->down = false;
    if ((stride != 1) || (inplanes != (expansion*planes))){
        this->down = true;
        this->downsample = nn::Sequential(
            nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/inplanes, /*out_channels=*/expansion*planes, /*kernel_size=*/1).stride(stride).padding(0).bias(false)),
            nn::BatchNorm2d(expansion*planes)
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

