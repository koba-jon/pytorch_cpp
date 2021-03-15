#include <typeinfo>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;


// ----------------------------------------------------------------------
// struct{YOLOv1Impl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
YOLOv1Impl::YOLOv1Impl(po::variables_map &vm){

    size_t nc = vm["nc"].as<size_t>();  // the number of image channels
    size_t nb = vm["nb"].as<size_t>();  // the number of bounding box in each grid
    size_t class_num = vm["class_num"].as<size_t>();  // total classes
    this->grid = (long int)vm["ng"].as<size_t>();  // the number of grid
    this->final_features = (long int)(class_num + nb * 5);  // total classes + BB * 5=len[center_x, center_y, width, height, confidence]

    // -----------------------------------
    // 1. Convolutional Layers
    // -----------------------------------

    // 1st Layers  {C,448,448} ===> {64,112,112}
    Convolution(this->features, /*in_nc=*/nc, /*out_nc=*/64, /*ksize=*/7, /*stride=*/2, /*pad=*/3, /*BN=*/true, /*LReLU=*/true);         // {C,448,448} ===> {64,224,224}
    this->features->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));                                         // {64,224,224} ===> {64,112,112}

    // 2nd Layers  {64,112,112} ===> {192,56,56}
    Convolution(this->features, /*in_nc=*/64, /*out_nc=*/192, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);        // {64,112,112} ===> {192,112,112}
    this->features->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));                                         // {192,112,112} ===> {192,56,56}

    // 3rd Layers  {192,56,56} ===> {512,28,28}
    Convolution(this->features, /*in_nc=*/192, /*out_nc=*/128, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);       // {192,56,56} ===> {128,56,56}
    Convolution(this->features, /*in_nc=*/128, /*out_nc=*/256, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);       // {128,56,56} ===> {256,56,56}
    Convolution(this->features, /*in_nc=*/256, /*out_nc=*/256, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);       // {256,56,56} ===> {256,56,56}
    Convolution(this->features, /*in_nc=*/256, /*out_nc=*/512, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);       // {256,56,56} ===> {512,56,56}
    this->features->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));                                         // {512,56,56} ===> {512,28,28}

    // 4th Layers  {512,28,28} ===> {1024,14,14}
    for (size_t i = 0; i < 4; i++){
        Convolution(this->features, /*in_nc=*/512, /*out_nc=*/256, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);   // {512,28,28} ===> {256,28,28}
        Convolution(this->features, /*in_nc=*/256, /*out_nc=*/512, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);   // {256,28,28} ===> {512,28,28}
    }
    Convolution(this->features, /*in_nc=*/512, /*out_nc=*/512, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);       // {512,28,28} ===> {512,28,28}
    Convolution(this->features, /*in_nc=*/512, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);      // {512,28,28} ===> {1024,28,28}
    this->features->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));                                         // {1024,28,28} ===> {1024,14,14}

    // 5th Layers  {1024,14,14} ===> {1024,7,7}
    for (size_t i = 0; i < 2; i++){
        Convolution(this->features, /*in_nc=*/1024, /*out_nc=*/512, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);  // {1024,14,14} ===> {512,14,14}
        Convolution(this->features, /*in_nc=*/512, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);  // {512,14,14} ===> {1024,14,14}
    }
    Convolution(this->features, /*in_nc=*/1024, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);     // {1024,14,14} ===> {1024,14,14}
    Convolution(this->features, /*in_nc=*/1024, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/2, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);     // {1024,14,14} ===> {1024,7,7}

    // 6th Layers  {1024,7,7} ===> {1024,7,7}
    Convolution(this->features, /*in_nc=*/1024, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);     // {1024,7,7} ===> {1024,7,7}
    Convolution(this->features, /*in_nc=*/1024, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);     // {1024,7,7} ===> {1024,7,7}
    register_module("features", this->features);

    // -----------------------------------
    // 2. Average Pooling Layers
    // -----------------------------------
    this->avgpool = nn::Sequential(nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({this->grid, this->grid})));  // {1024,X,X} ===> {1024,7,7}
    register_module("avgpool", this->avgpool);

    // -----------------------------------
    // 3. Fully Connected Layers
    // -----------------------------------
    this->classifier = nn::Sequential(
        nn::Linear(/*in_channels=*/1024 * 7 * 7, /*out_channels=*/4096),                                    // {1024*7*7} ===> {4096}
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)),
        nn::Dropout(0.5),
        nn::Linear(/*in_channels=*/4096, /*out_channels=*/this->grid * this->grid * this->final_features),  // {4096} ===> {G*G*FF}
        nn::Sigmoid()                                                                                       // [-inf,+inf] ===> (0,1)
    );
    register_module("classifier", this->classifier);

}


// ---------------------------------------------------------
// struct{YOLOv1Impl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor YOLOv1Impl::forward(torch::Tensor x){
    torch::Tensor feature, out;
    feature = this->features->forward(x);                                         // {C,448,448} ===> {1024,7,7}
    feature = this->avgpool->forward(feature);                                    // {1024,X,X} ===> {1024,7,7}
    feature = feature.view({feature.size(0), -1});                                // {1024,7,7} ===> {1024*7*7}
    out = this->classifier->forward(feature);                                     // {1024*7*7} ===> {G*G*FF}
    out = out.view({out.size(0), this->grid, this->grid, this->final_features});  // {G*G*FF} ===> {G,G,FF}
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
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
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
        sq->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
    }
    return;
}
