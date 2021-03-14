#include <vector>
#include <typeinfo>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;


// ----------------------------------------------------------------------
// struct{YOLOv2Impl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
YOLOv2Impl::YOLOv2Impl(po::variables_map &vm){

    constexpr size_t stride = 2;  // stride of reorganization

    size_t nc = vm["nc"].as<size_t>();  // the number of image channels
    size_t na = vm["na"].as<size_t>();  // the number of anchor
    size_t class_num = vm["class_num"].as<size_t>();  // total classes
    long int final_features = (long int)(na * (class_num + 5));  // anchors * (total classes + 5=len[t_x, t_y, t_w, t_h, confidence])

    // -----------------------------------
    // 1. Convolutional Layers: Stage 1
    // -----------------------------------

    // 1st Layers  {C,608,608} ===> {32,304,304}
    Convolution(this->stage1, /*in_nc=*/nc, /*out_nc=*/32, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);                  // {C,608,608} ===> {32,608,608}
    this->stage1->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));                                                  // {32,608,608} ===> {32,304,304}

    // 2nd Layers  {32,304,304} ===> {64,152,152}
    Convolution(this->stage1, /*in_nc=*/32, /*out_nc=*/64, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);                  // {32,304,304} ===> {64,304,304}
    this->stage1->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));                                                  // {64,304,304} ===> {64,152,152}

    // 3rd Layers  {64,152,152} ===> {128,76,76}
    Convolution(this->stage1, /*in_nc=*/64, /*out_nc=*/128, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);                 // {64,152,152} ===> {128,152,152}
    Convolution(this->stage1, /*in_nc=*/128, /*out_nc=*/64, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);                 // {128,152,152} ===> {64,152,152}
    Convolution(this->stage1, /*in_nc=*/64, /*out_nc=*/128, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);                 // {64,152,152} ===> {128,152,152}
    this->stage1->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));                                                  // {128,152,152} ===> {128,76,76}

    // 4th Layers  {128,76,76} ===> {256,38,38}
    Convolution(this->stage1, /*in_nc=*/128, /*out_nc=*/256, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);                // {128,76,76} ===> {256,76,76}
    Convolution(this->stage1, /*in_nc=*/256, /*out_nc=*/128, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);                // {256,76,76} ===> {128,76,76}
    Convolution(this->stage1, /*in_nc=*/128, /*out_nc=*/256, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);                // {128,76,76} ===> {256,76,76}
    this->stage1->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));                                                  // {256,76,76} ===> {256,38,38}

    // 5th Layers  {256,38,38} ===> {512,38,38}
    for (size_t i = 0; i < 2; i++){
        Convolution(this->stage1, /*in_nc=*/256, /*out_nc=*/512, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);            // {256,38,38} ===> {512,38,38}
        Convolution(this->stage1, /*in_nc=*/512, /*out_nc=*/256, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);            // {512,38,38} ===> {256,38,38}
    }
    Convolution(this->stage1, /*in_nc=*/256, /*out_nc=*/512, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);                // {256,38,38} ===> {512,38,38}
    this->stage1->push_back(FloorAvgPool2dImpl({stride, stride}));                                                                              // {C,H,W} ===> {C,H-H_S,W-W_S} (H % stride = H_S, W % stride = W_S)
    register_module("stage1", this->stage1);

    // -----------------------------------
    // 2a. Convolutional Layers: Stage 2a
    // -----------------------------------

    // 1th Layers  {512,38,38} ===> {512,19,19}
    this->stage2a->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));                                                 // {512,38,38} ===> {512,19,19}

    // 2th Layers  {512,19,19} ===> {1024,19,19}
    for (size_t i = 0; i < 2; i++){
        Convolution(this->stage2a, /*in_nc=*/512, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);          // {512,19,19} ===> {1024,19,19}
        Convolution(this->stage2a, /*in_nc=*/1024, /*out_nc=*/512, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);          // {1024,19,19} ===> {512,19,19}
    }
    Convolution(this->stage2a, /*in_nc=*/512, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);              // {512,19,19} ===> {1024,19,19}
    Convolution(this->stage2a, /*in_nc=*/1024, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);             // {1024,19,19} ===> {1024,19,19}
    Convolution(this->stage2a, /*in_nc=*/1024, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);             // {1024,19,19} ===> {1024,19,19}
    register_module("stage2a", this->stage2a);

    // -----------------------------------
    // 2b. Convolutional Layers: Stage 2b
    // -----------------------------------

    // 1th Layers  {512,38,38} ===> {256,19,19}
    Convolution(this->stage2b, /*in_nc=*/512, /*out_nc=*/64, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true);                // {512,38,38} ===> {64,38,38}
    this->stage2b->push_back(ReorganizeImpl(stride));                                                                                           // {64,38,38} ===> {256,19,19}
    register_module("stage2b", this->stage2b);

    // -----------------------------------
    // 3. Convolutional Layers: Stage 3
    // -----------------------------------

    // 1th Layers  {1024+256,19,19} ===> {A*(CN+5),19,19}
    Convolution(this->stage3, /*in_nc=*/1024+256, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true);          // {1024+256,19,19} ===> {1024,19,19}
    Convolution(this->stage3, /*in_nc=*/1024, /*out_nc=*/final_features, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/false, /*LReLU=*/false, /*bias=*/true);  // {1024,19,19} ===> {A*(CN+5),19,19}
    register_module("stage3", this->stage3);

}


// ---------------------------------------------------------
// struct{YOLOv2Impl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor YOLOv2Impl::forward(torch::Tensor x){
    torch::Tensor feature1, feature2a, feature2b, feature2, out;
    feature1 = this->stage1->forward(x);                       // {C,608,608} ===> {512,38,38}
    feature2a = this->stage2a->forward(feature1);              // {512,38,38} ===> {1024,19,19}
    feature2b = this->stage2b->forward(feature1);              // {512,38,38} ===> {256,19,19}
    feature2 = torch::cat({feature2a, feature2b}, /*dim=*/1);  // {1024,19,19} + {256,19,19} ===> {1024+256,19,19}
    out = this->stage3->forward(feature2);                     // {1024+256,19,19} ===> {A*(CN+5),19,19}
    out = out.permute({0, 2, 3, 1}).contiguous();              // {A*(CN+5),19,19} ===> {19,19,A*(CN+5)}
    return out;
}


// ---------------------------------------------------------
// struct{FloorAvgPool2dImpl}(nn::Module) -> constructor
// ---------------------------------------------------------
FloorAvgPool2dImpl::FloorAvgPool2dImpl(std::vector<long int> multiple_){
    this->multiple = multiple_;
}


// ---------------------------------------------------------------
// struct{FloorAvgPool2dImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------------
torch::Tensor FloorAvgPool2dImpl::forward(torch::Tensor x){
    torch::Tensor out;
    long int H = x.size(2);
    long int W = x.size(3);
    long int H_surplus = H % this->multiple.at(0);
    long int W_surplus = W % this->multiple.at(1);
    out = F::adaptive_avg_pool2d(x, F::AdaptiveAvgPool2dFuncOptions({H - H_surplus, W - W_surplus}));
    return out;
}


// ---------------------------------------------------------
// struct{ReorganizeImpl}(nn::Module) -> constructor
// ---------------------------------------------------------
ReorganizeImpl::ReorganizeImpl(long int stride_){
    this->stride = stride_;
}


// ---------------------------------------------------------
// struct{ReorganizeImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor ReorganizeImpl::forward(torch::Tensor x){
    torch::Tensor out;
    long int N = x.size(0);
    long int C = x.size(1);
    long int H = x.size(2);
    long int W = x.size(3);
    long int ws = this->stride;
    long int hs = this->stride;
    out = x.view({N, C, (H / hs), hs, (W / ws), ws}).transpose(3, 4).contiguous();       // {N,C,H,W} ===> {N,C,H/2,2,W/2,2} ===> {N,C,H/2,W/2,2,2}
    out = out.view({N, C, (H / hs * W / ws), (hs * ws)}).transpose(2, 3).contiguous();     // {N,C,H/2,W/2,2,2} ===> {N,C,H*W/4,4} ===> {N,C,4,H*W/4}
    out = out.view({N, C, (hs * ws), (H / hs), (W / ws)}).transpose(1, 2).contiguous();  // {N,C,4,H*W/4} ===> {N,C,4,H/2,W/2} ===> {N,4,C,H/2,W/2}
    out = out.view({N, (hs * ws * C), (H / hs), (W / ws)}).contiguous();                 // {N,4,C,H/2,W/2} ===> {N,4C,H/2,W/2}
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
