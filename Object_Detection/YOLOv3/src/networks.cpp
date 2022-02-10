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
// struct{YOLOv3Impl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
YOLOv3Impl::YOLOv3Impl(po::variables_map &vm){

    size_t nc = vm["nc"].as<size_t>();  // the number of image channels
    size_t na = vm["na"].as<size_t>();  // the number of anchor
    size_t class_num = vm["class_num"].as<size_t>();  // total classes
    long int final_features = (long int)(na * (class_num + 5));  // anchors * (total classes + 5=len[t_x, t_y, t_w, t_h, confidence])

    // -----------------------------------
    // 1. Convolutional Layers: Stage 1
    // -----------------------------------

    // 1st Layers  {C,608,608} ===> {64,304,304}
    this->stage1->push_back(ConvBlockImpl(/*in_nc=*/nc, /*out_nc=*/32, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                  // {C,608,608} ===> {32,608,608}
    this->stage1->push_back(ConvBlockImpl(/*in_nc=*/32, /*out_nc=*/64, /*ksize=*/3, /*stride=*/2, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                  // {32,608,608} ===> {64,304,304}

    // 2nd Layers  {64,304,304} ===> {128,152,152}
    this->stage1->push_back(ResBlockImpl(64, 32));                                                                                                           // {64,304,304} ===> {32,304,304} ===> {64,304,304}
    this->stage1->push_back(ConvBlockImpl(/*in_nc=*/64, /*out_nc=*/128, /*ksize=*/3, /*stride=*/2, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                 // {64,304,304} ===> {128,152,152}

    // 3rd Layers  {128,152,152} ===> {256,76,76}
    this->stage1->push_back(ResBlockImpl(128, 64));                                                                                                          // {128,152,152} ===> {64,152,152} ===> {128,152,152}
    this->stage1->push_back(ResBlockImpl(128, 64));                                                                                                          // {128,152,152} ===> {64,152,152} ===> {128,152,152}
    this->stage1->push_back(ConvBlockImpl(/*in_nc=*/128, /*out_nc=*/256, /*ksize=*/3, /*stride=*/2, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                // {128,152,152} ===> {256,76,76}
    for (size_t i = 0; i < 8; i++){
        this->stage1->push_back(ResBlockImpl(256, 128));                                                                                                     // {256,76,76} ===> {128,76,76} ===> {256,76,76}
    }
    register_module("stage1", this->stage1);

    // -----------------------------------
    // 2. Convolutional Layers: Stage 2
    // -----------------------------------
    // Layers  {256,76,76} ===> {512,38,38}
    this->stage2->push_back(ConvBlockImpl(/*in_nc=*/256, /*out_nc=*/512, /*ksize=*/3, /*stride=*/2, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                // {256,76,76} ===> {512,38,38}
    for (size_t i = 0; i < 8; i++){
        this->stage2->push_back(ResBlockImpl(512, 256));                                                                                                     // {512,38,38} ===> {256,38,38} ===> {512,38,38}
    }
    register_module("stage2", this->stage2);

    // -----------------------------------
    // 3. Convolutional Layers: Stage 3
    // -----------------------------------
    // Layers  {512,38,38} ===> {512,19,19}
    this->stage3->push_back(ConvBlockImpl(/*in_nc=*/512, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/2, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));               // {512,38,38} ===> {1024,19,19}
    for (size_t i = 0; i < 4; i++){
        this->stage3->push_back(ResBlockImpl(1024, 512));                                                                                                    // {1024,19,19} ===> {512,19,19} ===> {1024,19,19}
    }
    this->stage3->push_back(ConvBlockImpl(/*in_nc=*/1024, /*out_nc=*/512, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));               // {1024,19,19} ===> {512,19,19}
    this->stage3->push_back(ConvBlockImpl(/*in_nc=*/512, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));               // {512,19,19} ===> {1024,19,19}
    this->stage3->push_back(ConvBlockImpl(/*in_nc=*/1024, /*out_nc=*/512, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));               // {1024,19,19} ===> {512,19,19}
    this->stage3->push_back(ConvBlockImpl(/*in_nc=*/512, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));               // {512,19,19} ===> {1024,19,19}
    this->stage3->push_back(ConvBlockImpl(/*in_nc=*/1024, /*out_nc=*/512, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));               // {1024,19,19} ===> {512,19,19}
    register_module("stage3", this->stage3);

    // -----------------------------------
    // 4. Convolutional Layers: Stage 4
    // -----------------------------------
    // Layers  {512,19,19} ===> {256,19,19}
    this->stage4->push_back(ConvBlockImpl(/*in_nc=*/512, /*out_nc=*/256, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));                // {512,19,19} ===> {256,19,19}
    register_module("stage4", this->stage4);

    // -----------------------------------
    // 5. Convolutional Layers: Stage 5
    // -----------------------------------
    // Layers  {512+256,38,38} ===> {256,38,38}
    this->stage5->push_back(ConvBlockImpl(/*in_nc=*/512+256, /*out_nc=*/256, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));            // {512+256,38,38} ===> {256,38,38}
    this->stage5->push_back(ConvBlockImpl(/*in_nc=*/256, /*out_nc=*/512, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                // {256,38,38} ===> {512,38,38}
    this->stage5->push_back(ConvBlockImpl(/*in_nc=*/512, /*out_nc=*/256, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));                // {512,38,38} ===> {256,38,38}
    this->stage5->push_back(ConvBlockImpl(/*in_nc=*/256, /*out_nc=*/512, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                // {256,38,38} ===> {512,38,38}
    this->stage5->push_back(ConvBlockImpl(/*in_nc=*/512, /*out_nc=*/256, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));                // {512,38,38} ===> {256,38,38}
    register_module("stage5", this->stage5);

    // -----------------------------------
    // 6. Convolutional Layers: Stage 6
    // -----------------------------------
    // Layers  {256,38,38} ===> {128,38,38}
    this->stage6->push_back(ConvBlockImpl(/*in_nc=*/256, /*out_nc=*/128, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));                // {256,38,38} ===> {128,38,38}
    register_module("stage6", this->stage6);

    // -----------------------------------
    // A. Convolutional Layers: Stage A
    // -----------------------------------
    // Layers  {512,19,19} ===> {A*(CN+5),19,19}
    this->stageA->push_back(ConvBlockImpl(/*in_nc=*/512, /*out_nc=*/1024, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));               // {512,19,19} ===> {1024,19,19}
    this->stageA->push_back(ConvBlockImpl(/*in_nc=*/1024, /*out_nc=*/final_features, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/false, /*LReLU=*/false));  // {1024,19,19} ===> {A*(CN+5),19,19}
    register_module("stageA", this->stageA);

    // -----------------------------------
    // B. Convolutional Layers: Stage B
    // -----------------------------------
    // Layers  {256,38,38} ===> {A*(CN+5),38,38}
    this->stageB->push_back(ConvBlockImpl(/*in_nc=*/256, /*out_nc=*/512, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                // {256,38,38} ===> {512,38,38}
    this->stageB->push_back(ConvBlockImpl(/*in_nc=*/512, /*out_nc=*/final_features, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/false, /*LReLU=*/false));   // {512,38,38} ===> {A*(CN+5),38,38}
    register_module("stageB", this->stageB);

    // -----------------------------------
    // C. Convolutional Layers: Stage C
    // -----------------------------------
    // Layers  {256+128,76,76} ===> {A*(CN+5),76,76}
    this->stageC->push_back(ConvBlockImpl(/*in_nc=*/256+128, /*out_nc=*/128, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));            // {256+128,76,76} ===> {128,76,76}
    this->stageC->push_back(ConvBlockImpl(/*in_nc=*/128, /*out_nc=*/256, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                // {128,76,76} ===> {256,76,76}
    this->stageC->push_back(ConvBlockImpl(/*in_nc=*/256, /*out_nc=*/128, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));                // {256,76,76} ===> {128,76,76}
    this->stageC->push_back(ConvBlockImpl(/*in_nc=*/128, /*out_nc=*/256, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                // {128,76,76} ===> {256,76,76}
    this->stageC->push_back(ConvBlockImpl(/*in_nc=*/256, /*out_nc=*/128, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));                // {256,76,76} ===> {128,76,76}
    this->stageC->push_back(ConvBlockImpl(/*in_nc=*/128, /*out_nc=*/256, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));                // {128,76,76} ===> {256,76,76}
    this->stageC->push_back(ConvBlockImpl(/*in_nc=*/256, /*out_nc=*/final_features, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/false, /*LReLU=*/false));   // {256,76,76} ===> {A*(CN+5),76,76}
    register_module("stageC", this->stageC);

}


// ---------------------------------------------------------
// struct{YOLOv3Impl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
std::vector<torch::Tensor> YOLOv3Impl::forward(torch::Tensor x){

    torch::Tensor feature1, feature2, feature3, feature4, feature5, feature6;
    torch::Tensor outA, outB, outC;
    std::vector<torch::Tensor> out;

    feature1 = this->stage1->forward(x);                                            // {C,608,608} ===> {256,76,76}
    feature2 = this->stage2->forward(feature1);                                     // {256,76,76} ===> {512,38,38}
    feature3 = this->stage3->forward(feature2);                                     // {512,38,38} ===> {512,19,19}
    feature4 = this->stage4->forward(feature3);                                     // {512,19,19} ===> {256,19,19}
    feature4 = UpSampling(feature4, {feature2.size(2), feature2.size(3)});          // {256,19,19} ===> {256,38,38}
    feature5 = this->stage5->forward(torch::cat({feature2, feature4}, /*dim=*/1));  // {512+256,38,38} ===> {256,38,38}
    feature6 = this->stage6->forward(feature5);                                     // {256,38,38} ===> {128,38,38}
    feature6 = UpSampling(feature6, {feature1.size(2), feature1.size(3)});          // {128,38,38} ===> {128,76,76}

    outA = this->stageA->forward(feature3);                                         // {512,19,19} ===> {A*(CN+5),19,19}
    outA = outA.permute({0, 2, 3, 1}).contiguous();                                 // {A*(CN+5),19,19} ===> {19,19,A*(CN+5)}
    out.push_back(outA);

    outB = this->stageB->forward(feature5);                                         // {256,38,38} ===> {A*(CN+5),38,38}
    outB = outB.permute({0, 2, 3, 1}).contiguous();                                 // {A*(CN+5),38,38} ===> {38,38,A*(CN+5)}
    out.push_back(outB);

    outC = this->stageC->forward(torch::cat({feature1, feature6}, /*dim=*/1));      // {256+128,76,76} ===> {A*(CN+5),76,76}
    outC = outC.permute({0, 2, 3, 1}).contiguous();                                 // {A*(CN+5),76,76} ===> {76,76,A*(CN+5)}
    out.push_back(outC);

    return out;

}


// ---------------------------------------------------------
// struct{ResBlockImpl}(nn::Module) -> constructor
// ---------------------------------------------------------
ResBlockImpl::ResBlockImpl(const size_t outside_nc, const size_t inside_nc){
    this->sq->push_back(ConvBlockImpl(/*in_nc=*/outside_nc, /*out_nc=*/inside_nc, /*ksize=*/1, /*stride=*/1, /*pad=*/0, /*BN=*/true, /*LReLU=*/true));
    this->sq->push_back(ConvBlockImpl(/*in_nc=*/inside_nc, /*out_nc=*/outside_nc, /*ksize=*/3, /*stride=*/1, /*pad=*/1, /*BN=*/true, /*LReLU=*/true));
    register_module("ResBlock", this->sq);
}


// ---------------------------------------------------------------
// struct{ResBlockImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------------
torch::Tensor ResBlockImpl::forward(torch::Tensor x){
    torch::Tensor out;
    out = x + this->sq->forward(x);
    return out;
}


// ---------------------------------------------------------
// struct{ConvBlockImpl}(nn::Module) -> constructor
// ---------------------------------------------------------
ConvBlockImpl::ConvBlockImpl(const size_t in_nc, const size_t out_nc, const size_t ksize, const size_t stride, const size_t pad, const bool BN, const bool LReLU, const bool bias){
    this->sq->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, ksize).stride(stride).padding(pad).bias(bias)));
    if (BN){
        this->sq->push_back(nn::BatchNorm2d(out_nc));
    }
    if (LReLU){
        this->sq->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
    }
    register_module("ConvBlock", this->sq);
}


// ---------------------------------------------------------------
// struct{ConvBlockImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------------
torch::Tensor ConvBlockImpl::forward(torch::Tensor x){
    torch::Tensor out;
    out = this->sq->forward(x);
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
// function{UpSampling}
// ----------------------------
torch::Tensor UpSampling(torch::Tensor x, const std::vector<long int> shape){
    torch::Tensor out;
    out = F::interpolate(x, F::InterpolateFuncOptions().size(shape).mode(torch::kNearest));
    return out;
}

