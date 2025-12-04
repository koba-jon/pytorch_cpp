#include <iostream>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <cstdlib>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;


// ----------------------------------------------------------------------
// struct{StochasticDepthImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
StochasticDepthImpl::StochasticDepthImpl(const float p_){
    this->p = p_;
}


// ----------------------------------------------------------------------
// struct{StochasticDepthImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor StochasticDepthImpl::forward(torch::Tensor x){

    constexpr float eps = 1e-5;
    float p_bar;
    torch::Tensor mask, out;

    if (!this->is_training() || p < eps) return x;
    else if (p > 1.0 - eps) return torch::zeros_like(x);

    p_bar = 1.0 - this->p;
    mask = torch::bernoulli(torch::full({x.size(0), 1, 1, 1}, p_bar, x.options()));
    out = x / p_bar * mask;

    return out;

}


// ----------------------------------------------------------------------
// struct{StochasticDepthImpl}(nn::Module) -> function{pretty_print}
// ----------------------------------------------------------------------
void StochasticDepthImpl::pretty_print(std::ostream& stream) const{
    stream << "StochasticDepth(p=" << this->p << ")";
    return;
}


// ----------------------------------------------------------------------
// struct{Conv2dNormActivationImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
Conv2dNormActivationImpl::Conv2dNormActivationImpl(const size_t in_nc, const size_t out_nc, const size_t kernel_size, const size_t stride, const size_t padding, const size_t groups, const float eps, const float momentum, const bool SiLU){
    this->model = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/in_nc, /*out_channels=*/out_nc, /*kernel_size=*/kernel_size).stride(stride).padding(padding).groups(groups).bias(false)),
        nn::BatchNorm2d(nn::BatchNormOptions(out_nc).eps(eps).momentum(momentum))  
    );
    if (SiLU) this->model->push_back(nn::SiLU());
    register_module("Conv2dNormActivation", this->model);
}


// ----------------------------------------------------------------------
// struct{Conv2dNormActivationImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor Conv2dNormActivationImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);
    return out;
}


// ----------------------------------------------------------------------
// struct{SqueezeExcitationImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
SqueezeExcitationImpl::SqueezeExcitationImpl(const size_t in_nc, const size_t mid){

    this->avgpool = nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({1, 1}));
    register_module("avgpool", this->avgpool);

    this->fc1 = nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/in_nc, /*out_channels=*/mid, /*kernel_size=*/1).stride(1).padding(0).groups(1).bias(true));
    register_module("fc1", this->fc1);

    this->act = nn::SiLU();
    register_module("activation", this->act);

    this->fc2 = nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/mid, /*out_channels=*/in_nc, /*kernel_size=*/1).stride(1).padding(0).groups(1).bias(true));
    register_module("fc2", this->fc2);

    this->scale_act = nn::Sigmoid();
    register_module("scale_activation", this->scale_act);

}


// ----------------------------------------------------------------------
// struct{SqueezeExcitationImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor SqueezeExcitationImpl::forward(torch::Tensor x){
    torch::Tensor out;
    out = this->avgpool->forward(x);
    out = this->fc1->forward(out);
    out = this->act->forward(out);
    out = this->fc2->forward(out);
    out = this->scale_act->forward(out) * x;
    return out;
}


// ----------------------------------------------------------------------
// struct{MBConvImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MBConvImpl::MBConvImpl(const size_t in_nc, const size_t out_nc, const size_t kernel_size, const size_t stride, const size_t exp, const float eps, const float momentum, const float dropconnect){

    constexpr size_t reduce = 4;
    size_t mid = in_nc * exp;
    this->residual = ((stride == 1) && (in_nc == out_nc));

    if (exp != 1) this->block->push_back(Conv2dNormActivation(in_nc, mid, /*kernel_size=*/1, /*stride=*/1, /*padding=*/0, /*groups=*/1, /*eps=*/eps, /*momentum=*/momentum, /*SiLU=*/true));
    this->block->push_back(Conv2dNormActivation(mid, mid, /*kernel_size=*/kernel_size, /*stride=*/stride, /*padding=*/kernel_size / 2, /*groups=*/mid, /*eps=*/eps, /*momentum=*/momentum, /*SiLU=*/true));
    this->block->push_back(SqueezeExcitation(mid, std::max(1, int(in_nc / reduce))));
    this->block->push_back(Conv2dNormActivation(mid, out_nc, /*kernel_size=*/1, /*stride=*/1, /*padding=*/0, /*groups=*/1, /*eps=*/eps, /*momentum=*/momentum, /*SiLU=*/false));
    register_module("block", this->block);
    
    this->sd = StochasticDepth(dropconnect);
    register_module("stochastic_depth", this->sd);

}


// ----------------------------------------------------------------------
// struct{MBConvImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor MBConvImpl::forward(torch::Tensor x){

    torch::Tensor out;

    out = this->block->forward(x);
    if (this->residual){
        out = this->sd->forward(out) + x;
    }

    return out;

}


// ----------------------------------------------------------------------
// struct{MC_EfficientNetImpl}(nn::Module) -> function{make_divisible}
// ----------------------------------------------------------------------
size_t MC_EfficientNetImpl::make_divisible(size_t v, size_t divisor){
    int new_v = std::max(divisor, (v + divisor / 2) / divisor * divisor);
    if (new_v < 0.9 * v) new_v += divisor;
    return new_v;
}


// ----------------------------------------------------------------------
// struct{MC_EfficientNetImpl}(nn::Module) -> function{round_filters}
// ----------------------------------------------------------------------
size_t MC_EfficientNetImpl::round_filters(size_t c, double width_mul){
    return make_divisible(size_t(std::round(c * width_mul)));
}


// ----------------------------------------------------------------------
// struct{MC_EfficientNetImpl}(nn::Module) -> function{round_repeats}
// ----------------------------------------------------------------------
size_t MC_EfficientNetImpl::round_repeats(size_t r, double depth_mul){
    return std::max(1, int(std::ceil(r * depth_mul)));
}


// ----------------------------------------------------------------------
// struct{MC_EfficientNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MC_EfficientNetImpl::MC_EfficientNetImpl(po::variables_map &vm){

    constexpr size_t stem_feature = 32;
    constexpr size_t head_feature = 1280;
    size_t stem_nc, head_nc, in_nc, out_nc, block_idx, total_blocks, repeats, stride;
    float dropconnect;

    // (0.a) Setting for network's config
    std::string network = vm["network"].as<std::string>();
    if (network == "B0") this->cfg = {1.0, 1.0, 224, 0.2, 1e-5, 0.1, 0.2};
    else if (network == "B1") this->cfg = {1.0, 1.1, 240, 0.2, 1e-5, 0.1, 0.2};
    else if (network == "B2") this->cfg = {1.1, 1.2, 260, 0.3, 1e-5, 0.1, 0.2};
    else if (network == "B3") this->cfg = {1.2, 1.4, 300, 0.3, 1e-5, 0.1, 0.2};
    else if (network == "B4") this->cfg = {1.4, 1.8, 380, 0.4, 1e-5, 0.1, 0.2};
    else if (network == "B5") this->cfg = {1.6, 2.2, 456, 0.4, 0.001, 0.01, 0.2};
    else if (network == "B6") this->cfg = {1.8, 2.6, 528, 0.5, 0.001, 0.01, 0.2};
    else if (network == "B7") this->cfg = {2.0, 3.1, 600, 0.5, 0.001, 0.01, 0.2};
    else if (network == "B8") this->cfg = {2.2, 3.6, 672, 0.5, 0.001, 0.01, 0.2};
    else if (network == "L2") this->cfg = {4.3, 5.3, 800, 0.5, 0.001, 0.01, 0.2};
    else{
        std::cerr << "Error : The type of network is " << network << '.' << std::endl;
        std::cerr << "Error : Please choose B0, B1, B2, B3, B4, B5, B6, B7, B8 or L2." << std::endl;
        std::exit(1);
    }

    // (0.b) Setting for block's config
    std::vector<BlockConfig> bcfg = {
        {3, 1, 16, 1, 1}, {3, 6, 24, 2, 2}, {5, 6, 40, 2, 2}, {3, 6, 80, 3, 2}, {5, 6, 112, 3, 1}, {5, 6, 192, 4, 2}, {3, 6, 320, 1, 1}
    };

    // (1) Stem layer
    stem_nc = this->round_filters(stem_feature, this->cfg.width_mul);
    this->features->push_back(Conv2dNormActivation(vm["nc"].as<size_t>(), stem_nc, /*kernel_size=*/3, /*stride=*/2, /*padding=*/1, /*groups=*/1, /*eps=*/this->cfg.eps, /*momentum=*/this->cfg.momentum, /*SiLU=*/true));

    // (2.a) Bone layer
    total_blocks = 0;
    for (size_t i = 0; i < bcfg.size(); i++){
        total_blocks += this->round_repeats(bcfg[i].r, this->cfg.depth_mul);
    }

    // (2.b) Bone layer
    in_nc = stem_nc;
    block_idx = 0;
    for (size_t i = 0; i < bcfg.size(); i++){
        repeats = this->round_repeats(bcfg[i].r, this->cfg.depth_mul);
        out_nc = this->round_filters(bcfg[i].c, this->cfg.width_mul);
        for (size_t j = 0; j < repeats; j++){
            stride = (j == 0) ? bcfg[i].s : 1;
            dropconnect = this->cfg.stochastic_depth_prob * (double)block_idx / (double)std::max(1, int(total_blocks));
            this->features->push_back(MBConvImpl(in_nc, out_nc, bcfg[i].k, stride, bcfg[i].exp, this->cfg.eps, this->cfg.momentum, dropconnect));
            in_nc = out_nc;
            block_idx++;
        }
    }

    // (3) Head layer
    head_nc = this->round_filters(head_feature, this->cfg.width_mul);
    this->features->push_back(Conv2dNormActivation(in_nc, head_nc, /*kernel_size=*/1, /*stride=*/1, /*padding=*/0, /*groups=*/1, /*eps=*/this->cfg.eps, /*momentum=*/this->cfg.momentum, /*SiLU=*/true));
    register_module("features", this->features);

    // (4) Global Average Pooling
    this->avgpool = nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({1, 1}));
    register_module("avgpool", this->avgpool);

    // (5) Fully Connected Layer
    this->classifier = nn::Sequential(
        nn::Dropout(this->cfg.dropout),
        nn::Linear(head_nc, vm["class_num"].as<size_t>())
    );
    register_module("classifier", this->classifier);

}


// ---------------------------------------------------------
// struct{MC_EfficientNetImpl}(nn::Module) -> function{init}
// ---------------------------------------------------------
void MC_EfficientNetImpl::init(){
    this->apply(weights_init);
    return;
}


// ---------------------------------------------------------
// struct{MC_EfficientNetImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor MC_EfficientNetImpl::forward(torch::Tensor x){
    torch::Tensor feature, out;
    feature = this->features->forward(x);
    feature = this->avgpool->forward(feature);
    feature = feature.view({feature.size(0), -1});
    out = this->classifier->forward(feature);
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
        if (w != nullptr) nn::init::kaiming_normal_(*w, /*a=*/0.0, torch::kFanOut);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        double bound = 1.0 / std::sqrt((double)(*w).size(0));
        if (w != nullptr) nn::init::uniform_(*w, -bound, bound);
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

