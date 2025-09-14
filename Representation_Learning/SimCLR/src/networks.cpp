#include <cmath>
#include <tuple>
#include <vector>
#include <random>
#include <typeinfo>
#include <cstdlib>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;


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


// ----------------------------------------------------------------------
// struct{ResNetEncoderImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ResNetEncoderImpl::ResNetEncoderImpl(size_t n_layers, size_t nf, size_t &head){

    constexpr size_t nc = 3;

    bool basic_block;
    std::vector<long int> cfg;
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
    this->inplanes = nf;

    // First Downsampling
    this->first = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/nc, /*out_channels=*/this->inplanes, /*kernel_size=*/7).stride(2).padding(3).bias(false)),  // {C,224,224} ===> {F,112,112}
        nn::BatchNorm2d(this->inplanes),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1))                                                                                 // {F,112,112} ===> {F,56,56}
    );
    register_module("first", this->first);

    // After the Second Time
    if (basic_block){
        BasicBlock block;
        head = nf * 8 * block->expansion;
        this->layer1 = this->make_layers(block, nf, /*num_blocks=*/cfg.at(0), /*stride=*/1);    // {F,56,56} ===> {F*E,56,56}
        this->layer2 = this->make_layers(block, nf*2, /*num_blocks=*/cfg.at(1), /*stride=*/2);  // {F*E,56,56} ===> {2F*E,28,28}
        this->layer3 = this->make_layers(block, nf*4, /*num_blocks=*/cfg.at(2), /*stride=*/2);  // {2F*E,28,28} ===> {4F*E,14,14}
        this->layer4 = this->make_layers(block, nf*8, /*num_blocks=*/cfg.at(3), /*stride=*/2);  // {4F*E,14,14} ===> {8F*E,7,7}
    }
    else{
        Bottleneck block;
        head = nf * 8 * block->expansion;
        this->layer1 = this->make_layers(block, nf, /*num_blocks=*/cfg.at(0), /*stride=*/1);    // {F,56,56} ===> {F*E,56,56}
        this->layer2 = this->make_layers(block, nf*2, /*num_blocks=*/cfg.at(1), /*stride=*/2);  // {F*E,56,56} ===> {2F*E,28,28}
        this->layer3 = this->make_layers(block, nf*4, /*num_blocks=*/cfg.at(2), /*stride=*/2);  // {2F*E,28,28} ===> {4F*E,14,14}
        this->layer4 = this->make_layers(block, nf*8, /*num_blocks=*/cfg.at(3), /*stride=*/2);  // {4F*E,14,14} ===> {8F*E,7,7}
    }
    register_module("layer1", this->layer1);
    register_module("layer2", this->layer2);
    register_module("layer3", this->layer3);
    register_module("layer4", this->layer4);

}


// -----------------------------------------------------------
// struct{ResNetEncoderImpl}(nn::Module) -> function{make_layers}
// -----------------------------------------------------------
template <typename T>
nn::Sequential ResNetEncoderImpl::make_layers(T &block, const size_t planes, const size_t num_blocks, const size_t stride){
    nn::Sequential layers = nn::Sequential(T(this->inplanes, planes, stride));
    this->inplanes = planes * block->expansion;
    for (size_t i = 1; i < num_blocks; i++){
        layers->push_back(T(this->inplanes, planes, /*stride=*/1));
    }
    return layers;
}


// ---------------------------------------------------------
// struct{ResNetEncoderImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor ResNetEncoderImpl::forward(torch::Tensor x){
    torch::Tensor feature, out;
    feature = this->first->forward(x);              // {C,224,224} ===> {F,56,56}
    feature = this->layer1->forward(feature);       // {F,56,56} ===> {F*E,56,56}
    feature = this->layer2->forward(feature);       // {F*E,56,56} ===> {2F*E,28,28}
    feature = this->layer3->forward(feature);       // {2F*E,28,28} ===> {4F*E,14,14}
    out = this->layer4->forward(feature);           // {4F*E,14,14} ===> {8F*E,7,7}
    return out;
}


// -----------------------------------------------
// struct{SimCLRImpl}(nn::Module) -> constructor
// -----------------------------------------------
SimCLRImpl::SimCLRImpl(po::variables_map &vm){

    size_t head;
    this->mt = std::mt19937(std::rand());
    this->jitter_prob = vm["jitter_prob"].as<double>();
    this->drop_prob = vm["drop_prob"].as<double>();
    this->jitter = vm["jitter"].as<float>();
    this->jitter_hue = vm["jitter_hue"].as<float>();
    
    // Convolution
    this->encoder = ResNetEncoder(vm["n_layers"].as<size_t>(), vm["nf"].as<size_t>(), head);
    register_module("Encoder", this->encoder);

    // Average Pooling
    this->avgpool = nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({1, 1}));
    register_module("Average Pooling", this->avgpool);

    // Multi Layer Perceptron
    this->mlp = nn::Sequential(
        nn::Linear(head, head),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Linear(head, vm["nz"].as<size_t>())
    );
    register_module("Multi Layer Perceptron", this->mlp);

}


// ---------------------------------------------------------------------
// struct{SimCLRImpl}(nn::Module) -> function{random_crop_resize_flip}
// ---------------------------------------------------------------------
torch::Tensor SimCLRImpl::random_crop_resize_flip(torch::Tensor image){

    long int ch, cw, sy, sx, N;
    torch::Tensor crop_h, crop_w, start_y, start_x, patch, resize, out, flip_x_mask, flip_y_mask;
    std::vector<torch::Tensor> resizes;

    N = image.size(0);
    crop_h = torch::randint(image.size(2) / 2, image.size(2) + 1, {N}).to(torch::kLong).to(image.device());
    crop_w = torch::randint(image.size(3) / 2, image.size(3) + 1, {N}).to(torch::kLong).to(image.device());
    start_y = (torch::rand({N}).to(image.device()) * (image.size(2) - crop_h.to(torch::kFloat)) + 0.5).to(torch::kLong);
    start_x = (torch::rand({N}).to(image.device()) * (image.size(3) - crop_w.to(torch::kFloat)) + 0.5).to(torch::kLong);

    for (long int n = 0; n < N; n++){
        ch = crop_h[n].item<long int>();
        cw = crop_w[n].item<long int>();
        sy = start_y[n].item<long int>();
        sx = start_x[n].item<long int>();
        patch = image.index({n, Slice(), Slice(sy, sy + ch), Slice(sx, sx + cw)}).unsqueeze(0);
        resize = F::interpolate(patch, F::InterpolateFuncOptions().size(std::vector<long int>{image.size(2), image.size(3)}).mode(torch::kBilinear).align_corners(false));
        resizes.push_back(resize);
    }
    out = torch::cat(resizes, /*dim=*/0);

    flip_y_mask = (torch::rand({N, 1, 1, 1}).to(image.device()) < 0.5).to(torch::kFloat);
    out = flip_y_mask * (out.flip({2})) + (1.0 - flip_y_mask) * out;

    flip_x_mask = (torch::rand({N, 1, 1, 1}).to(image.device()) < 0.5).to(torch::kFloat);
    out = flip_x_mask * (out.flip({3})) + (1.0 - flip_x_mask) * out;

    return out;

}


// --------------------------------------------------------
// struct{SimCLRImpl}(nn::Module) -> function{random_hsv}
// --------------------------------------------------------
torch::Tensor SimCLRImpl::random_hsv(torch::Tensor image, const float eps){

    torch::Tensor r, g, b, max_rgb, argmax_rgb, min_rgb, argmin_rgb, sub, h1, h2, h3, h, s, v, h_, c, x, zero, y, idx, out;
    std::vector<torch::Tensor> rgb;
    std::tuple<torch::Tensor, torch::Tensor> max_tuple, min_tuple;

    rgb = image.split(1, /*dim=*/1);
    r = rgb[0];
    g = rgb[1];
    b = rgb[2];

    max_tuple = image.max(/*dim=*/1, /*keepdim=*/true);
    max_rgb = std::get<0>(max_tuple);
    argmax_rgb = std::get<1>(max_tuple);

    min_tuple = image.min(/*dim=*/1, /*keepdim=*/true);
    min_rgb = std::get<0>(min_tuple);
    argmin_rgb = std::get<1>(min_tuple);

    sub = max_rgb - min_rgb + eps;
    h1 = 60.0 * (g - r) / sub + 60.0;
    h2 = 60.0 * (b - g) / sub + 180.0;
    h3 = 60.0 * (r - b) / sub + 300.0;

    h = torch::cat({h2, h3, h1}, /*dim=*/1).gather(/*dim=*/1, /*index=*/argmin_rgb);
    s = sub / (max_rgb + eps);
    v = max_rgb;

    // Augmentation
    h = h + (torch::rand({image.size(0), 1, 1, 1}).to(image.device()) - 0.5) * 360.0 * this->jitter_hue;
    s = s + (torch::rand({image.size(0), 1, 1, 1}).to(image.device()) - 0.5) * this->jitter;
    v = v + (torch::rand({image.size(0), 1, 1, 1}).to(image.device()) - 0.5) * this->jitter;

    // Clamp
    h = (h - torch::floor(h / 360.0) * 360.0);  // 0.0 <= h <= 360.0
    s = s.clamp(/*min=*/0.0, /*max=*/1.0);  // 0.0 <= s <= 1.0
    v = v.clamp(/*min=*/0.0, /*max=*/1.0);  // 0.0 <= v <= 1.0

    h_ = h / 60.0;
    c = s * v;
    x = c * (1.0 - torch::abs(torch::fmod(h_, 2.0) - 1.0));

    zero = torch::zeros_like(c);
    y = torch::stack({
        torch::cat({c, x, zero}, /*dim=*/1),
        torch::cat({x, c, zero}, /*dim=*/1),
        torch::cat({zero, c, x}, /*dim=*/1),
        torch::cat({zero, x, c}, /*dim=*/1),
        torch::cat({x, zero, c}, /*dim=*/1),
        torch::cat({c, zero, x}, /*dim=*/1)
    }, /*dim=*/0);

    idx = torch::repeat_interleave(torch::floor(h_), /*repeats=*/3, /*dim=*/1).unsqueeze(0).to(torch::kLong);
    idx = torch::where(idx == 6, torch::full_like(idx, 5), idx);
    out = y.gather(/*dim=*/0, /*index=*/idx) + (v - c);
    out = out.squeeze(0);

    return out;

}


// -------------------------------------------------------------
// struct{SimCLRImpl}(nn::Module) -> function{random_contrast}
// -------------------------------------------------------------
torch::Tensor SimCLRImpl::random_contrast(torch::Tensor image, const float eps){
    torch::Tensor mean, factor, out;
    mean = image.mean(/*dims=*/{1, 2, 3}, /*keepdim=*/true);
    factor = torch::pow(2.0, (torch::rand({image.size(0), 1, 1, 1}).to(image.device()) - 0.5) * this->jitter);
    out = (image - mean) * factor + mean;
    out = out.clamp(/*min=*/0.0, /*max=*/1.0);
    return out;
}


// -----------------------------------------------------
// struct{SimCLRImpl}(nn::Module) -> function{to_gray}
// -----------------------------------------------------
torch::Tensor SimCLRImpl::to_gray(torch::Tensor image){
    torch::Tensor r, g, b, gray, out;
    r = image.index({Slice(), 0, Slice(), Slice()});
    g = image.index({Slice(), 1, Slice(), Slice()});
    b = image.index({Slice(), 2, Slice(), Slice()});
    gray = 0.299 * r + 0.587 * g + 0.114 * b;
    out = gray.unsqueeze(1).expand_as(image).contiguous();
    return out;
}



// ---------------------------------------------------------
// struct{SimCLRImpl}(nn::Module) -> function{random_blur}
// ---------------------------------------------------------
torch::Tensor SimCLRImpl::random_blur(torch::Tensor image, const float eps){

    long int K, pad;
    torch::Tensor sigma, coords, coords2, sigma2, g1d, g2d, weight, image_flat, image_pad, out;

    sigma = F::softplus(torch::randn({image.size(0), 1}).to(image.device())) + eps;
    K = (2.0 * torch::ceil(3.0 * sigma) + 1.0).to(torch::kLong).max().item<long int>();
    K = std::min(K, 2 * std::min(image.size(2), image.size(3)) - 1);
    if (K % 2 == 0) K++;
    pad = K / 2;

    coords = torch::arange(-pad, pad + 1).to(image.device()).to(torch::kFloat);
    coords2 = coords.pow(2.0).unsqueeze(0);
    sigma2 = sigma.pow(2.0);
    g1d = torch::exp(-coords2 / (2.0 * sigma2));
    g1d = g1d / g1d.sum(1, /*keepdim=*/true);
    g2d = g1d.unsqueeze(2) * g1d.unsqueeze(1);
    g2d = g2d / g2d.sum({1, 2}, /*keepdim=*/true);

    weight = g2d.unsqueeze(1).repeat({1, image.size(1), 1, 1}).view({image.size(0) * image.size(1), 1, K, K}).contiguous();

    image_flat = image.view({1, image.size(0) * image.size(1), image.size(2), image.size(3)});
    image_pad = F::pad(image_flat, F::PadFuncOptions({pad, pad, pad, pad}).mode(torch::kReflect));
    out = F::conv2d(image_pad, weight, F::Conv2dFuncOptions().groups(image.size(0) * image.size(1)));
    out = out.view({image.size(0), image.size(1), image.size(2), image.size(3)}).contiguous();

    return out;
}


// ----------------------------------------------------------
// struct{SimCLRImpl}(nn::Module) -> function{augmentation}
// ----------------------------------------------------------
torch::Tensor SimCLRImpl::augmentation(torch::Tensor x){

    torch::Tensor jitter_mask, drop_mask;

    x = this->random_crop_resize_flip(x);

    jitter_mask = (torch::rand({x.size(0), 1, 1, 1}).to(x.device()) < this->jitter_prob).to(torch::kFloat);
    x = jitter_mask * this->random_contrast(this->random_hsv(x)) + (1.0 - jitter_mask) * x;

    drop_mask = (torch::rand({x.size(0), 1, 1, 1}).to(x.device()) < this->drop_prob).to(torch::kFloat);
    x = drop_mask * this->to_gray(x) + (1.0 - drop_mask) * x;

    x = this->random_blur(x);

    return x;

}


// -----------------------------------------------------
// struct{SimCLRImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
torch::Tensor SimCLRImpl::forward(torch::Tensor x){
    torch::Tensor feature;
    feature = this->encoder->forward(x);
    feature = this->avgpool->forward(feature);
    feature = feature.view({feature.size(0), -1});
    feature = this->mlp->forward(feature);
    return feature;
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
