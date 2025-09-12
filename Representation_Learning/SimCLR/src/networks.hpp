#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <random>
#include <tuple>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);


// -------------------------------------------------
// struct{BasicBlockImpl}(nn::Module)
// -------------------------------------------------
struct BasicBlockImpl : nn::Module{
private:
    bool down;
    nn::Sequential layerA, layerB;
    nn::Sequential downsample;
    nn::Sequential last;
public:
    static const size_t expansion = 1;
    BasicBlockImpl(){}
    BasicBlockImpl(const size_t inplanes, const size_t planes, const size_t stride);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(BasicBlock);

// -------------------------------------------------
// struct{BottleneckImpl}(nn::Module)
// -------------------------------------------------
struct BottleneckImpl : nn::Module{
private:
    bool down;
    nn::Sequential layerA, layerB, layerC;
    nn::Sequential downsample;
    nn::Sequential last;
public:
    static const size_t expansion = 4;
    BottleneckImpl(){}
    BottleneckImpl(const size_t inplanes, const size_t planes, const size_t stride);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Bottleneck);

// ---------------------------------------
// struct{ResNetEncoderImpl}(nn::Module)
// ---------------------------------------
struct ResNetEncoderImpl : nn::Module{
private:
    size_t inplanes;
    nn::Sequential first;
    nn::Sequential layer1, layer2, layer3, layer4;
    nn::Sequential avgpool, classifier;
public:
    ResNetEncoderImpl(){}
    ResNetEncoderImpl(size_t n_layers, size_t nf, size_t &head);
    template <typename T> nn::Sequential make_layers(T &block, const size_t planes, const size_t num_blocks, const size_t stride);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResNetEncoder);

// -------------------------------------------------
// struct{SimCLRImpl}(nn::Module)
// -------------------------------------------------
struct SimCLRImpl : nn::Module{
private:
    ResNetEncoder encoder;
    nn::AdaptiveAvgPool2d avgpool = nullptr;
    nn::Sequential mlp;
    std::mt19937 mt;
    double jitter_prob, drop_prob;
    float jitter, jitter_hue;
public:
    SimCLRImpl(){}
    SimCLRImpl(po::variables_map &vm);
    torch::Tensor random_crop_resize_flip(torch::Tensor image);
    torch::Tensor random_hsv(torch::Tensor image, const float eps=1e-9);
    torch::Tensor random_contrast(torch::Tensor image, const float eps=1e-9);
    torch::Tensor to_gray(torch::Tensor image);
    torch::Tensor random_blur(torch::Tensor image, const float eps=1e-9);
    torch::Tensor augmentation(torch::Tensor x);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SimCLR);


#endif