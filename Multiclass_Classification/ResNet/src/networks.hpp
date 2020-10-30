#ifndef NETWORKS_HPP
#define NETWORKS_HPP

// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);


// -------------------------------------------------
// struct{MC_ResNetImpl}(nn::Module)
// -------------------------------------------------
struct MC_ResNetImpl : nn::Module{
private:
    size_t inplanes;
    nn::Sequential first;
    nn::Sequential layer1, layer2, layer3, layer4;
    nn::Sequential avgpool, classifier;
public:
    MC_ResNetImpl(){}
    MC_ResNetImpl(po::variables_map &vm);
    void init();
    template <typename T> nn::Sequential make_layers(T &block, const size_t planes, const size_t num_blocks, const size_t stride);
    torch::Tensor forward(torch::Tensor x);
};

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

TORCH_MODULE(MC_ResNet);
TORCH_MODULE(BasicBlock);
TORCH_MODULE(Bottleneck);


#endif
