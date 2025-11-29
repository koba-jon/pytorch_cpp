#ifndef NETWORKS_HPP
#define NETWORKS_HPP

// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;


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
    template <typename T> nn::Sequential make_layers(T &block, const size_t expansion, const size_t planes, const size_t num_blocks, const size_t stride);
    torch::Tensor embedding_concat(torch::Tensor x, torch::Tensor y);
    torch::Tensor forward(torch::Tensor x);
    size_t get_total_dim(torch::Device device);
};
TORCH_MODULE(MC_ResNet);

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
    BasicBlockImpl(){}
    BasicBlockImpl(const size_t expansion, const size_t inplanes, const size_t planes, const size_t stride);
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
    BottleneckImpl(){}
    BottleneckImpl(const size_t expansion, const size_t inplanes, const size_t planes, const size_t stride);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Bottleneck);

// -------------------------------------------------
// struct{SelectIndexImpl}(nn::Module)
// -------------------------------------------------
struct SelectIndexImpl : nn::Module{
private:
    torch::Tensor idx;
public:
    SelectIndexImpl(){}
    SelectIndexImpl(const size_t total_dim, const size_t select_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SelectIndex);

#endif