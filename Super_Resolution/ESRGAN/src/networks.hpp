#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <vector>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);
nn::Sequential make_layers(const size_t nc, const std::vector<long int> cfg, const bool BN);


// -------------------------------------------------
// struct{MC_VGGNetImpl}(nn::Module)
// -------------------------------------------------
struct MC_VGGNetImpl : nn::Module{
private:
    nn::Sequential features, avgpool, classifier;
public:
    MC_VGGNetImpl();
    void init();
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MC_VGGNet);

// ----------------------------------------------------------
// struct{DenseBlockImpl}(nn::Module)
// ----------------------------------------------------------
struct DenseBlockImpl : nn::Module{
private:
    std::vector<nn::Conv2d> convs;
    size_t growth;
public:
    DenseBlockImpl(){}
    DenseBlockImpl(const size_t in_nc, const size_t growth_);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DenseBlock);

// ----------------------------------------------------------
// struct{RRDBImpl}(nn::Module)
// ----------------------------------------------------------
struct RRDBImpl : nn::Module{
private:
    nn::Sequential blocks;
public:
    RRDBImpl(){}
    RRDBImpl(const size_t in_nc, const size_t growth);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(RRDB);

// ----------------------------------------------------------
// struct{UpsampleBlockImpl}(nn::Module)
// ----------------------------------------------------------
struct UpsampleBlockImpl : nn::Module{
private:
    nn::Sequential block;
public:
    UpsampleBlockImpl(){}
    UpsampleBlockImpl(const size_t in_nc);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(UpsampleBlock);

// ----------------------------------------------------------
// struct{ESRGAN_GeneratorImpl}(nn::Module)
// ----------------------------------------------------------
struct ESRGAN_GeneratorImpl : nn::Module{
private:
    nn::Sequential head, body, tail;
public:
    ESRGAN_GeneratorImpl(){}
    ESRGAN_GeneratorImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ESRGAN_Generator);

// ----------------------------------------------------------
// struct{ConvBlockImpl}(nn::Module)
// ----------------------------------------------------------
struct ConvBlockImpl : nn::Module{
private:
    nn::Sequential model;
public:
    ConvBlockImpl(){}
    ConvBlockImpl(const size_t in_nc, const size_t out_nc, const int stride, const bool BN=true);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ConvBlock);

// ----------------------------------------------------------
// struct{ESRGAN_DiscriminatorImpl}(nn::Module)
// ----------------------------------------------------------
struct ESRGAN_DiscriminatorImpl : nn::Module{
private:
    nn::Sequential model;
public:
    ESRGAN_DiscriminatorImpl(){}
    ESRGAN_DiscriminatorImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ESRGAN_Discriminator);


#endif
