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
// struct{ResidualBlockImpl}(nn::Module)
// ----------------------------------------------------------
struct ResidualBlockImpl : nn::Module{
private:
    nn::Sequential block;
public:
    ResidualBlockImpl(){}
    ResidualBlockImpl(const size_t nc);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResidualBlock);

// ----------------------------------------------------------
// struct{UpsampleBlockImpl}(nn::Module)
// ----------------------------------------------------------
struct UpsampleBlockImpl : nn::Module{
private:
    nn::Sequential block;
public:
    UpsampleBlockImpl(){}
    UpsampleBlockImpl(const size_t in_nc, const size_t scale_factor);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(UpsampleBlock);

// ----------------------------------------------------------
// struct{SRGAN_GeneratorImpl}(nn::Module)
// ----------------------------------------------------------
struct SRGAN_GeneratorImpl : nn::Module{
private:
    nn::Sequential head, body, tail;
public:
    SRGAN_GeneratorImpl(){}
    SRGAN_GeneratorImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SRGAN_Generator);

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
// struct{SRGAN_DiscriminatorImpl}(nn::Module)
// ----------------------------------------------------------
struct SRGAN_DiscriminatorImpl : nn::Module{
private:
    nn::Sequential model;
public:
    SRGAN_DiscriminatorImpl(){}
    SRGAN_DiscriminatorImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SRGAN_Discriminator);


#endif
