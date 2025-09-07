#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <utility>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);
void DownSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool LReLU, const bool bias=false);
void UpSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias=false);


// -------------------------------------------------
// struct{ResNetBlockImpl}(nn::Module)
// -------------------------------------------------
struct ResNetBlockImpl : nn::Module{
private:
    nn::Sequential model;
public:
    ResNetBlockImpl(){}    
    ResNetBlockImpl(const size_t dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResNetBlock);


// -------------------------------------------------
// struct{ResNet_GeneratorImpl}(nn::Module)
// -------------------------------------------------
struct ResNet_GeneratorImpl : nn::Module{
private:
    nn::Sequential model;
public:
    ResNet_GeneratorImpl(){}
    ResNet_GeneratorImpl(size_t input_nc, size_t output_nc, size_t ngf, size_t n_blocks);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResNet_Generator);


// -------------------------------------------------
// struct{PatchGAN_DiscriminatorImpl}(nn::Module)
// -------------------------------------------------
struct PatchGAN_DiscriminatorImpl : nn::Module{
private:
    nn::Sequential model;
public:
    PatchGAN_DiscriminatorImpl(){}
    PatchGAN_DiscriminatorImpl(size_t input_nc, size_t ndf, size_t n_layers);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PatchGAN_Discriminator);


#endif