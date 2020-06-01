#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <utility>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
using namespace torch;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);
void DownSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias=false);
void UpSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias=false);


// -------------------------------------------------
// struct{UNet_GeneratorImpl}(nn::Module)
// -------------------------------------------------
struct UNet_GeneratorImpl : nn::Module{
private:
    nn::Sequential model;
public:
    UNet_GeneratorImpl(){}
    UNet_GeneratorImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};

// -------------------------------------------------
// struct{UNetBlockImpl}(nn::Module)
// -------------------------------------------------
struct UNetBlockImpl : nn::Module{
private:
    bool outermost;
    nn::Sequential model;
public:
    UNetBlockImpl(){}    
    UNetBlockImpl(const std::pair<size_t, size_t> outside_nc, const size_t inside_nc, UNetBlockImpl &submodule, bool outermost_=false, bool innermost=false, bool use_dropout=false);
    torch::Tensor forward(torch::Tensor x);
};

// -------------------------------------------------
// struct{PatchGAN_DiscriminatorImpl}(nn::Module)
// -------------------------------------------------
struct PatchGAN_DiscriminatorImpl : nn::Module{
private:
    nn::Sequential model;
public:
    PatchGAN_DiscriminatorImpl(){}
    PatchGAN_DiscriminatorImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(UNet_Generator);
TORCH_MODULE(UNetBlock);
TORCH_MODULE(PatchGAN_Discriminator);


#endif