#ifndef NETWORKS_HPP
#define NETWORKS_HPP

// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
using namespace torch;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);
void DownSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool LReLU, const bool bias=false);
void UpSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias=false);


// ----------------------------------------------------------
// struct{GAN_GeneratorImpl}(nn::Module)
// ----------------------------------------------------------
struct GAN_GeneratorImpl : nn::Module{
private:
    nn::Sequential model;
public:
    GAN_GeneratorImpl(){}
    GAN_GeneratorImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor z);
};

// ----------------------------------------------------------
// struct{GAN_DiscriminatorImpl}(nn::Module)
// ----------------------------------------------------------
struct GAN_DiscriminatorImpl : nn::Module{
private:
    nn::Sequential model;
public:
    GAN_DiscriminatorImpl(){}
    GAN_DiscriminatorImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(GAN_Generator);
TORCH_MODULE(GAN_Discriminator);


#endif