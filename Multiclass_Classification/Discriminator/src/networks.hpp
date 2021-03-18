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
void Convolution(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const size_t ksize, const size_t stride, const size_t pad, const bool BN, const bool LReLU, const bool bias=false);


// -------------------------------------------------
// struct{MC_DiscriminatorImpl}(nn::Module)
// -------------------------------------------------
struct MC_DiscriminatorImpl : nn::Module{
private:
    nn::Sequential features, avgpool, classifier;
public:
    MC_DiscriminatorImpl(){}
    MC_DiscriminatorImpl(po::variables_map &vm);
    void init();
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MC_Discriminator);


#endif
