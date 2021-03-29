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
void LinearLayer(nn::Sequential &sq, const size_t in_dim, const size_t out_dim, const bool ReLU);


// -------------------------------------------------
// struct{AutoEncoder1dImpl}(nn::Module)
// -------------------------------------------------
struct AutoEncoder1dImpl : nn::Module{
private:
    nn::Sequential encoder, decoder;
public:
    AutoEncoder1dImpl(){}
    AutoEncoder1dImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(AutoEncoder1d);


#endif