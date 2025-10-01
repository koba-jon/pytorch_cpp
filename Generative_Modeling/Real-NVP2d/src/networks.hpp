#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <vector>
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
// struct{RealNVP_CouplingLayerImpl}(nn::Module)
// -------------------------------------------------
struct RealNVP_CouplingLayerImpl : nn::Module{
private:
    long int dim, split, h_dim;
    nn::Sequential s_net, t_net;
public:
    RealNVP_CouplingLayerImpl(){}
    RealNVP_CouplingLayerImpl(long int dim_, long int h_dim_);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    torch::Tensor inverse(torch::Tensor y);
};
TORCH_MODULE(RealNVP_CouplingLayer);


// -------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module)
// -------------------------------------------------
struct NormalizingFlowImpl : nn::Module{
private:
    nn::ModuleList model;
public:
    NormalizingFlowImpl(){}
    NormalizingFlowImpl(po::variables_map &vm); 
    torch::Tensor swap(torch::Tensor z, long int split);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor z);
    torch::Tensor inverse(torch::Tensor z);
};
TORCH_MODULE(NormalizingFlow);


#endif