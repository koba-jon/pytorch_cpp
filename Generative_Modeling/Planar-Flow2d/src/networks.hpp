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
// struct{PlanarFlowImpl}(nn::Module)
// -------------------------------------------------
struct PlanarFlowImpl : nn::Module{
private:
    torch::Tensor u, w, b;
public:
    PlanarFlowImpl(){}
    PlanarFlowImpl(long int dim);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor z);
    torch::Tensor inverse(torch::Tensor z);
    void pretty_print(std::ostream& stream) const override;
};
TORCH_MODULE(PlanarFlow);


// -------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module)
// -------------------------------------------------
struct NormalizingFlowImpl : nn::Module{
private:
    nn::ModuleList model;
public:
    NormalizingFlowImpl(){}
    NormalizingFlowImpl(po::variables_map &vm);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor z);
    torch::Tensor inverse(torch::Tensor z);
};
TORCH_MODULE(NormalizingFlow);


#endif