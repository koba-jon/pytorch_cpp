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
// struct{RadialFlowImpl}(nn::Module)
// -------------------------------------------------
struct RadialFlowImpl : nn::Module{
private:
    torch::Tensor c, a, b;
public:
    RadialFlowImpl(){}
    RadialFlowImpl(long int dim);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor z);
    torch::Tensor inverse(torch::Tensor z);
    void pretty_print(std::ostream& stream) const override;
};
TORCH_MODULE(RadialFlow);


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