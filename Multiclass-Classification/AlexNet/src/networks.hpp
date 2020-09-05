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


// -------------------------------------------------
// struct{MC_AlexNetImpl}(nn::Module)
// -------------------------------------------------
struct MC_AlexNetImpl : nn::Module{
private:
    nn::Sequential features, avgpool, classifier;
public:
    MC_AlexNetImpl(){}
    MC_AlexNetImpl(po::variables_map &vm);
    void init();
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MC_AlexNet);


#endif