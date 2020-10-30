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
    MC_VGGNetImpl(){}
    MC_VGGNetImpl(po::variables_map &vm);
    void init();
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MC_VGGNet);


#endif
