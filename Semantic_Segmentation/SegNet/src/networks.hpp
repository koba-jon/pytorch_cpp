#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <tuple>
#include <vector>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);


// --------------------------------
// struct{SegNetImpl}(nn::Module)
// --------------------------------
struct SegNetImpl : nn::Module{
private:
    size_t num_downs;
    nn::ModuleList encoder, decoder;
    nn::Sequential classifier;
public:
    SegNetImpl(){}
    SegNetImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};

// -------------------------------------
// struct{DownSamplingImpl}(nn::Module)
// -------------------------------------
struct DownSamplingImpl : nn::Module{
private:
    nn::Sequential features;
    nn::MaxPool2d pool{nullptr};
public:
    DownSamplingImpl(){}
    DownSamplingImpl(const size_t in_nc, const size_t out_nc, const size_t n_layers=2, const bool use_dropout=false);
    std::tuple<torch::Tensor, torch::Tensor, std::vector<long int>> forward(torch::Tensor x);
};

// -----------------------------------
// struct{UpSamplingImpl}(nn::Module)
// -----------------------------------
struct UpSamplingImpl : nn::Module{
private:
    nn::Sequential features;
    nn::MaxUnpool2d unpool{nullptr};
public:
    UpSamplingImpl(){}
    UpSamplingImpl(const size_t in_nc, const size_t out_nc, const size_t n_layers=2, const bool use_dropout=false);
    torch::Tensor forward(torch::Tensor x, torch::Tensor indices, const std::vector<long int> unpool_sizes);
};


TORCH_MODULE(SegNet);
TORCH_MODULE(DownSampling);
TORCH_MODULE(UpSampling);


#endif