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
// struct{MaskedConv2dImpl}(nn::Module)
// -------------------------------------------------
struct MaskedConv2dImpl : nn::Module{
private:
    char mask_type;
    long int padding;
    torch::Tensor weight, mask;
public:
    MaskedConv2dImpl(){}
    MaskedConv2dImpl(char mask_type_, long int in_nc, long int out_nc, long int kernel);
    torch::Tensor forward(torch::Tensor x);
    void pretty_print(std::ostream& stream) const override;
};
TORCH_MODULE(MaskedConv2d);


// -------------------------------------------------
// struct{MaskedConv2dBlockImpl}(nn::Module)
// -------------------------------------------------
struct MaskedConv2dBlockImpl : nn::Module{
private:
    bool residual;
    nn::Sequential model;
public:
    MaskedConv2dBlockImpl(){}
    MaskedConv2dBlockImpl(char mask_type, long int dim, bool residual_);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MaskedConv2dBlock);


// -------------------------------------------------
// struct{PixelCNNImpl}(nn::Module)
// -------------------------------------------------
struct PixelCNNImpl : nn::Module{
private:
    long int dim, level;
    nn::Sequential layers;
    nn::Conv2d output_conv = nullptr;
public:
    PixelCNNImpl(){}
    PixelCNNImpl(po::variables_map &vm);
    torch::Tensor sampling(const std::vector<long int> x_shape, torch::Device device);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PixelCNN);



#endif