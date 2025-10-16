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


// -----------------------------------
// struct{DownSamplingImpl}(nn::Module)
// -----------------------------------
struct DownSamplingImpl : nn::Module{
private:
    nn::Sequential convs, mlp;
public:
    DownSamplingImpl(){}
    DownSamplingImpl(const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const size_t time_embed);
    torch::Tensor forward(torch::Tensor x, torch::Tensor v);
};
TORCH_MODULE(DownSampling);


// -----------------------------------
// struct{UpSamplingImpl}(nn::Module)
// -----------------------------------
struct UpSamplingImpl : nn::Module{
private:
    nn::Sequential convs, mlp;
public:
    UpSamplingImpl(){}
    UpSamplingImpl(const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const size_t time_embed);
    torch::Tensor forward(torch::Tensor x, torch::Tensor v);
};
TORCH_MODULE(UpSampling);


// -----------------------------------
// struct{UNetBlockImpl}(nn::Module)
// -----------------------------------
struct UNetBlockImpl : nn::Module{
private:
    bool outermost, innermost;
    nn::Sequential down, sub, up;
public:
    UNetBlockImpl(){}
    UNetBlockImpl(const std::pair<size_t, size_t> outside_nc, const size_t inside_nc, UNetBlockImpl &submodule, const bool outermost_, const bool innermost_, const size_t time_embed);
    torch::Tensor forward(torch::Tensor x, torch::Tensor v);
};
TORCH_MODULE(UNetBlock);


// ------------------------------
// struct{UNetImpl}(nn::Module)
// ------------------------------
struct UNetImpl : nn::Module{
private:
    size_t time_embed;
    nn::Sequential model;
    torch::Tensor pos_encoding(torch::Tensor t, long int dim);
public:
    UNetImpl(){}
    UNetImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);
};
TORCH_MODULE(UNet);


// ------------------------------
// struct{RFImpl}(nn::Module)
// ------------------------------
struct RFImpl : nn::Module{
private:
    size_t timesteps;
    UNet model;
    torch::Tensor sampling(torch::Tensor &mean, torch::Tensor &var);
public:
    RFImpl(){}
    RFImpl(po::variables_map &vm, torch::Device device);
    std::tuple<torch::Tensor, torch::Tensor> add_noise(torch::Tensor x, torch::Tensor t);
    torch::Tensor ode_step(torch::Tensor x_t, torch::Tensor t, torch::Tensor dt);
    torch::Tensor forward(torch::Tensor x_t, torch::Tensor t);
    torch::Tensor forward_z(torch::Tensor z, size_t steps);
    torch::Tensor forward_z(torch::Tensor z);
};
TORCH_MODULE(RF);


#endif
