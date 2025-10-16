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
// struct{WNConv2dImpl}(nn::Module)
// -------------------------------------------------
struct WNConv2dImpl : nn::Module{
private:
    nn::Conv2d conv = nullptr;
public:
    torch::Tensor v, g;
    WNConv2dImpl(){}
    WNConv2dImpl(long int in_nc, long int out_nc, std::vector<long int> kernel, long int stride=1, std::vector<long int> padding={0, 0}, bool bias=true);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(WNConv2d);


// -------------------------------------------------
// struct{WNLinearImpl}(nn::Module)
// -------------------------------------------------
struct WNLinearImpl : nn::Module{
private:
    nn::Linear linear = nullptr;
public:
    torch::Tensor v, g;
    WNLinearImpl(){}
    WNLinearImpl(long int in_nc, long int out_nc);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(WNLinear);


// -------------------------------------------------
// struct{CausalConv2dImpl}(nn::Module)
// -------------------------------------------------
struct CausalConv2dImpl : nn::Module{
private:
    long int causal;
    nn::ZeroPad2d zero_pad = nullptr;
    WNConv2d conv;
public:
    CausalConv2dImpl(){}
    CausalConv2dImpl(long int in_nc, long int out_nc, std::vector<long int> kernel, long int stride=1, std::string padding="downright");
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CausalConv2d);


// -------------------------------------------------
// struct{GatedResBlockImpl}(nn::Module)
// -------------------------------------------------
struct GatedResBlockImpl : nn::Module{
private:
    nn::Sequential conv1, conv2, aux_conv;
    nn::ELU activation = nullptr;
    nn::Dropout dropout = nullptr;
    nn::GLU gate = nullptr;
public:
    GatedResBlockImpl(){}
    GatedResBlockImpl(long int in_nc, long int nc, std::vector<long int> kernel, std::string conv="wnconv2d", std::string act="ELU", float droprate=0.1, long int aux_nc=0);
    torch::Tensor forward(torch::Tensor x, torch::Tensor aux=torch::Tensor());
};
TORCH_MODULE(GatedResBlock);


// -------------------------------------------------
// struct{CausalAttentionImpl}(nn::Module)
// -------------------------------------------------
struct CausalAttentionImpl : nn::Module{
private:
    long int dim_head, n_head;
    WNLinear query_linear, key_linear, value_linear;
    nn::Dropout dropout = nullptr;
public:
    CausalAttentionImpl(){}
    CausalAttentionImpl(long int query_nc, long int key_nc, long int nc, long int n_head_, float droprate);
    torch::Tensor forward(torch::Tensor query, torch::Tensor key);
};
TORCH_MODULE(CausalAttention);


// -------------------------------------------------
// struct{PixelBlockImpl}(nn::Module)
// -------------------------------------------------
struct PixelBlockImpl : nn::Module{
private:
    long int res_block, attention;
    nn::ModuleList resblocks;
    GatedResBlock key_resblock, query_resblock, out_resblock;
    CausalAttention causal_attention;
    WNConv2d out_conv;
public:
    PixelBlockImpl(){}
    PixelBlockImpl(long int in_nc, long int nc, long int kernel, long int res_block_, bool attention_, float droprate);
    torch::Tensor forward(torch::Tensor x, torch::Tensor background);
};
TORCH_MODULE(PixelBlock);


// -------------------------------------------------
// struct{PixelSnailImpl}(nn::Module)
// -------------------------------------------------
struct PixelSNAILImpl : nn::Module{
private:
    long int block, level;
    CausalConv2d horizontal, vertical;
    nn::ModuleList blocks;
    nn::Sequential out_module;
public:
    PixelSNAILImpl(){}
    PixelSNAILImpl(po::variables_map &vm);
    torch::Tensor sampling(const std::vector<long int> x_shape, torch::Device device);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PixelSNAIL);




#endif