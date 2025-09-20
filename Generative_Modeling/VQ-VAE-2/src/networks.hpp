#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <string>
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
    nn::Sequential conv1, conv2, aux_conv, cond;
    nn::ELU activation = nullptr;
    nn::Dropout dropout = nullptr;
    nn::GLU gate = nullptr;
public:
    GatedResBlockImpl(){}
    GatedResBlockImpl(long int in_nc, long int nc, std::vector<long int> kernel, std::string conv="wnconv2d", std::string act="ELU", float droprate=0.1, long int aux_nc=0, long int cond_dim=0);
    torch::Tensor forward(torch::Tensor x, torch::Tensor aux=torch::Tensor(), torch::Tensor condition=torch::Tensor());
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
    GatedResBlock key_resblock, query_resblock, causal_attention, out_resblock;
    WNConv2d out_conv;
public:
    PixelBlockImpl(){}
    PixelBlockImpl(long int in_nc, long int nc, long int kernel, long int res_block_, bool attention_, float droprate, long int cond_dim);
    torch::Tensor forward(torch::Tensor x, torch::Tensor background, torch::Tensor condition);
};
TORCH_MODULE(PixelBlock);


// -------------------------------------------------
// struct{CondResNetImpl}(nn::Module)
// -------------------------------------------------
struct CondResNetImpl : nn::Module{
private:
    long int res_block;
    nn::ModuleList blocks;
public:
    CondResNetImpl(){}
    CondResNetImpl(long int in_nc, long int nc, long int kernel, long int res_block_);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CondResNet);


// -------------------------------------------------
// struct{PixelSnailImpl}(nn::Module)
// -------------------------------------------------
struct PixelSnailImpl : nn::Module{
private:
    long int K, block;
    CausalConv2d horizontal, vertical;
    nn::ModuleList blocks;
    CondResNet cond_resnet;
    nn::Sequential out_module;
public:
    PixelSnailImpl(){}
    PixelSnailImpl(long int K_, long int nc, long int kernel, long int block_, long int res_block, long int res_nc, bool attention=true, float droprate=0.1, long int cond_res_block=0, long int cond_res_nc=0, long int cond_res_kernel=3, long int out_res_block=0);
    torch::Tensor forward(torch::Tensor x, torch::Tensor condition=torch::Tensor());
};
TORCH_MODULE(PixelSnail);


// -------------------------------------------------
// struct{VectorQuantizerImpl}(nn::Module)
// -------------------------------------------------
struct VectorQuantizerImpl : nn::Module{
private:
    float decay, eps;
public:
    torch::Tensor e, cluster_size, e_avg;
    VectorQuantizerImpl(){}
    VectorQuantizerImpl(const size_t D, const size_t K, const float decay_=0.99, const float eps_=1e-5);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor z_e);
};
TORCH_MODULE(VectorQuantizer);


// -------------------------------------------------
// struct{ResidualLayerImpl}(nn::Module)
// -------------------------------------------------
struct ResidualLayerImpl : nn::Module{
private:
    nn::Sequential model;
public:
    ResidualLayerImpl(){}
    ResidualLayerImpl(const size_t dim, const size_t h_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ResidualLayer);


// -------------------------------------------------
// struct{EncoderImpl}(nn::Module)
// -------------------------------------------------
struct EncoderImpl : nn::Module{
private:
    nn::Sequential model;
public:
    EncoderImpl(){}
    EncoderImpl(long int in_nc, long int out_nc, long int n_res_block, long int n_res_nc, long int stride);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Encoder);


// -------------------------------------------------
// struct{DecoderImpl}(nn::Module)
// -------------------------------------------------
struct DecoderImpl : nn::Module{
private:
    nn::Sequential model;
public:
    DecoderImpl(){}
    DecoderImpl(long int in_nc, long int out_nc, long int mid_nc, long int res_block, long int res_nc, long int stride);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Decoder);


// -------------------------------------------------
// struct{VQVAE2Impl}(nn::Module)
// -------------------------------------------------
struct VQVAE2Impl : nn::Module{
private:
    Encoder enc_b, enc_t;
    nn::Conv2d quantize_conv_t = nullptr;
    VectorQuantizer quantize_t;
    Decoder dec_t;
    nn::Conv2d quantize_conv_b = nullptr;
    VectorQuantizer quantize_b;
    nn::ConvTranspose2d upsample_t = nullptr;
    Decoder dec;
public:
    VQVAE2Impl(){}
    VQVAE2Impl(po::variables_map &vm);
    torch::Tensor sampling(const std::tuple<std::vector<long int>, std::vector<long int>> idx_shape, PixelSnail pixelsnail_t, PixelSnail pixelsnail_b, torch::Device device);
    torch::Tensor synthesis(torch::Tensor x, torch::Tensor y, const float alpha);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> encode(torch::Tensor x);
    torch::Tensor decode(torch::Tensor quant_t, torch::Tensor quant_b);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    std::tuple<torch::Tensor, torch::Tensor> forward_idx(torch::Tensor x);
    std::tuple<std::vector<long int>, std::vector<long int>> get_idx_shape(const std::vector<long int> x_shape, torch::Device &device);
};
TORCH_MODULE(VQVAE2);


#endif