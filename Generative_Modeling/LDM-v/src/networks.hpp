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
// struct{AutoEncoderImpl}(nn::Module)
// -----------------------------------
struct AutoEncoderImpl : nn::Module{
private:
    nn::Sequential encoder, decoder;
public:
    AutoEncoderImpl(){}
    AutoEncoderImpl(po::variables_map &vm);
    torch::Tensor encode(torch::Tensor x);
    torch::Tensor decode(torch::Tensor z);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(AutoEncoder);


// -----------------------------------
// struct{ResBlockImpl}(nn::Module)
// -----------------------------------
struct ResBlockImpl : nn::Module{
private:
    nn::Sequential conv1, conv2, time_emb_proj;
    nn::Conv2d skip_conv = nullptr;
    bool use_skip;
public:
    ResBlockImpl(){}
    ResBlockImpl(const size_t in_nc, const size_t out_nc, const size_t time_embed);
    torch::Tensor forward(torch::Tensor x, torch::Tensor v);
};
TORCH_MODULE(ResBlock);


// -----------------------------------
// struct{AttentionBlockImpl}(nn::Module)
// -----------------------------------
struct AttentionBlockImpl : nn::Module{
private:
    size_t heads;
    nn::GroupNorm norm = nullptr;
    nn::Conv2d to_qkv = nullptr;
    nn::Conv2d proj_out = nullptr;
public:
    AttentionBlockImpl(){}
    AttentionBlockImpl(const size_t nc, const size_t heads_);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(AttentionBlock);


// ------------------------------
// struct{UNetImpl}(nn::Module)
// ------------------------------
struct UNetImpl : nn::Module{
private:
    long int max_level, max_block;
    size_t time_embed;
    nn::Sequential time_mlp;
    nn::Conv2d input_conv = nullptr;
    std::vector<ResBlock> down_resblocks, up_resblocks;
    std::vector<AttentionBlock> down_attentions, up_attentions;
    std::vector<nn::Conv2d> downsample_layers;
    ResBlock mid_block1, mid_block2;
    AttentionBlock mid_attn;
    std::vector<nn::ConvTranspose2d> upsample_layers;
    nn::Sequential out_conv;
    torch::Tensor timestep_embedding(torch::Tensor t, long int dim);
public:
    UNetImpl(){}
    UNetImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);
};
TORCH_MODULE(UNet);


// ------------------------------
// struct{LDMImpl}(nn::Module)
// ------------------------------
struct LDMImpl : nn::Module{
private:
    long int nc, size;
    size_t timesteps, timesteps_inf;
    float eta;
    torch::Tensor betas, alphas, alpha_bars;
    UNet model;
    torch::Tensor sampling(torch::Tensor &mean, torch::Tensor &var);
public:
    AutoEncoder ae;
    LDMImpl(){}
    LDMImpl(po::variables_map &vm, torch::Device device);
    std::tuple<torch::Tensor, torch::Tensor> add_noise(torch::Tensor z_0, torch::Tensor t);
    torch::Tensor denoise(torch::Tensor z_t, torch::Tensor t, torch::Tensor t_prev);
    torch::Tensor denoise_t(torch::Tensor z_t, torch::Tensor t);
    std::vector<long int> get_z_shape(torch::Device &device);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x_0, torch::Tensor t);
    torch::Tensor forward_z(torch::Tensor z);
};
TORCH_MODULE(LDM);


#endif
