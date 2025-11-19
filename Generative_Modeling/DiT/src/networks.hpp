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


// -------------------------------------------------
// struct{FeedForwardImpl}(nn::Module)
// -------------------------------------------------
struct FeedForwardImpl : nn::Module{
private:
    nn::Sequential mlp;
public:
    FeedForwardImpl(){}
    FeedForwardImpl(const size_t dim, const size_t hidden_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(FeedForward);


// -------------------------------------------------
// struct{AttentionImpl}(nn::Module)
// -------------------------------------------------
struct AttentionImpl : nn::Module{
private:
    size_t heads;
    float scale;
    nn::Softmax attend = nullptr;
    nn::Dropout dropout = nullptr;
    nn::Linear to_qkv = nullptr;
    nn::Sequential to_out;
public:
    AttentionImpl(){}
    AttentionImpl(const size_t dim, const size_t heads_, const size_t dim_head);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Attention);


// -------------------------------------------------
// struct{DiTBlockImpl}(nn::Module)
// -------------------------------------------------
struct DiTBlockImpl : nn::Module{
private:
    nn::LayerNorm norm1 = nullptr, norm2 = nullptr;
    Attention attn;
    FeedForward ff;
    nn::Sequential adaln;
public:
    DiTBlockImpl(){}
    DiTBlockImpl(const size_t dim, const size_t heads, const size_t dim_head, const size_t mlp_dim);
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);
};
TORCH_MODULE(DiTBlock);


// -------------------------------------------------
// struct{DiTBackboneImpl}(nn::Module)
// -------------------------------------------------
struct DiTBackboneImpl : nn::Module{
private:
    size_t z_size, patch_size, num_patches, dim, time_embed_dim;
    nn::Conv2d patch_embed = nullptr;
    torch::Tensor pos_embedding;
    nn::Dropout dropout = nullptr;
    nn::Sequential time_mlp;
    nn::ModuleList blocks;
    nn::LayerNorm final_norm = nullptr;
    nn::ConvTranspose2d head = nullptr;
    torch::Tensor timestep_embedding(torch::Tensor t, long int dim_);
public:
    DiTBackboneImpl(){}
    DiTBackboneImpl(po::variables_map &vm, std::vector<long int> z_shape);
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);
};
TORCH_MODULE(DiTBackbone);


// -------------------------------------------------
// struct{DiTImpl}(nn::Module)
// -------------------------------------------------
struct DiTImpl : nn::Module{
private:
    long int nc, size;
    size_t timesteps, timesteps_inf;
    float eta;
    torch::Tensor betas, alphas, alpha_bars;
    DiTBackbone model;
public:
    AutoEncoder ae;
    DiTImpl(){}
    DiTImpl(po::variables_map &vm, torch::Device device);
    std::tuple<torch::Tensor, torch::Tensor> add_noise(torch::Tensor z_0, torch::Tensor t);
    torch::Tensor denoise(torch::Tensor z_t, torch::Tensor t, torch::Tensor t_prev);
    torch::Tensor denoise_t(torch::Tensor z_t, torch::Tensor t);
    std::vector<long int> get_z_shape(torch::Device &device);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x_t, torch::Tensor t);
    torch::Tensor forward_z(torch::Tensor z);
};
TORCH_MODULE(DiT);


#endif
