#ifndef NETWORKS_HPP
#define NETWORKS_HPP

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
    nn::LayerNorm norm = nullptr;
    nn::Softmax attend = nullptr;
    nn::Dropout dropout = nullptr;
    nn::Linear to_qkv = nullptr;
    nn::Sequential to_out;
public:
    AttentionImpl(){}
    AttentionImpl(const size_t dim, const size_t heads_);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Attention);


// -------------------------------------------------
// struct{TransformerImpl}(nn::Module)
// -------------------------------------------------
struct TransformerImpl : nn::Module{
private:
    nn::ModuleList layer;
public:
    TransformerImpl(){}
    TransformerImpl(size_t dim, size_t heads, float mlp_ratio);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Transformer);


// -------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module)
// -------------------------------------------------
struct MaskedAutoEncoderImpl : nn::Module{
private:
    size_t split, image_size, patch_size, enc_dim, dec_dim, keep_num;
    nn::Conv2d conv = nullptr;
    torch::Tensor class_token, mask_token, enc_pos_encoding, dec_pos_encoding;
    nn::LayerNorm enc_norm = nullptr;
    nn::LayerNorm dec_norm = nullptr;
    nn::Sequential enc_block, dec_block;
    nn::Linear latent = nullptr;
    nn::Linear pred = nullptr;
    torch::Tensor get_1d_sincos_pos_embed_from_grid(size_t dim, torch::Tensor pos);
    torch::Tensor get_2d_sincos_pos_embed_for_grid(size_t dim, torch::Tensor grid);
    torch::Tensor get_2d_sincos_pos_embed(size_t dim, size_t grid_size);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> random_masking(torch::Tensor x);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> encoder(torch::Tensor x);
    torch::Tensor decoder(torch::Tensor z, torch::Tensor ids_restore);
public:
    MaskedAutoEncoderImpl(){}
    MaskedAutoEncoderImpl(po::variables_map &vm);
    void init();
    torch::Tensor patchify(torch::Tensor x);
    torch::Tensor unpatchify(torch::Tensor x);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(MaskedAutoEncoder);


#endif