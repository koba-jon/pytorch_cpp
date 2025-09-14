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
    AttentionImpl(const size_t dim, const size_t heads_, const size_t dim_head);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Attention);


// -------------------------------------------------
// struct{TransformerImpl}(nn::Module)
// -------------------------------------------------
struct TransformerImpl : nn::Module{
private:
    size_t depth;
    nn::LayerNorm norm = nullptr;
    nn::ModuleList layers;
public:
    TransformerImpl(){}
    TransformerImpl(size_t dim, size_t depth_, size_t heads, size_t dim_head, size_t mlp_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Transformer);


// -------------------------------------------------
// struct{ViTImpl}(nn::Module)
// -------------------------------------------------
struct ViTImpl : nn::Module{
private:
    size_t image_size, dim;
    nn::Conv2d conv = nullptr;
    nn::LayerNorm norm = nullptr;
    torch::Tensor class_token, pos_encoding;
    nn::Dropout dropout = nullptr;
    Transformer transformer;
    nn::Linear mlp_head = nullptr;
public:
    ViTImpl(){}
    ViTImpl(po::variables_map &vm);
    void init();
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ViT);


#endif
