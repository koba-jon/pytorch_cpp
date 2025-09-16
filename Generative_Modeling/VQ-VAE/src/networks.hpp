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
    long int padding;
    nn::Conv2d conv = nullptr;
    torch::Tensor weight, mask;
public:
    MaskedConv2dImpl(){}
    MaskedConv2dImpl(char mask_type, long int in_nc, long int out_nc, long int kernel);
    torch::Tensor forward(torch::Tensor x);
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
    MaskedConv2dBlockImpl(char mask_type, long int dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MaskedConv2dBlock);


// -------------------------------------------------
// struct{GatedPixelCNNImpl}(nn::Module)
// -------------------------------------------------
struct GatedPixelCNNImpl : nn::Module{
private:
    long int dim;
    nn::Embedding token_emb = nullptr;
    nn::Sequential layers;
    nn::Conv2d output_conv = nullptr;
public:
    GatedPixelCNNImpl(){}
    GatedPixelCNNImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(GatedPixelCNN);


// -------------------------------------------------
// struct{VectorQuantizerImpl}(nn::Module)
// -------------------------------------------------
struct VectorQuantizerImpl : nn::Module{
private:
    size_t K, D;
public:
    torch::Tensor e;
    VectorQuantizerImpl(){}
    VectorQuantizerImpl(const size_t K, const size_t D);
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
// struct{VQVAEImpl}(nn::Module)
// -------------------------------------------------
struct VQVAEImpl : nn::Module{
private:
    nn::Sequential encoder, decoder;
    VectorQuantizer vq;
public:
    VQVAEImpl(){}
    VQVAEImpl(po::variables_map &vm);
    torch::Tensor sampling(const std::vector<long int> z_shape, GatedPixelCNN pixelcnn, torch::Device device);
    torch::Tensor synthesis(torch::Tensor x, torch::Tensor y, const float alpha);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    torch::Tensor forward_idx(torch::Tensor x);
    std::vector<long int> get_z_shape(const std::vector<long int> x_shape, torch::Device &device);
};
TORCH_MODULE(VQVAE);


#endif