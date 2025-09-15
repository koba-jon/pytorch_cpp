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
// struct{GatedActivationImpl}(nn::Module)
// -------------------------------------------------
struct GatedActivationImpl : nn::Module{
public:
    GatedActivationImpl(){}
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(GatedActivation);


// -------------------------------------------------
// struct{GatedMaskedConv2dImpl}(nn::Module)
// -------------------------------------------------
struct GatedMaskedConv2dImpl : nn::Module{
private:
    char mask_type;
    bool residual;
    nn::Embedding class_cond = nullptr;
    nn::Conv2d vert_stack = nullptr;
    nn::Conv2d vert_to_horiz = nullptr;
    nn::Conv2d horiz_stack = nullptr;
    nn::Conv2d horiz_resid = nullptr;
    GatedActivation gate;
    torch::Tensor vmask, hmask;
public:
    GatedMaskedConv2dImpl(){}
    GatedMaskedConv2dImpl(char mask_type_, long int dim, long int kernel, bool residual_=true);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x_v, torch::Tensor x_h);
};
TORCH_MODULE(GatedMaskedConv2d);


// -------------------------------------------------
// struct{GatedPixelCNNImpl}(nn::Module)
// -------------------------------------------------
struct GatedPixelCNNImpl : nn::Module{
private:
    long int dim;
    nn::Embedding token_emb = nullptr;
    nn::ModuleList layers;
    nn::Sequential output_conv;
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
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    torch::Tensor forward_idx(torch::Tensor x);
    torch::Tensor forward_z(torch::Tensor z);
    std::vector<long int> get_z_shape(const std::vector<long int> x_shape, torch::Device &device);
};
TORCH_MODULE(VQVAE);


#endif