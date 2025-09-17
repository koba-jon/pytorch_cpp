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
// struct{PixelSnailImpl}(nn::Module)
// -------------------------------------------------
struct PixelSnailImpl : nn::Module{
private:
    long int dim;
    nn::Embedding token_emb = nullptr;
    nn::Sequential cond_resnet;
    nn::Sequential layers;
    nn::Conv2d output_conv = nullptr;
public:
    PixelSnailImpl(){}
    PixelSnailImpl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x, std::vector<torch::Tensor> condition={});
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