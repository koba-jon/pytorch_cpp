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


// ------------------------------
// struct{PositionalEncodingImpl}(nn::Module)
// ------------------------------
struct PositionalEncodingImpl : nn::Module{
private:
    size_t freqs;
public:
    PositionalEncodingImpl(){}
    PositionalEncodingImpl(size_t freqs_);
    torch::Tensor forward(torch::Tensor x);
    size_t get_out_dim(size_t in_dim);
};
TORCH_MODULE(PositionalEncoding);


// ------------------------------
// struct{NeRFMLPImpl}(nn::Module)
// ------------------------------
struct NeRFMLPImpl : nn::Module{
private:
    size_t n_layers;
    nn::ModuleList base_layers;
    nn::Linear sigma_head{nullptr}, feature_head{nullptr};
    nn::Sequential color_head;
public:
    NeRFMLPImpl(){}
    NeRFMLPImpl(size_t pos_dim, size_t dir_dim, size_t hid_dim, size_t n_layers_);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor pos, torch::Tensor view_dirs);
};
TORCH_MODULE(NeRFMLP);


// ------------------------------
// struct{NeRFImpl}(nn::Module)
// ------------------------------
struct NeRFImpl : nn::Module{
private:
    size_t size, samples_coarse, samples_fine;
    float focal_length, near_plane, far_plane;
    PositionalEncoding pos_encoder, dir_encoder;
    NeRFMLP coarse_field, fine_field;
    std::tuple<torch::Tensor, torch::Tensor> build_rays(torch::Tensor pose);
    std::tuple<torch::Tensor, torch::Tensor> volume_render(NeRFMLP &field, torch::Tensor rays_o, torch::Tensor dirs, torch::Tensor z_vals);
    torch::Tensor sample_pdf(torch::Tensor bins, torch::Tensor weights, size_t n_samples);
public:
    NeRFImpl(){}
    NeRFImpl(po::variables_map &vm);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor rays_o, torch::Tensor rays_d);
};
TORCH_MODULE(NeRF);


#endif