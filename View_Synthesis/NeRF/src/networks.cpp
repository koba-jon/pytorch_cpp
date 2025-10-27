#include <vector>
#include <tuple>
#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>
// For Original Header
#include "networks.hpp"

// Define
#define PI 3.14159265358979

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;
using torch::indexing::Slice;
namespace po = boost::program_options;


// -----------------------------------------------------------
// struct{PositionalEncodingImpl}(nn::Module) -> constructor
// -----------------------------------------------------------
PositionalEncodingImpl::PositionalEncodingImpl(size_t freqs_){
    this->freqs = freqs_;
}


// -----------------------------------------------------------------
// struct{PositionalEncodingImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------
torch::Tensor PositionalEncodingImpl::forward(torch::Tensor x){

    torch::Tensor freq_idx, freq, argument, sin_enc, cos_enc, out;

    freq_idx = torch::arange(this->freqs).to(torch::kFloat).to(x.device());
    freq = torch::pow(2.0, freq_idx);
    argument = x.unsqueeze(-1) * freq * PI;
    sin_enc = torch::sin(argument).flatten(-2);
    cos_enc = torch::cos(argument).flatten(-2);
    out = torch::cat({x, sin_enc, cos_enc}, -1);

    return out;

}


// -----------------------------------------------------------------
// struct{PositionalEncodingImpl}(nn::Module) -> function{get_dim}
// -----------------------------------------------------------------
size_t PositionalEncodingImpl::get_out_dim(size_t in_dim){
    return in_dim + in_dim * 2 * this->freqs;
}


// ------------------------------------------------
// struct{NeRFMLPImpl}(nn::Module) -> constructor
// ------------------------------------------------
NeRFMLPImpl::NeRFMLPImpl(size_t pos_dim, size_t dir_dim, size_t hid_dim, size_t n_layers_){

    size_t in_dim;

    this->n_layers = n_layers_;
    
    in_dim = pos_dim;
    for (size_t i = 0; i < this->n_layers; i++){
        this->base_layers->push_back(nn::Linear(in_dim, hid_dim));
        in_dim = hid_dim;
        if ((i + 1) == this->n_layers / 2){
            in_dim += pos_dim;
        }
    }
    register_module("base_layers", this->base_layers);

    this->sigma_head = register_module("sigma_head", nn::Linear(hid_dim, 1));
    this->feature_head = register_module("feature_head", nn::Linear(hid_dim, hid_dim));
    
    this->color_head = nn::Sequential(
        nn::Linear(hid_dim + dir_dim, hid_dim / 2),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Linear(hid_dim / 2, 3)
    );
    register_module("color_head", this->color_head);

}


// ------------------------------------------------------
// struct{NeRFMLPImpl}(nn::Module) -> function{forward}
// ------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> NeRFMLPImpl::forward(torch::Tensor pos, torch::Tensor view_dirs){

    torch::Tensor x, sigma, features, color_in, rgb;

    x = pos;
    for (size_t i = 0; i < this->n_layers; i++){
        x = this->base_layers[i]->as<nn::Linear>()->forward(x);
        x = torch::relu(x);
        if ((i + 1) == this->n_layers / 2){
            x = torch::cat({x, pos}, -1);
        }
    }

    sigma = torch::relu(this->sigma_head->forward(x));
    features = torch::relu(this->feature_head->forward(x));
    color_in = torch::cat({features, view_dirs}, -1);
    rgb = torch::sigmoid(this->color_head->forward(color_in));

    return {rgb, sigma};

}


// ---------------------------------------------
// struct{NeRFImpl}(nn::Module) -> constructor
// ---------------------------------------------
NeRFImpl::NeRFImpl(po::variables_map &vm){

    this->size = vm["size"].as<size_t>();
    this->focal_length = vm["focal_length"].as<float>();
    this->samples_coarse = vm["samples_coarse"].as<size_t>();
    this->samples_fine = vm["samples_fine"].as<size_t>();
    this->near_plane = vm["near"].as<float>();
    this->far_plane = vm["far"].as<float>();

    this->pos_encoder = register_module("pos_encoder", PositionalEncoding(vm["pos_freqs"].as<size_t>()));
    this->dir_encoder = register_module("dir_encoder", PositionalEncoding(vm["dir_freqs"].as<size_t>()));

    size_t pos_dim = this->pos_encoder->get_out_dim(3);
    size_t dir_dim = this->dir_encoder->get_out_dim(3);
    size_t hid_dim = vm["hid_dim"].as<size_t>();
    size_t n_layers = vm["n_layers"].as<size_t>();

    this->coarse_field = register_module("coarse_field", NeRFMLP(pos_dim, dir_dim, hid_dim, n_layers));
    this->fine_field = register_module("coarse_field", NeRFMLP(pos_dim, dir_dim, hid_dim, n_layers));
    
}


// ------------------------------------------------------
// struct{NeRFImpl}(nn::Module) -> function{build_rays}
// ------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> NeRFImpl::build_rays(torch::Tensor pose){

    float fx, fy, cx, cy;
    torch::Tensor xs, ys, grid_x, grid_y, dirs_x, dirs_y, dirs_z, dirs_cam, R, t, rays_d, rays_o;

    fx = this->focal_length;
    fy = this->focal_length;
    cx = (this->size - 1.0) * 0.5;
    cy = (this->size - 1.0) * 0.5;

    xs = torch::arange(this->size).to(torch::kFloat).to(pose.device());  // {W}
    ys = torch::arange(this->size).to(torch::kFloat).to(pose.device());  // {H}
    grid_x = xs.unsqueeze(0).repeat({(long int)this->size, 1}).view({-1});  // {W} ===> {1,W} ===> {H,W} ===> {H*W}
    grid_y = ys.unsqueeze(1).repeat({1, (long int)this->size}).view({-1});  // {H} ===> {H,1} ===> {H,W} ===> {H*W}

    dirs_x = (grid_x - cx) / fx;  // {H*W}
    dirs_y = -(grid_y - cy) / fy;  // {H*W}
    dirs_z = torch::ones_like(dirs_x);  // {H*W}
    dirs_cam = torch::stack({dirs_x, dirs_y, dirs_z}, 1);  // {H*W,3}
    dirs_cam = F::normalize(dirs_cam, F::NormalizeFuncOptions().p(2).dim(1));  // {H*W,3}
    dirs_cam = dirs_cam.unsqueeze(0).expand({pose.size(0), (long int)(this->size * this->size), 3}).contiguous();  // {H*W,3} ===> {1,H*W,3} ===> {N,H*W,3}

    R = pose.index({Slice(), Slice(0, 3), Slice(0, 3)}).contiguous();  // {N,3,3}
    t = pose.index({Slice(), Slice(0, 3), 3}).contiguous();  // {N,3}
    rays_d = torch::bmm(dirs_cam, R.transpose(1, 2));  // {N,H*W,3}
    rays_d = F::normalize(rays_d, F::NormalizeFuncOptions().p(2).dim(2));  // {N,H*W,3}
    rays_o = t.unsqueeze(1).expand({pose.size(0), (long int)(this->size * this->size), 3}).contiguous();  // {N,3} ===> {N,1,3} ===> {N,H*W,3}

    return {rays_o, rays_d};

}


// -----------------------------------------------------
// struct{NeRFImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> NeRFImpl::volume_render(NeRFMLP &field, torch::Tensor rays_o, torch::Tensor dirs, torch::Tensor z_vals){

    torch::Tensor dirs_expanded, origins_expanded, points, encoded_points, encoded_dirs, rgb, sigma;
    torch::Tensor deltas, alpha, accum, transmittance, weights, rgb_map;

    dirs_expanded = dirs.unsqueeze(2).expand({rays_o.size(0), rays_o.size(1), z_vals.size(2), 3}).contiguous();
    origins_expanded = rays_o.unsqueeze(2).expand({rays_o.size(0), rays_o.size(1), z_vals.size(2), 3}).contiguous();
    points = origins_expanded + dirs_expanded * z_vals.unsqueeze(-1);

    encoded_points = this->pos_encoder->forward(points.view({-1, 3}));
    encoded_dirs = this->dir_encoder->forward(dirs_expanded.view({-1, 3}));
    std::tie(rgb, sigma) = field->forward(encoded_points, encoded_dirs);
    rgb = rgb.view({rays_o.size(0), rays_o.size(1), z_vals.size(2), 3});
    sigma = sigma.view({rays_o.size(0), rays_o.size(1), z_vals.size(2)});

    deltas = z_vals.narrow(2, 1, z_vals.size(2) - 1) - z_vals.narrow(2, 0, z_vals.size(2) - 1);
    deltas = torch::cat({deltas, torch::full({rays_o.size(0), rays_o.size(1), 1}, 1e10).to(rays_o.device())}, 2);

    alpha = 1.0 - torch::exp(-sigma * deltas);
    accum = torch::cat({torch::ones({rays_o.size(0), rays_o.size(1), 1}).to(rays_o.device()), 1.0 - alpha + 1e-10}, -1);
    accum = torch::cumprod(accum, -1);
    transmittance = accum.narrow(2, 0, z_vals.size(2));
    weights = alpha * transmittance;

    rgb_map = torch::sum(weights.unsqueeze(-1) * rgb, -2);

    return {rgb_map, weights};

}


// -----------------------------------------------------
// struct{NeRFImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
torch::Tensor NeRFImpl::sample_pdf(torch::Tensor bins, torch::Tensor weights, size_t n_samples){

    long int max_cdf_idx, max_bin_idx;
    torch::Tensor pdf, cdf, zeros, u, inds, below_cdf, above_cdf, below_bins, above_bins;
    torch::Tensor cdf_g0, cdf_g1, bins_g0, bins_g1, denom, t, samples;

    pdf = (weights + 1e-5) / torch::sum(weights + 1e-5, -1, true);
    cdf = torch::cumsum(pdf, -1);
    zeros = torch::zeros({cdf.size(0), cdf.size(1), 1}).to(cdf.device());
    cdf = torch::cat({zeros, cdf}, -1);

    u = torch::rand({cdf.size(0), cdf.size(1), (long int)n_samples}).to(cdf.device());

    inds = torch::searchsorted(cdf, u, true);
    max_cdf_idx = (long int)(cdf.size(-1) - 1);
    max_bin_idx = (long int)(bins.size(-1) - 1);
    below_cdf = torch::clamp(inds - 1, 0, max_cdf_idx);
    above_cdf = torch::clamp(inds, 0, max_cdf_idx);
    below_bins = torch::clamp(inds - 1, 0, max_bin_idx);
    above_bins = torch::clamp(inds, 0, max_bin_idx);

    cdf_g0 = torch::gather(cdf, -1, below_cdf);
    cdf_g1 = torch::gather(cdf, -1, above_cdf);
    bins_g0 = torch::gather(bins, -1, below_bins);
    bins_g1 = torch::gather(bins, -1, above_bins);

    denom = cdf_g1 - cdf_g0;
    denom = torch::where(denom < 1e-5, torch::ones_like(denom), denom);
    t = (u - cdf_g0) / denom;
    samples = bins_g0 + t * (bins_g1 - bins_g0);

    return samples.detach();

}


// -----------------------------------------------------
// struct{NeRFImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> NeRFImpl::forward(torch::Tensor rays_o, torch::Tensor rays_d){

    torch::Tensor dirs, dir_norm, z_vals_coarse, upper, lower, rand, rgb_coarse, weights_coarse;
    torch::Tensor z_vals_mid, weights_mid, z_samples, z_vals_all, z_vals_fine, rgb_fine, _;

    dirs = rays_d;
    dir_norm = torch::norm(dirs, 2, -1, true);
    dirs = dirs / (dir_norm + 1e-6);

    z_vals_coarse = torch::linspace(this->near_plane, this->far_plane, this->samples_coarse).to(rays_o.device());
    z_vals_coarse = z_vals_coarse.view({1, 1, -1}).expand({rays_o.size(0), rays_o.size(1), (long int)this->samples_coarse}).contiguous();
    upper = torch::cat({z_vals_coarse.narrow(2, 1, this->samples_coarse - 1), torch::full({rays_o.size(0), rays_o.size(1), 1}, this->far_plane).to(rays_o.device())}, 2);
    lower = torch::cat({torch::full({rays_o.size(0), rays_o.size(1), 1}, this->near_plane).to(rays_o.device()), z_vals_coarse.narrow(2, 0, this->samples_coarse - 1)}, 2);
    rand = torch::rand_like(z_vals_coarse);
    z_vals_coarse = lower + (upper - lower) * rand;
    std::tie(rgb_coarse, weights_coarse) = this->volume_render(this->coarse_field, rays_o, dirs, z_vals_coarse);

    z_vals_mid = 0.5 * (z_vals_coarse.narrow(2, 0, this->samples_coarse - 1) + z_vals_coarse.narrow(2, 1, this->samples_coarse - 1));
    weights_mid = weights_coarse.narrow(2, 0, this->samples_coarse - 1).detach();
    z_samples = this->sample_pdf(z_vals_mid, weights_mid, this->samples_fine);
    z_vals_all = torch::cat({z_vals_coarse, z_samples}, -1);
    z_vals_fine = std::get<0>(torch::sort(z_vals_all, -1));
    std::tie(rgb_fine, _) = this->volume_render(this->fine_field, rays_o, dirs, z_vals_fine);

    return {rgb_fine, rgb_coarse};

}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}

