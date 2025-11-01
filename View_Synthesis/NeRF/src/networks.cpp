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

    freq_idx = torch::arange(this->freqs).to(torch::kFloat).to(x.device());  // {F}
    freq = torch::pow(2.0, freq_idx);  // {F}
    argument = x.unsqueeze(2) * freq * PI;  // {N,ID,F}
    sin_enc = torch::sin(argument).flatten(1);  // {N,ID*F}
    cos_enc = torch::cos(argument).flatten(1);  // {N,ID*F}
    out = torch::cat({x, sin_enc, cos_enc}, 1);  // {N,ID+2*ID*F}

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

    x = pos;  // {N,PS}
    for (size_t i = 0; i < this->n_layers; i++){
        x = this->base_layers[i]->as<nn::Linear>()->forward(x);  // {*} ===> {N,HD}
        x = torch::relu(x);  // {N,HD}
        if ((i + 1) == this->n_layers / 2){
            x = torch::cat({x, pos}, 1);  // {N,HD+PD}
        }
    }

    sigma = torch::softplus(this->sigma_head->forward(x));  // {N,1}
    features = torch::relu(this->feature_head->forward(x));  // {N,HD}
    color_in = torch::cat({features, view_dirs}, 1);  // {N,HD+DD}
    rgb = torch::sigmoid(this->color_head->forward(color_in));  // {N,3}

    return {rgb, sigma};

}


// ---------------------------------------------
// struct{NeRFImpl}(nn::Module) -> constructor
// ---------------------------------------------
NeRFImpl::NeRFImpl(po::variables_map &vm){

    this->size = vm["size"].as<size_t>();
    this->focal_length = vm["focal_length"].as<float>();
    this->samples_fine = vm["samples_fine"].as<size_t>();
    this->samples_coarse = vm["samples_coarse"].as<size_t>();
    this->near_plane = vm["near"].as<float>();
    this->far_plane = vm["far"].as<float>();

    this->pos_encoder = register_module("pos_encoder", PositionalEncoding(vm["pos_freqs"].as<size_t>()));
    this->dir_encoder = register_module("dir_encoder", PositionalEncoding(vm["dir_freqs"].as<size_t>()));

    size_t pos_dim = this->pos_encoder->get_out_dim(3);
    size_t dir_dim = this->dir_encoder->get_out_dim(3);
    size_t hid_dim = vm["hid_dim"].as<size_t>();
    size_t n_layers = vm["n_layers"].as<size_t>();

    this->fine_field = register_module("fine_field", NeRFMLP(pos_dim, dir_dim, hid_dim, n_layers));
    this->coarse_field = register_module("coarse_field", NeRFMLP(pos_dim, dir_dim, hid_dim, n_layers));
    
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
    dirs_y = (grid_y - cy) / fy;  // {H*W}
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
torch::Tensor NeRFImpl::render_image(torch::Tensor pose){

    torch::NoGradGuard no_grad;
    torch::Tensor rays_o, rays_d, rgb, _;

    std::tie(rays_o, rays_d) = this->build_rays(pose);  // {N,4,4} ===> {N,H*W,3}, {N,H*W,3}
    std::tie(rgb, _) = this->forward_chunked(rays_o, rays_d);  // {N,H*W,3}, {N,H*W,3} ===> {N,H*W,3}
    rgb = rgb.view({pose.size(0), (long int)this->size, (long int)this->size, 3});  // {N,H*W,3} ===> {N,H,W,3}
    rgb = rgb.permute({0, 3, 1, 2}).contiguous();  // {N,3,H,W}

    return rgb;

}


// -----------------------------------------------------
// struct{NeRFImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> NeRFImpl::volume_render(NeRFMLP &field, torch::Tensor rays_o, torch::Tensor dirs, torch::Tensor z_vals){

    torch::Tensor dirs_expanded, origins_expanded, points, encoded_points, encoded_dirs, rgb, sigma;
    torch::Tensor deltas, alpha, accum, transmittance, weights, rgb_map;

    dirs_expanded = dirs.unsqueeze(2).expand({rays_o.size(0), rays_o.size(1), z_vals.size(2), 3}).contiguous();  // {N,R,S,3}
    origins_expanded = rays_o.unsqueeze(2).expand({rays_o.size(0), rays_o.size(1), z_vals.size(2), 3}).contiguous();  // {N,R,S,3}
    points = origins_expanded + dirs_expanded * z_vals.unsqueeze(3);  // {N,R,S,3}

    encoded_points = this->pos_encoder->forward(points.view({-1, 3}));  // {N*R*S,3} ===> {N*R*S,PD}
    encoded_dirs = this->dir_encoder->forward(dirs_expanded.view({-1, 3}));  // {N*R*S,3} ===> {N*R*S,DD}
    std::tie(rgb, sigma) = field->forward(encoded_points, encoded_dirs);  // {N*R*S,PD}, {N*R*S,DD} ===> {N*R*S,3}, {N*R*S,1}
    rgb = rgb.view({rays_o.size(0), rays_o.size(1), z_vals.size(2), 3});  // {N,R,S,3}
    sigma = sigma.view({rays_o.size(0), rays_o.size(1), z_vals.size(2)});  // {N,R,S}

    deltas = z_vals.narrow(2, 1, z_vals.size(2) - 1) - z_vals.narrow(2, 0, z_vals.size(2) - 1);  // {N,R,S-1}
    deltas = torch::cat({deltas, torch::full({rays_o.size(0), rays_o.size(1), 1}, 1e10).to(rays_o.device())}, 2);  // {N,R,S}

    alpha = 1.0 - torch::exp(-sigma * deltas);  // {N,R,S}
    accum = torch::cat({torch::ones({rays_o.size(0), rays_o.size(1), 1}).to(rays_o.device()), 1.0 - alpha + 1e-10}, 2);  // {N,R,S+1}
    accum = torch::cumprod(accum, 2);  // {N,R,S+1}
    transmittance = accum.narrow(2, 0, z_vals.size(2));  // {N,R,S}
    weights = alpha * transmittance;  // {N,R,S}

    rgb_map = torch::sum(weights.unsqueeze(3) * rgb, 2);  // {N,R,S,3} ===> {N,R,3}

    return {rgb_map, weights};

}


// -----------------------------------------------------
// struct{NeRFImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
torch::Tensor NeRFImpl::sample_pdf(torch::Tensor bins, torch::Tensor weights, size_t n_samples){

    long int max_cdf_idx, max_bin_idx;
    torch::Tensor pdf, cdf, zeros, u, inds, below_cdf, above_cdf, below_bins, above_bins;
    torch::Tensor cdf_g0, cdf_g1, bins_g0, bins_g1, denom, t, samples;

    pdf = (weights + 1e-5) / torch::sum(weights + 1e-5, /*dim=*/2, /*keepdim=*/true);  // {N,R,S-1}
    cdf = torch::cumsum(pdf, /*dim=*/2);  // {N,R,S-1}
    zeros = torch::zeros({cdf.size(0), cdf.size(1), 1}).to(cdf.device());  // {N,R,1}
    cdf = torch::cat({zeros, cdf}, 2);  // {N,R,S}

    u = torch::rand({cdf.size(0), cdf.size(1), (long int)n_samples}).to(cdf.device());  // {N,R,NS}

    inds = torch::searchsorted(cdf, u, true);  // {N,R,NS}
    max_cdf_idx = (long int)(cdf.size(2) - 1);
    max_bin_idx = (long int)(bins.size(2) - 1);
    below_cdf = torch::clamp(inds - 1, 0, max_cdf_idx);  // {N,R,NS}
    above_cdf = torch::clamp(inds, 0, max_cdf_idx);  // {N,R,NS}
    below_bins = torch::clamp(inds - 1, 0, max_bin_idx);  // {N,R,NS}
    above_bins = torch::clamp(inds, 0, max_bin_idx);  // {N,R,NS}

    cdf_g0 = torch::gather(cdf, 2, below_cdf);  // {N,R,NS}
    cdf_g1 = torch::gather(cdf, 2, above_cdf);  // {N,R,NS}
    bins_g0 = torch::gather(bins, 2, below_bins);  // {N,R,NS}
    bins_g1 = torch::gather(bins, 2, above_bins);  // {N,R,NS}

    denom = cdf_g1 - cdf_g0;  // {N,R,NS}
    denom = torch::where(denom < 1e-5, torch::ones_like(denom), denom);  // {N,R,NS}
    t = (u - cdf_g0) / denom;  // {N,R,NS}
    samples = bins_g0 + t * (bins_g1 - bins_g0);  // {N,R,NS}

    return samples.detach();

}


// -----------------------------------------------------
// struct{NeRFImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> NeRFImpl::forward(torch::Tensor rays_o, torch::Tensor rays_d){

    torch::Tensor dirs, z_vals_coarse, upper, lower, rand, rgb_coarse, weights_coarse;
    torch::Tensor z_vals_mid, weights_mid, z_samples, z_vals_all, z_vals_fine, rgb_fine, _;

    dirs = rays_d;  // {N,R,3}
    dirs = dirs / (torch::norm(dirs, 2, /*dim=*/2, /*keepdim=*/true) + 1e-6);  // {N,R,3}

    z_vals_coarse = torch::linspace(this->near_plane, this->far_plane, this->samples_coarse).to(rays_o.device());  // {Sc}
    z_vals_coarse = z_vals_coarse.view({1, 1, -1}).expand({rays_o.size(0), rays_o.size(1), (long int)this->samples_coarse}).contiguous();  // {N,R,Sc}
    upper = torch::cat({z_vals_coarse.narrow(2, 1, this->samples_coarse - 1), torch::full({rays_o.size(0), rays_o.size(1), 1}, this->far_plane).to(rays_o.device())}, 2);  // {N,R,Sc}
    lower = torch::cat({torch::full({rays_o.size(0), rays_o.size(1), 1}, this->near_plane).to(rays_o.device()), z_vals_coarse.narrow(2, 0, this->samples_coarse - 1)}, 2);  // {N,R,Sc}
    rand = torch::rand_like(z_vals_coarse);  // {N,R,Sc}
    z_vals_coarse = lower + (upper - lower) * rand;  // {N,R,Sc}
    std::tie(rgb_coarse, weights_coarse) = this->volume_render(this->coarse_field, rays_o, dirs, z_vals_coarse);  // {N,R,3}, {N,R,Sc}

    z_vals_mid = 0.5 * (z_vals_coarse.narrow(2, 0, this->samples_coarse - 1) + z_vals_coarse.narrow(2, 1, this->samples_coarse - 1));  // {N,R,Sc-1}
    weights_mid = weights_coarse.narrow(2, 0, this->samples_coarse - 1).detach();  // {N,R,Sc-1}
    z_samples = this->sample_pdf(z_vals_mid, weights_mid, this->samples_fine);  // {N,R,Sf}
    z_vals_all = torch::cat({z_vals_coarse, z_samples}, 2);  // {N,R,Sc+Sf}
    z_vals_fine = std::get<0>(torch::sort(z_vals_all, 2));  // {N,R,Sc+Sf}
    std::tie(rgb_fine, _) = this->volume_render(this->fine_field, rays_o, dirs, z_vals_fine);  // {N,R,3}

    return {rgb_fine.contiguous(), rgb_coarse.contiguous()};  // {N,R,3}, {N,R,3}

}


// -----------------------------------------------------------
// struct{NeRFImpl}(nn::Module) -> function{forward_chunked}
// -----------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> NeRFImpl::forward_chunked(torch::Tensor rays_o, torch::Tensor rays_d, long int chunk){

    long int start, end;
    torch::Tensor ro, rd, rgb_fine, rgb_coarse;
    std::tuple<torch::Tensor, torch::Tensor> out;
    std::vector<torch::Tensor> rgb_fine_list, rgb_coarse_list;
    
    for (start = 0; start < rays_o.size(1); start += chunk){
        end = std::min(start + chunk, rays_o.size(1));
        ro = rays_o.index({Slice(), Slice(start, end), Slice()});  // {N,chunk,3}
        rd = rays_d.index({Slice(), Slice(start, end), Slice()});  // {N,chunk,3}
        out = this->forward(ro.contiguous(), rd.contiguous());  // {N,chunk,3}, {N,chunk,3}
        rgb_fine_list.push_back(std::get<0>(out));  // append {N,chunk,3}
        rgb_coarse_list.push_back(std::get<1>(out));  // append {N,chunk,3}
    }

    rgb_fine = torch::cat(rgb_fine_list, 1).contiguous();  // {N,R,3}
    rgb_coarse = torch::cat(rgb_coarse_list, 1).contiguous();  // {N,R,3}

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

