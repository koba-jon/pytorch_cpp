#include <vector>
#include <tuple>
#include <typeinfo>
#include <algorithm>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;
using torch::indexing::Slice;
namespace po = boost::program_options;


// ---------------------------------------------
// struct{GS3DImpl}(nn::Module) -> constructor
// ---------------------------------------------
GS3DImpl::GS3DImpl(po::variables_map &vm){

    this->size = vm["size"].as<size_t>();
    this->focal_length = vm["focal_length"].as<float>();
    this->num_gaussians = vm["num_gaussians"].as<size_t>();
    this->init_radius = vm["init_radius"].as<float>();
    this->base_scale = 1.0;
    this->init_opacity = 0.0005;

    this->density_control_interval = vm["density_control_interval"].as<size_t>();
    this->density_control_max_new = vm["density_control_max_new"].as<size_t>();
    this->density_control_position_noise = 0.01;
    this->density_control_scale_noise = 0.01;

    this->mu_world = register_parameter("mu_world", torch::randn({(long int)this->num_gaussians, 3}) * init_radius);
    this->log_scale = register_parameter("log_scale", torch::full({(long int)this->num_gaussians, 3}, std::log(std::exp(this->base_scale) - 1.0)));
    this->quat = register_parameter("quat", torch::tensor({{1.0, 0.0, 0.0, 0.0}}, torch::kFloat).repeat({(long int)this->num_gaussians, 1}));
    this->colors = register_parameter("colors", torch::randn({1, (long int)this->num_gaussians, 3}) * 0.01);
    this->log_opacity = register_parameter("log_opacity", torch::full({1, (long int)this->num_gaussians, 1}, std::log(this->init_opacity) - std::log(1.0 - this->init_opacity)));
    this->mask = register_buffer("mask", torch::cat({torch::ones({(long int)this->num_gaussians / 10}, torch::kBool), torch::zeros({(long int)this->num_gaussians - (long int)this->num_gaussians / 10}, torch::kBool)}));
    this->background_logit = register_parameter("background_logit", torch::full({1, 1, 3}, 0.0));

}


// --------------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{render_image}
// --------------------------------------------------------
torch::Tensor GS3DImpl::render_image(torch::Tensor pose){
    torch::NoGradGuard no_grad;
    return this->forward(pose);
}


// ----------------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{quat_to_rotmat}
// ----------------------------------------------------------
torch::Tensor GS3DImpl::quat_to_rotmat(torch::Tensor q){

    torch::Tensor w, x, y, z;
    torch::Tensor r00, r01, r02, r10, r11, r12, r20, r21, r22, out;

    q = F::normalize(q, F::NormalizeFuncOptions().p(2).dim(1));
    w = q.index({Slice(), 0});
    x = q.index({Slice(), 1});
    y = q.index({Slice(), 2});
    z = q.index({Slice(), 3});

    r00 = 1.0 - 2.0 * (y * y + z * z);
    r01 = 2.0 * (x * y - w * z);
    r02 = 2.0 * (x * z + w * y);
    r10 = 2.0 * (x * y + w * z);
    r11 = 1.0 - 2.0 * (x * x + z * z);
    r12 = 2.0 * (y * z - w * x);
    r20 = 2.0 * (x * z - w * y);
    r21 = 2.0 * (y * z + w * x);
    r22 = 1.0 - 2.0 * (x * x + y * y);

    out = torch::stack({
        torch::stack({r00, r01, r02}, 1),
        torch::stack({r10, r11, r12}, 1),
        torch::stack({r20, r21, r22}, 1)
    }, 1);

    return out;

}


// -----------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
torch::Tensor GS3DImpl::forward(torch::Tensor pose){


    torch::Device device = pose.device();
    long int N = pose.size(0);

    
    // ----------------------------------------
    // 1. Build 2-dimensional mean
    // ----------------------------------------
    float fx, fy, cx, cy;
    torch::Tensor mu_world_expand, W, t, W_expand, t_expand, mu_cam, tx, ty, tz, valid, u, v, mu_2d;

    // (1) Build 3-dimensional mean of world coordinate
    mu_world_expand = this->mu_world.view({1, (long int)this->num_gaussians, 3, 1}).expand({N, (long int)this->num_gaussians, 3, 1});  // {N,G,3,1}

    // (2) Build 3-dimensional mean of camera coordinate
    W = pose.index({Slice(), Slice(0, 3), Slice(0, 3)});  // {N,3,3}
    t = pose.index({Slice(), Slice(0, 3), 3});  // {N,3}
    W_expand = W.unsqueeze(1).expand({N, (long int)this->num_gaussians, 3, 3});  // {N,G,3,3}
    t_expand = t.view({N, 1, 3, 1}).expand({N, (long int)this->num_gaussians, 3, 1});  // {N,G,3,1}
    mu_cam = torch::matmul(W_expand, mu_world_expand) + t_expand;  // {N,G,3,1}
    mu_cam = mu_cam.squeeze(-1);  // {N,G,3}

    // (3) Build 2-dimensional mean of camera coordinate
    tx = mu_cam.index({Slice(), Slice(), 0});  // {N,G}
    ty = mu_cam.index({Slice(), Slice(), 1});  // {N,G}
    tz = mu_cam.index({Slice(), Slice(), 2});  // {N,G}
    valid = tz > 1e-2;  // {N,G}
    valid = valid * mask.view({1, (long int)this->num_gaussians});  // {N,G}
    tz = torch::where(valid, tz, torch::full_like(tz, 1e-2));  // {N,G}
    fx = this->focal_length;
    fy = this->focal_length;
    cx = (this->size - 1.0) * 0.5;
    cy = (this->size - 1.0) * 0.5;
    u = fx * tx / tz + cx;  // {N,G}
    v = fy * ty / tz + cy;  // {N,G}
    mu_2d = torch::stack({u, v}, 2);  // {N,G,2}


    // ----------------------------------------
    // 2. Build 2x2 covariance matrix
    // ----------------------------------------
    torch::Tensor scale, S2, R, cov_world, cov_cam, zeros, row0, row1, J, trace, eps, cov_2d, eye2;

    // (1) Build 3x3 covariance matrix of world coordinate
    scale = torch::softplus(this->log_scale) + 1e-6;  // {G,3}
    S2 = torch::zeros({(long int)this->num_gaussians, 3, 3}).to(device);  // {G,3,3}
    S2.index_put_({Slice(), 0, 0}, scale.index({Slice(), 0}).pow(2.0));  // {G}
    S2.index_put_({Slice(), 1, 1}, scale.index({Slice(), 1}).pow(2.0));  // {G}
    S2.index_put_({Slice(), 2, 2}, scale.index({Slice(), 2}).pow(2.0));  // {G}
    R = this->quat_to_rotmat(this->quat);
    cov_world = torch::matmul(torch::matmul(R, S2), R.transpose(-1, -2));  // {G,3,3}
    cov_world = cov_world.unsqueeze(0).expand({N, (long int)this->num_gaussians, 3, 3});  // {N,G,3,3}

    // (2) Build 3x3 covariance matrix of camera coordinate
    cov_cam = torch::matmul(torch::matmul(W_expand, cov_world), W_expand.transpose(-1, -2));  // {N,G,3,3}

    // (3) Build 2x2 covariance matrix of camera coordinate
    zeros = torch::zeros_like(tx);  // {N,G}
    row0 = torch::stack({fx / tz, zeros, -fx * tx / (tz * tz)}, -1);  // {N,G,3}
    row1 = torch::stack({zeros, fy / tz, -fy * ty / (tz * tz)}, -1);  // {N,G,3}
    J = torch::stack({row0, row1}, -2);  // {N,G,2,3}
    cov_2d = torch::matmul(torch::matmul(J, cov_cam), J.transpose(-1, -2));  // {N,G,2,2}
    eye2 = torch::eye(2).view({1, 1, 2, 2}).to(device);  // {1,1,2,2}
    trace = cov_2d.index({Slice(), Slice(), 0, 0}) + cov_2d.index({Slice(), Slice(), 1, 1});  // {N,G}
    eps = (1e-6 * trace.abs() + 1e-8).unsqueeze(-1).unsqueeze(-1);  // {N,G,1,1}
    cov_2d = 0.5 * (cov_2d + cov_2d.transpose(-1, -2)) + eps * eye2;  // {N,G,2,2}


    // ----------------------------------------
    // 3. Calculate PDF of 2D Gaussian
    // ----------------------------------------
    torch::Tensor inv_cov_2d, xs, ys, grid_x, grid_y, grid, diff, exponent, pdf;

    // (1) Calculate inverse matrix of cov_2d
    inv_cov_2d = torch::linalg_inv(cov_2d);  // {N,G,2,2}

    // (2) Calculate grid
    xs = torch::arange((long int)this->size, torch::kFloat).to(device);  // {W}
    ys = torch::arange((long int)this->size, torch::kFloat).to(device);  // {H}
    grid_x = xs.unsqueeze(0).repeat({(long int)this->size, 1}).view({1, 1, 1, -1});  // {W} ===> {1,W} ===> {H,W} ===> {1,1,1,H*W}
    grid_y = ys.unsqueeze(1).repeat({1, (long int)this->size}).view({1, 1, 1, -1});  // {H} ===> {H,1} ===> {H,W} ===> {1,1,1,H*W}
    grid = torch::cat({grid_x, grid_y}, 2);  // {1,1,2,H*W}

    // (3) Calculate PDF
    diff = grid - mu_2d.unsqueeze(-1);  // {N,G,2,H*W}
    exponent = -0.5 * (diff * torch::matmul(inv_cov_2d, diff)).sum(2);  // {N,G,H*W}
    pdf = torch::exp(exponent) * this->mask.view({1, (long int)this->num_gaussians, 1}).to(torch::kFloat);  // {N,G,H*W}


    // ----------------------------------------
    // 4. Calculate RGB image
    // ----------------------------------------
    torch::Tensor tz_safe, indices, idx_alpha, idx_color, color, opacity, alpha, cumprod, T, trans, bg, rgb;

    // (1) Calculate alpha and color
    tz_safe = torch::where(valid, tz, torch::full_like(tz, 1e6));  // {N,G}
    indices = std::get<1>(torch::sort(tz_safe, 1, /*descending=*/false)).unsqueeze(-1);  // {N,G,1}
    idx_alpha = indices.expand({N, (long int)this->num_gaussians, (long int)(this->size * this->size)});  // {N,G} ===> {N,G,1} ===> {N,G,H*W}
    idx_color = indices.expand({N, (long int)this->num_gaussians, 3});  // {N,G} ===> {N,G,1} ===> {N,G,3}
    opacity = torch::sigmoid(this->log_opacity);  // {1,G,1}
    alpha = opacity * pdf * valid.unsqueeze(-1).to(torch::kFloat);  // {N,G,H*W}
    alpha = alpha.gather(1, idx_alpha);  // {N,G,H*W}
    color = torch::sigmoid(this->colors);  // {1,G,3}
    color = color.expand({N, (long int)this->num_gaussians, 3}).gather(1, idx_color);  // {G,3} ===> {1,G,3} ===> {N,G,3} ===> {N,G,3}

    // (2) Calculate RGB image with foreground color
    cumprod = (1.0 - alpha).cumprod(/*dim=*/1);  // {N,G,H*W}
    T = torch::ones_like(alpha);  // {N,G,H*W}
    T.index_put_({Slice(), Slice(1, alpha.size(1)), Slice()}, cumprod.index({Slice(), Slice(0, alpha.size(1) - 1), Slice()}));  // {N,G,H*W}
    rgb = torch::bmm((alpha * T).transpose(-1, -2), color);  // {N,H*W,3}

    // (3) Calculate RGB image with background color
    trans = (1.0 - alpha).prod(/*dim=*/1).unsqueeze(-1);  // {N,H*W,1}
    bg = torch::sigmoid(this->background_logit.to(device));  // {1,1,3}
    rgb = rgb + trans * bg;  // {N,H*W,3}
    rgb = rgb.view({N, (long int)this->size, (long int)this->size, 3});  // {N,H*W,3} ===> {N,H,W,3}
    rgb = rgb.permute({0, 3, 1, 2}).contiguous();  // {N,H,W,3} ===> {N,3,H,W}

    return rgb;

    
}



// -----------------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{adaptive_density_control}
// -----------------------------------------------------------
void GS3DImpl::adaptive_density_control(){

    long int new_gaussians, src_idx, dst_idx;
    torch::Tensor candidate_indices, free_indices;
    torch::Tensor position_noise, scale_noise, opacity, sorted_indices, parent_log_opacity, shared_log_opacity;
    torch::NoGradGuard no_grad;

    candidate_indices = torch::nonzero(this->mask).view({-1});
    free_indices = torch::nonzero(torch::logical_not(this->mask)).view({-1});

    if ((candidate_indices.size(0) == 0) || (free_indices.size(0) == 0)){
        return;
    }

    new_gaussians = std::min<long int>({(long int)this->density_control_max_new, free_indices.size(0), candidate_indices.size(0)});
    if (new_gaussians <= 0){
        return;
    }

    position_noise = torch::randn({new_gaussians, 3}, this->mu_world.options()) * this->density_control_position_noise;
    scale_noise = torch::randn({new_gaussians, 3}, this->log_scale.options()) * this->density_control_scale_noise;

    opacity = torch::sigmoid(this->log_opacity).view({(long int)this->num_gaussians});
    sorted_indices = std::get<1>(torch::sort(opacity, /*dim=*/0, /*descending=*/true));
    candidate_indices = sorted_indices.index({Slice(0, new_gaussians)});

    for (long int i = 0; i < new_gaussians; ++i){
        src_idx = candidate_indices.index({i % candidate_indices.size(0)}).item<long int>();
        dst_idx = free_indices.index({i}).item<long int>();

        parent_log_opacity = this->log_opacity.index({0, src_idx, 0});
        shared_log_opacity = parent_log_opacity - std::log(2.0);
        this->log_opacity.index_put_({0, src_idx, 0}, shared_log_opacity);
        this->log_opacity.index_put_({0, dst_idx, 0}, shared_log_opacity);

        this->mu_world.index_put_({dst_idx}, this->mu_world.index({src_idx}) + position_noise.index({i}));
        this->log_scale.index_put_({dst_idx}, this->log_scale.index({src_idx}) + scale_noise.index({i}));
        this->quat.index_put_({dst_idx}, this->quat.index({src_idx}));
        this->colors.index_put_({0, dst_idx}, this->colors.index({0, src_idx}));
        this->mask.index_put_({dst_idx}, true);
    }

    return;

}


// -----------------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{active_gaussians}
// -----------------------------------------------------------
size_t GS3DImpl::active_gaussians() const{

    return this->mask.to(torch::kLong).sum().item<int64_t>();

}


// -----------------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{total_gaussians}
// -----------------------------------------------------------
size_t GS3DImpl::total_gaussians() const{

    return this->num_gaussians;

}


// -----------------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{density_interval}
// -----------------------------------------------------------
size_t GS3DImpl::density_interval() const{

    return this->density_control_interval;

}

