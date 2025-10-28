#include <vector>
#include <tuple>
#include <typeinfo>
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

    this->positions = register_parameter("positions", torch::randn({(long int)this->num_gaussians, 3}) * init_radius);
    this->cov_lower = register_parameter("cov_lower", torch::zeros({(long int)this->num_gaussians, 6}));
    this->colors = register_parameter("colors", torch::randn({(long int)this->num_gaussians, 3}) * 0.01);
    this->log_opacity = register_parameter("log_opacity", torch::full({(long int)this->num_gaussians, 1}, -10.0));
    this->background_logit = register_parameter("background_logit", torch::zeros({3}));
    
}


// --------------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{render_image}
// --------------------------------------------------------
torch::Tensor GS3DImpl::render_image(torch::Tensor pose){
    torch::NoGradGuard no_grad;
    return this->forward(pose);
}


// -----------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
torch::Tensor GS3DImpl::forward(torch::Tensor pose){

    constexpr float eps = 1e-3;
    
    torch::Device device = pose.device();
    long int N = pose.size(0);
    long int hw = this->size * this->size;

    float fx, fy, cx, cy;
    torch::Tensor xs, ys, grid_x, grid_y, R, t, cs, opacity, pos_expand, diff, Rwc, pos_cam;
    torch::Tensor xs_cam, ys_cam, zs_cam, valid, safe_z, inv_z, inv_z2;
    torch::Tensor mean_x, mean_y, grid_xv, grid_yv;
    torch::Tensor dx, dy, exponent, normalization, gaussian;
    torch::Tensor cov_params, l11, l21, l22, l31, l32, l33, L, cov_world, cov_cam;
    torch::Tensor fx_div_z, fy_div_z, fx_x_over_z2, fy_y_over_z2, zeros, row0, row1, J, cov2d;
    torch::Tensor Rwc_expand, R_expand;
    torch::Tensor eye2, cov_xx, cov_xy, cov_yy, det, inv_det, inv_xx, inv_xy, inv_yy;
    torch::Tensor alpha, sort_key, indices, idx_alpha, idx_color;
    torch::Tensor color_sorted, rgb, trans, a, c, weight, bg;

    xs = torch::arange((long int)this->size, torch::kFloat).to(device);
    ys = torch::arange((long int)this->size, torch::kFloat).to(device);
    grid_x = xs.unsqueeze(0).repeat({(long int)this->size, 1}).view({-1});
    grid_y = ys.unsqueeze(1).repeat({1, (long int)this->size}).view({-1});
    R = pose.index({Slice(), Slice(0, 3), Slice(0, 3)});
    t = pose.index({Slice(), Slice(0, 3), 3});

    cs = torch::sigmoid(this->colors);
    opacity = torch::sigmoid(this->log_opacity).squeeze(-1);

    pos_expand = this->positions.unsqueeze(0).expand({N, (long int)this->num_gaussians, 3});
    diff = pos_expand - t.unsqueeze(1);
    Rwc = R.transpose(1, 2);
    pos_cam = torch::bmm(diff, Rwc);

    xs_cam = pos_cam.index({Slice(), Slice(), 0});
    ys_cam = pos_cam.index({Slice(), Slice(), 1});
    zs_cam = pos_cam.index({Slice(), Slice(), 2});

    valid = zs_cam > eps;
    safe_z = torch::where(valid, zs_cam, torch::ones_like(zs_cam));
    inv_z = 1.0 / safe_z;

    fx = this->focal_length;
    fy = this->focal_length;
    cx = (this->size - 1.0) * 0.5;
    cy = (this->size - 1.0) * 0.5;

    mean_x = fx * (xs_cam * inv_z) + cx;
    mean_y = fy * (-(ys_cam * inv_z)) + cy;

    cov_params = this->cov_lower;
    l11 = torch::softplus(cov_params.index({Slice(), 0})) + eps;
    l21 = cov_params.index({Slice(), 1});
    l22 = torch::softplus(cov_params.index({Slice(), 2})) + eps;
    l31 = cov_params.index({Slice(), 3});
    l32 = cov_params.index({Slice(), 4});
    l33 = torch::softplus(cov_params.index({Slice(), 5})) + eps;

    L = torch::zeros({(long int)this->num_gaussians, 3, 3}).to(device);
    L.index_put_({Slice(), 0, 0}, l11);
    L.index_put_({Slice(), 1, 0}, l21);
    L.index_put_({Slice(), 1, 1}, l22);
    L.index_put_({Slice(), 2, 0}, l31);
    L.index_put_({Slice(), 2, 1}, l32);
    L.index_put_({Slice(), 2, 2}, l33);
    cov_world = torch::matmul(L, L.transpose(1, 2));

    cov_world = cov_world.unsqueeze(0).expand({N, (long int)this->num_gaussians, 3, 3});
    Rwc_expand = Rwc.unsqueeze(1).expand_as(cov_world);
    R_expand = R.unsqueeze(1).expand_as(cov_world);
    cov_cam = torch::matmul(torch::matmul(Rwc_expand, cov_world), R_expand);

    fx_div_z = fx * inv_z;
    fy_div_z = fy * inv_z;
    inv_z2 = inv_z * inv_z;
    fx_x_over_z2 = fx * xs_cam * inv_z2;
    fy_y_over_z2 = fy * ys_cam * inv_z2;

    zeros = torch::zeros_like(xs_cam);
    row0 = torch::stack({fx_div_z, zeros, -fx_x_over_z2}, -1);
    row1 = torch::stack({zeros, -fy_div_z, fy_y_over_z2}, -1);
    J = torch::stack({row0, row1}, -2);

    cov2d = torch::matmul(torch::matmul(J, cov_cam), J.transpose(-1, -2));
    eye2 = torch::eye(2).view({1, 1, 2, 2}).to(device);
    cov2d = cov2d + eye2 * eps;
    cov_xx = cov2d.index({Slice(), Slice(), 0, 0}).clamp_min(eps);
    cov_xy = cov2d.index({Slice(), Slice(), 0, 1});
    cov_yy = cov2d.index({Slice(), Slice(), 1, 1}).clamp_min(eps);
    
    det = torch::clamp(cov_xx * cov_yy - cov_xy.square(), eps);
    inv_det = 1.0 / det;
    inv_xx = cov_yy * inv_det;
    inv_xy = -cov_xy * inv_det;
    inv_yy = cov_xx * inv_det;
    normalization = torch::sqrt(inv_det);

    grid_xv = grid_x.view({1, 1, hw});
    grid_yv = grid_y.view({1, 1, hw});

    dx = grid_xv - mean_x.unsqueeze(-1);
    dy = grid_yv - mean_y.unsqueeze(-1);
    exponent = -0.5 * (inv_xx.unsqueeze(-1) * dx.square() + 2.0 * inv_xy.unsqueeze(-1) * dx * dy + inv_yy.unsqueeze(-1) * dy.square());
    gaussian = torch::exp(exponent) * normalization.unsqueeze(-1);

    alpha = opacity.unsqueeze(-1) * gaussian;
    alpha = alpha * valid.unsqueeze(-1).to(torch::kFloat);

    sort_key = torch::where(valid, zs_cam, torch::full_like(zs_cam, 1e6));
    indices = std::get<1>(torch::sort(sort_key, 1, /*descending=*/false));

    idx_alpha = indices.unsqueeze(-1).expand({N, (long int)this->num_gaussians, hw});
    idx_color = indices.unsqueeze(-1).expand({N, (long int)this->num_gaussians, 3});

    alpha = alpha.gather(1, idx_alpha);
    color_sorted = cs.unsqueeze(0).expand({N, (long int)this->num_gaussians, 3}).gather(1, idx_color);

    rgb = torch::zeros({N, hw, 3}).to(device);
    trans = torch::ones({N, hw}).to(device);

    for (size_t i = 0; i < this->num_gaussians; i++){
        a = alpha.index({Slice(), (long int)i, Slice()});
        c = color_sorted.index({Slice(), (long int)i, Slice()});
        weight = trans * a;
        rgb = rgb + weight.unsqueeze(-1) * c.unsqueeze(1);
        trans = trans * (1.0 - a);
    }

    bg = torch::sigmoid(this->background_logit.to(device));
    rgb = rgb + trans.unsqueeze(-1) * bg.unsqueeze(0).unsqueeze(1);

    rgb = rgb.view({N, (long int)this->size, (long int)this->size, 3});
    rgb = rgb.permute({0, 3, 1, 2}).contiguous();

    return rgb;

}


// ----------------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{init_gaussians}
// ----------------------------------------------------------
void GS3DImpl::init_gaussians(){
    torch::Tensor dist, new_pos;
    dist = torch::randn_like(this->positions);
    dist = F::normalize(dist, F::NormalizeFuncOptions().p(2).dim(1));
    new_pos = F::normalize(dist, F::NormalizeFuncOptions().p(2).dim(1)) * this->init_radius;
    this->positions.set_data(new_pos);
    return;
}



