#include <fstream>
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
    this->cov_lower = register_parameter("cov_lower", torch::randn({(long int)this->num_gaussians, 6}));
    this->colors = register_parameter("colors", torch::randn({(long int)this->num_gaussians, 3}) * 0.01);
    this->log_opacity = register_parameter("log_opacity", torch::full({(long int)this->num_gaussians, 1}, -2.0));
    this->background_logit = register_parameter("background_logit", torch::zeros({3}));

    std::ofstream ofs("tmp.txt", std::ios::out);
    ofs.close();
    
}


// --------------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{render_image}
// --------------------------------------------------------
torch::Tensor GS3DImpl::render_image(torch::Tensor pose){
    torch::NoGradGuard no_grad;
    return this->forward(pose);
}


inline void CHECK_TENSOR(const char* tag, const torch::Tensor& t, std::ofstream& ofs) {
    // まずは isfinite で一気に判定
    auto finite = torch::isfinite(t);
    bool any_nan = torch::isnan(t).any().item<bool>();
    bool any_inf = torch::isinf(t).any().item<bool>();

    // 統計値（finite だけで平均とmin/maxを出す）
    float mean_v = 0.0f, min_v = 0.0f, max_v = 0.0f;
    auto finite_sum = finite.sum().item<int64_t>();
    if (finite_sum > 0) {
        auto v = t.masked_select(finite);
        mean_v = v.mean().item<float>();
        min_v  = v.amin().item<float>();
        max_v  = v.amax().item<float>();
    }

    ofs << "[CHK] " << tag
        << " nan=" << any_nan
        << " inf=" << any_inf
        << " mean=" << mean_v
        << " min=" << min_v
        << " max=" << max_v
        << " shape=" << t.sizes() << std::endl;

    // 見つけたら即死（行番号で止めたいから）
    TORCH_CHECK(!any_nan && !any_inf, "NaN/Inf detected at: ", tag);
}

// -----------------------------------------------------
// struct{GS3DImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------
torch::Tensor GS3DImpl::forward(torch::Tensor pose){

    torch::Device device = pose.device();
    long int N = pose.size(0);
    long int hw = this->size * this->size;
    std::ofstream ofs("tmp.txt", std::ios::app);
    ofs << std::endl;

    float fx, fy, cx, cy;
    torch::Tensor xs, ys, grid_x, grid_y, R, t, cs, opacity, pos_expand, diff, Rwc, pos_cam;
    torch::Tensor xs_cam, ys_cam, zs_cam, valid, safe_z, inv_z, inv_z2;
    torch::Tensor mean_x, mean_y, grid_xv, grid_yv;
    torch::Tensor dx, dy, exponent, normalization, gaussian;
    torch::Tensor l11, l21, l22, l31, l32, l33, L, cov_world, cov_cam;
    torch::Tensor fx_div_z, fy_div_z, fx_x_over_z2, fy_y_over_z2, zeros, row0, row1, J, cov2d;
    torch::Tensor Rwc_expand, R_expand;
    torch::Tensor eye2, cov_xx, cov_xy, cov_yy, sgn, det, inv_det, inv_xx, inv_xy, inv_yy;
    torch::Tensor alpha, sort_key, indices, idx_alpha, idx_color;
    torch::Tensor color_sorted, rgb, T, trans, cumprod, weight, bg;

    xs = torch::arange((long int)this->size, torch::kFloat).to(device);  // {W}
    ys = torch::arange((long int)this->size, torch::kFloat).to(device);  // {H}
    grid_x = xs.unsqueeze(0).repeat({(long int)this->size, 1}).view({-1});  // {W} ===> {1,W} ===> {H,W} ===> {H*W}
    grid_y = ys.unsqueeze(1).repeat({1, (long int)this->size}).view({-1});  // {H} ===> {H,1} ===> {H,W} ===> {H*W}
    R = pose.index({Slice(), Slice(0, 3), Slice(0, 3)});  // {N,3,3}
    t = pose.index({Slice(), Slice(0, 3), 3});  // {N,3}

    cs = torch::sigmoid(this->colors);  // {G,3}
    opacity = torch::sigmoid(this->log_opacity).squeeze(-1);  // {G}

    pos_expand = this->positions.unsqueeze(0).expand({N, (long int)this->num_gaussians, 3});  // {G,3} ===> {1,G,3} ===> {N,G,3}
    CHECK_TENSOR("pos_expand", pos_expand, ofs);
    diff = pos_expand - t.unsqueeze(1);  // {N,G,3}
    Rwc = R.transpose(1, 2);  // {N,3,3}
    pos_cam = torch::bmm(diff, Rwc);  // {N,G,3}

    xs_cam = pos_cam.index({Slice(), Slice(), 0});  // {N,G}
    ys_cam = pos_cam.index({Slice(), Slice(), 1});  // {N,G}
    zs_cam = pos_cam.index({Slice(), Slice(), 2});  // {N,G}

    valid = zs_cam > 1e-2;  // {N,G}
    safe_z = torch::where(valid, zs_cam, torch::full_like(zs_cam, 1e-2));  // {N,G}
    inv_z = 1.0 / safe_z;  // {N,G}
    inv_z2 = inv_z * inv_z;  // {N,G}

    fx = this->focal_length;
    fy = this->focal_length;
    cx = (this->size - 1.0) * 0.5;
    cy = (this->size - 1.0) * 0.5;

    mean_x = fx * (xs_cam * inv_z) + cx;  // {N,G}
    mean_y = fy * (-(ys_cam * inv_z)) + cy;  // {N,G}

    l11 = torch::softplus(this->cov_lower.index({Slice(), 0})) + 1e-4;  // {G}
    l21 = this->cov_lower.index({Slice(), 1});
    l22 = torch::softplus(this->cov_lower.index({Slice(), 2})) + 1e-4;  // {G}
    l31 = this->cov_lower.index({Slice(), 3});
    l32 = this->cov_lower.index({Slice(), 4});
    l33 = torch::softplus(this->cov_lower.index({Slice(), 5})) + 1e-4;  // {G}
    CHECK_TENSOR("cov_lower", this->cov_lower, ofs);

    L = torch::zeros({(long int)this->num_gaussians, 3, 3}).to(device);  // {G,3,3}
    L.index_put_({Slice(), 0, 0}, l11);  // {G}
    L.index_put_({Slice(), 1, 0}, l21);  // {G}
    L.index_put_({Slice(), 1, 1}, l22);  // {G}
    L.index_put_({Slice(), 2, 0}, l31);  // {G}
    L.index_put_({Slice(), 2, 1}, l32);  // {G}
    L.index_put_({Slice(), 2, 2}, l33);  // {G}

    cov_world = torch::matmul(L, L.transpose(-1, -2));  // {G,3,3}
    cov_world = cov_world.unsqueeze(0).expand({N, (long int)this->num_gaussians, 3, 3});  // {N,G,3,3}
    CHECK_TENSOR("cov_world", cov_world, ofs);
    Rwc_expand = Rwc.unsqueeze(1).expand_as(cov_world);  // {N,G,3,3}
    R_expand = R.unsqueeze(1).expand_as(cov_world);  // {N,G,3,3}
    cov_cam = torch::matmul(torch::matmul(Rwc_expand, cov_world), R_expand);  // {N,G,3,3}
    CHECK_TENSOR("cov_cam", cov_cam, ofs);

    fx_div_z = fx * inv_z;  // {N,G}
    fy_div_z = fy * inv_z;  // {N,G}
    fx_x_over_z2 = fx * xs_cam * inv_z2;  // {N,G}
    fy_y_over_z2 = fy * ys_cam * inv_z2;  // {N,G}

    zeros = torch::zeros_like(xs_cam);  // {N,G}
    row0 = torch::stack({fx_div_z, zeros, -fx_x_over_z2}, -1);  // {N,G,3}
    row1 = torch::stack({zeros, -fy_div_z, fy_y_over_z2}, -1);  // {N,G,3}
    J = torch::stack({row0, row1}, -2);  // {N,G,2,3}

    cov2d = torch::matmul(torch::matmul(J, cov_cam), J.transpose(-1, -2));  // {N,G,2,2}
    eye2 = torch::eye(2).view({1, 1, 2, 2}).to(device);  // {1,1,2,2}
    cov2d = 0.5 * (cov2d + cov2d.transpose(-1, -2)) + eye2 * 1e-4;  // {N,G,2,2}
    cov_xx = cov2d.index({Slice(), Slice(), 0, 0});  // {N,G}
    cov_xy = cov2d.index({Slice(), Slice(), 0, 1});  // {N,G}
    cov_yy = cov2d.index({Slice(), Slice(), 1, 1});  // {N,G}
    CHECK_TENSOR("cov2d", cov2d, ofs);

    det = cov_xx * cov_yy - cov_xy * cov_xy;  // {N,G}
    det = det + 1e-6;  // {N,G}
    CHECK_TENSOR("cov_xx", cov_xx, ofs);
    CHECK_TENSOR("cov_yy", cov_yy, ofs);
    CHECK_TENSOR("cov_xy", cov_xy, ofs);
    CHECK_TENSOR("cov_xx * cov_yy", cov_xx * cov_yy, ofs);
    CHECK_TENSOR("cov_xy * cov_xy", cov_xy * cov_xy, ofs);
    CHECK_TENSOR("det", det, ofs);
    inv_det = 1.0 / det;  // {N,G}
    CHECK_TENSOR("inv_det", inv_det, ofs);
    inv_xx = cov_yy * inv_det;  // {N,G}
    CHECK_TENSOR("inv_xx", inv_xx, ofs);
    inv_xy = -cov_xy * inv_det;  // {N,G}
    inv_yy = cov_xx * inv_det;  // {N,G}
    normalization = torch::sqrt(inv_det);  // {N,G}
    CHECK_TENSOR("normalization", normalization, ofs);

    grid_xv = grid_x.view({1, 1, hw});  // {1,1,H*W}
    grid_yv = grid_y.view({1, 1, hw});  // {1,1,H*W}

    dx = grid_xv - mean_x.unsqueeze(-1);  // {N,G,H*W}
    dy = grid_yv - mean_y.unsqueeze(-1);  // {N,G,H*W}
    CHECK_TENSOR("dx", dx, ofs);
    CHECK_TENSOR("dy", dy, ofs);
    exponent = -0.5 * (inv_xx.unsqueeze(-1) * dx.square() + 2.0 * inv_xy.unsqueeze(-1) * dx * dy + inv_yy.unsqueeze(-1) * dy.square());  // {N,G,H*W}
    CHECK_TENSOR("exponent", exponent, ofs);
    exponent = exponent.clamp(-30.0, 30.0);  // {N,G,H*W}
    CHECK_TENSOR("exponent", exponent, ofs);
    gaussian = torch::exp(exponent) * normalization.unsqueeze(-1);  // {N,G,H*W}
    CHECK_TENSOR("gaussian", gaussian, ofs);

    alpha = opacity.view({1, -1, 1}) * gaussian;  // {N,G,H*W}
    alpha = alpha * valid.unsqueeze(-1).to(torch::kFloat);  // {N,G,H*W}

    sort_key = torch::where(valid, zs_cam, torch::full_like(zs_cam, 1e6));  // {N,G}
    indices = std::get<1>(torch::sort(sort_key, 1, /*descending=*/false));  // {N,G}

    idx_alpha = indices.unsqueeze(-1).expand({N, (long int)this->num_gaussians, hw});  // {N,G} ===> {N,G,1} ===> {N,G,H*W}
    idx_color = indices.unsqueeze(-1).expand({N, (long int)this->num_gaussians, 3});  // {N,G} ===> {N,G,1} ===> {N,G,3}

    alpha = alpha.gather(1, idx_alpha);  // {N,G,H*W}
    CHECK_TENSOR("alpha", alpha, ofs);
    color_sorted = cs.unsqueeze(0).expand({N, (long int)this->num_gaussians, 3}).gather(1, idx_color);  // {G,3} ===> {1,G,3} ===> {N,G,3} ===> {N,G,3}

    cumprod = (1.0 - alpha).cumprod(/*dim=*/1);  // {N,G,H*W}
    T = torch::ones_like(alpha);  // {N,G,H*W}
    T.index_put_({Slice(), Slice(1, alpha.size(1)), Slice()}, cumprod.index({Slice(), Slice(0, alpha.size(1) - 1), Slice()}));  // {N,G,H*W}
    weight = alpha * T;
    rgb = torch::bmm(weight.transpose(1, 2), color_sorted);
    CHECK_TENSOR("rgb", rgb, ofs);

    trans = (1.0 - alpha).prod(/*dim=*/1);  // {N,H*W}
    bg = torch::sigmoid(this->background_logit.to(device));  // {3}
    rgb = rgb + trans.unsqueeze(-1) * bg.view({1, 1, 3});  // {N,H*W,3}

    rgb = rgb.view({N, (long int)this->size, (long int)this->size, 3});  // {N,H*W,3} ===> {N,H,W,3}
    rgb = rgb.permute({0, 3, 1, 2}).contiguous();  // {N,H,W,3} ===> {N,3,H,W}
    CHECK_TENSOR("rgb", rgb, ofs);

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



