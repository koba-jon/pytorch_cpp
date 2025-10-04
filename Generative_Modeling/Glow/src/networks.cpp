#include <vector>
#include <tuple>
#include <algorithm>
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
namespace po = boost::program_options;
using Slice = torch::indexing::Slice;


// ----------------------------------------------------------------------
// struct{ActNorm2dImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ActNorm2dImpl::ActNorm2dImpl(long int dim){
    this->initialized = false;
    this->scale = register_parameter("scale", torch::ones({1, dim, 1, 1}));
    this->bias = register_parameter("bias", torch::zeros({1, dim, 1, 1}));
}


// ----------------------------------------------------------------------
// struct{ActNorm2dImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> ActNorm2dImpl::forward(torch::Tensor x){

    torch::Tensor mean, std, y, ld_per_channel, logdet;

    if (!this->initialized){
        mean = x.mean({0, 2, 3}, /*keepdim=*/true);  // {1,C,1,1}
        std = x.std({0, 2, 3}, /*unbiased=*/false, /*keepdim=*/true);  // {1,C,1,1}
        this->bias.set_data((-mean).detach());
        this->scale.set_data((1.0 / (std + 1e-6)).detach());
        this->initialized = true;
    }

    y = (x + this->bias) * this->scale;  // {N,C,H,W}
    ld_per_channel = torch::log(torch::abs(this->scale)).view({-1});  // {C}
    logdet = (ld_per_channel.sum() * x.size(2) * x.size(3)).expand({x.size(0)});  // {N}

    return {y, logdet};

}


// ----------------------------------------------------------------------
// struct{ActNorm2dImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor ActNorm2dImpl::inverse(torch::Tensor y){
    return y / this->scale - this->bias;
}


// ----------------------------------------------------------------------
// struct{InvConv1x1Impl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
InvConv1x1Impl::InvConv1x1Impl(long int dim){

    this->C = dim;
    torch::Tensor weight, q, LU, pivots, Pmat, Lmat, Umat, w_s_diag, U_strict, onesCC, u_mask_, l_mask_, eyeC;
    std::tuple<torch::Tensor, torch::Tensor> qr;

    // (1) Build Q with QR
    weight = torch::randn({this->C, this->C});
    qr = torch::linalg_qr(weight, /*mode=*/"reduced");
    q = std::get<0>(qr);

    // (2) Build P, L and U with LU decomposition of Q
    std::tie(LU, pivots) = torch::linalg_lu_factor(q);
    std::tie(Pmat, Lmat, Umat) = torch::lu_unpack(LU, pivots);

    // (3) Build mask and parameter
    w_s_diag = Umat.diag();
    U_strict = torch::triu(Umat, /*diagonal=*/1);
    onesCC = torch::ones_like(Umat);
    u_mask_ = torch::triu(onesCC, 1);
    l_mask_ = u_mask_.transpose(0, 1);
    eyeC = torch::eye(this->C);

    this->w_p = register_buffer("w_p", Pmat.detach());
    this->u_mask = register_buffer("u_mask", u_mask_.detach());
    this->l_mask = register_buffer("l_mask", l_mask_.detach());
    this->s_sign = register_buffer("s_sign", torch::sign(w_s_diag).detach());
    this->l_eye = register_buffer("l_eye", eyeC.detach());

    this->w_l = register_parameter("w_l", Lmat.detach());
    this->w_s = register_parameter("w_s", torch::log(torch::abs(w_s_diag)).detach());
    this->w_u = register_parameter("w_u", U_strict.detach());

}


// ----------------------------------------------------------------------
// struct{InvConv1x1Impl}(nn::Module) -> function{build_W}
// ----------------------------------------------------------------------
torch::Tensor InvConv1x1Impl::build_weight(){
    torch::Tensor L, U, W;
    L = this->w_l * this->l_mask + this->l_eye;
    U = (this->w_u * this->u_mask) + torch::diag(this->s_sign * torch::exp(this->w_s));
    W = torch::matmul(this->w_p, torch::matmul(L, U)).unsqueeze(2).unsqueeze(3);
    return W;
}


// ----------------------------------------------------------------------
// struct{InvConv1x1Impl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> InvConv1x1Impl::forward(torch::Tensor x){

    torch::Tensor weight, y, logdet;

    weight = this->build_weight();
    y = F::conv2d(x, weight);
    logdet = (this->w_s.sum() * x.size(2) * x.size(3)).expand({x.size(0)});

    return {y, logdet};

}


// ----------------------------------------------------------------------
// struct{InvConv1x1Impl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor InvConv1x1Impl::inverse(torch::Tensor y){

    torch::Tensor weight, invW, x;

    weight = this->build_weight();
    invW = weight.squeeze().inverse().unsqueeze(2).unsqueeze(3);
    x = F::conv2d(y, invW);

    return x;

}


// ----------------------------------------------------------------------
// struct{ZeroConv2dImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ZeroConv2dImpl::ZeroConv2dImpl(long int in_nc, long int out_nc){

    this->conv = nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 3).stride(1).padding(0));
    nn::init::constant_(this->conv->weight, 0.0);
    nn::init::constant_(this->conv->bias, 0.0);
    register_module("conv", this->conv);

    this->scale = register_parameter("scale", torch::zeros({1, out_nc, 1, 1}));

}


// ----------------------------------------------------------------------
// struct{ZeroConv2dImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor ZeroConv2dImpl::forward(torch::Tensor x){

    torch::Tensor out;

    out = F::pad(x, F::PadFuncOptions({1, 1, 1, 1}).value(1));
    out = this->conv->forward(out);
    out = out * torch::exp(this->scale * 3.0);

    return out;

}


// ----------------------------------------------------------------------
// struct{CouplingLayerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
CouplingLayerImpl::CouplingLayerImpl(long int dim, long int h_dim){

    this->net = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(dim / 2, h_dim, 3).stride(1).padding(1)),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(h_dim, h_dim, 1)),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        ZeroConv2d(h_dim, dim / 2)
    );
    register_module("net", this->net);

    auto conv1 = this->net->ptr<nn::Conv2dImpl>(0);
    nn::init::normal_(conv1->weight, 0.0, 0.05);
    nn::init::zeros_(conv1->bias);

    auto conv2 = this->net->ptr<nn::Conv2dImpl>(2);
    nn::init::normal_(conv2->weight, 0.0, 0.05);
    nn::init::zeros_(conv2->bias);

}


// ----------------------------------------------------------------------
// struct{CouplingLayerImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor CouplingLayerImpl::forward(torch::Tensor x){

    std::vector<torch::Tensor> x_ab;
    torch::Tensor x_a, x_b, t, y_b, y;

    x_ab = x.chunk(2, 1);
    x_a = x_ab[0];
    x_b = x_ab[1];

    t = this->net->forward(x_a);
    y_b = x_b + t;
    y = torch::cat({x_a, y_b}, 1);

    return y;

}


// ----------------------------------------------------------------------
// struct{CouplingLayerImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor CouplingLayerImpl::inverse(torch::Tensor y){

    std::vector<torch::Tensor> y_ab;
    torch::Tensor y_a, y_b, t, x_b, x;

    y_ab = y.chunk(2, 1);
    y_a = y_ab[0];
    y_b = y_ab[1];

    t = this->net->forward(y_a);
    x_b = y_b - t;
    x = torch::cat({y_a, x_b}, 1);

    return x;

}


// ----------------------------------------------------------------------
// struct{FlowImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
FlowImpl::FlowImpl(long int in_nc, long int h_dim){

    this->actnorm = ActNorm2d(in_nc);
    register_module("actnorm", this->actnorm);

    this->invconv = InvConv1x1(in_nc);
    register_module("invconv", this->invconv);

    this->coupling = CouplingLayer(in_nc, h_dim);
    register_module("coupling", this->coupling);

}


// ----------------------------------------------------------------------
// struct{FlowImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> FlowImpl::forward(torch::Tensor z){

    torch::Tensor logdet1, logdet2, sum_logdet;

    std::tie(z, logdet1) = this->actnorm->forward(z);
    std::tie(z, logdet2) = this->invconv->forward(z);
    z = this->coupling->forward(z);
    sum_logdet = logdet1 + logdet2;

    return {z, sum_logdet};

}


// ----------------------------------------------------------------------
// struct{FlowImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor FlowImpl::inverse(torch::Tensor z){
    z = this->coupling->inverse(z);
    z = this->invconv->inverse(z);
    z = this->actnorm->inverse(z);
    return z;
}


// ----------------------------------------------------------------------
// struct{BlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
BlockImpl::BlockImpl(long int dim, long int h_dim, long int n_flow, bool split_){

    long int squeeze_dim = dim * 4;
    this->split = split_;

    for (long int i = 0; i < n_flow; i++){
        this->flows->push_back(Flow(squeeze_dim, h_dim));
    }
    register_module("flows", this->flows);

    if (this->split){
        this->prior = ZeroConv2d(dim * 2, dim * 4);
    }
    else {
        this->prior = ZeroConv2d(dim * 4, dim * 8);
    }
    register_module("prior", this->prior);

}


// ----------------------------------------------------------------------
// struct{BlockImpl}(nn::Module) -> function{squeeze2x}
// ----------------------------------------------------------------------
torch::Tensor BlockImpl::squeeze2x(torch::Tensor x){
    torch::Tensor y;
    y = x.view({x.size(0), x.size(1), x.size(2) / 2, 2, x.size(3) / 2, 2});  // {N,C,H,W} ===> {N,C,H/2,2,W/2,2}
    y = y.permute({0, 1, 3, 5, 2, 4}).contiguous();  // {N,C,H/2,2,W/2,2} ===> {N,C,2,2,H/2,W/2}
    y = y.view({x.size(0), x.size(1) * 4, x.size(2) / 2, x.size(3) / 2});  // {N,C,2,2,H/2,W/2} ===> {N,C*4,H/2,W/2}
    return y;
}


// ----------------------------------------------------------------------
// struct{BlockImpl}(nn::Module) -> function{unsqueeze2x}
// ----------------------------------------------------------------------
torch::Tensor BlockImpl::unsqueeze2x(torch::Tensor y){
    torch::Tensor x;
    x = y.view({y.size(0), y.size(1) / 4, 2, 2, y.size(2), y.size(3)});  // {N,C*4,H/2,W/2} ===> {N,C,2,2,H/2,W/2}
    x = x.permute({0, 1, 4, 2, 5, 3}).contiguous();  // {N,C,2,2,H/2,W/2} ===> {N,C,H/2,2,W/2,2}
    x = x.view({y.size(0), y.size(1) / 4, y.size(2) * 2, y.size(3) * 2});  // {N,C,H/2,2,W/2,2} ===> {N,C,H,W}
    return x;
}


// ----------------------------------------------------------------------
// struct{BlockImpl}(nn::Module) -> function{gaussian_log_p}
// ----------------------------------------------------------------------
torch::Tensor BlockImpl::gaussian_log_p(torch::Tensor x, torch::Tensor mean, torch::Tensor log_sd){
    return -0.5 * std::log(2.0 * M_PI) - log_sd - 0.5 * (x - mean) * (x - mean) / torch::exp(2.0 * log_sd);
}


// ----------------------------------------------------------------------
// struct{BlockImpl}(nn::Module) -> function{gaussian_sample}
// ----------------------------------------------------------------------
torch::Tensor BlockImpl::gaussian_sample(torch::Tensor eps, torch::Tensor mean, torch::Tensor log_sd){
    return mean + torch::exp(log_sd) * eps;
}


// ----------------------------------------------------------------------
// struct{BlockImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BlockImpl::forward(torch::Tensor x){

    std::vector<torch::Tensor> yz, y_prior;
    torch::Tensor y, y_new, logdet, sum_logdet, z_new, mean, log_sd, log_p, zero;

    y = this->squeeze2x(x);
    sum_logdet = torch::zeros({x.size(0)}).to(x.device());
    for (long int i = 0; i < (long int)this->flows->size(); i++){
        std::tie(y_new, logdet) = this->flows[i]->as<Flow>()->forward(y);
        y = y_new;
        sum_logdet = sum_logdet + logdet;
    }

    if (this->split){
        yz = y.chunk(2, 1);
        y = yz[0];
        z_new = yz[1];
        y_prior = this->prior->forward(y).chunk(2, 1);
        mean = y_prior[0];
        log_sd = y_prior[1];
        log_p = this->gaussian_log_p(z_new, mean, log_sd).view({x.size(0), -1}).sum(1);
    }
    else{
        zero = torch::zeros_like(y);
        y_prior = this->prior->forward(zero).chunk(2, 1);
        mean = y_prior[0];
        log_sd = y_prior[1];
        log_p = this->gaussian_log_p(y, mean, log_sd).view({x.size(0), -1}).sum(1);
        z_new = y;
    }

    return  {y, sum_logdet, log_p, z_new};

}


// ----------------------------------------------------------------------
// struct{BlockImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor BlockImpl::inverse(torch::Tensor y, torch::Tensor eps){

    torch::Tensor x, mean, log_sd, z, zero;
    std::vector<torch::Tensor> x_prior;

    x = y;
    if (this->split){
        x_prior = this->prior->forward(x).chunk(2, 1);
        mean = x_prior[0];
        log_sd = x_prior[1];
        z = this->gaussian_sample(eps, mean, log_sd);
        x = torch::cat({y, z}, 1);
    }
    else{
        zero = torch::zeros_like(x);
        x_prior = this->prior->forward(zero).chunk(2, 1);
        mean = x_prior[0];
        log_sd = x_prior[1];
        z = this->gaussian_sample(eps, mean, log_sd);
        x = z;
    }

    for (long int i = this->flows->size() - 1; i >= 0; i--){
        x = this->flows[i]->as<Flow>()->inverse(x);
    }
    x = this->unsqueeze2x(x);

    return x;

}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
NormalizingFlowImpl::NormalizingFlowImpl(po::variables_map &vm){

    long int dim;

    dim = vm["nc"].as<size_t>();
    for (size_t i = 0; i < vm["n_block"].as<size_t>() - 1; i++){
        this->blocks->push_back(Block(dim, vm["h_dim"].as<size_t>(), vm["n_flow"].as<size_t>(), /*split=*/true));
        dim *= 2;
    }
    this->blocks->push_back(Block(dim, vm["h_dim"].as<size_t>(), vm["n_flow"].as<size_t>(), /*split=*/false));
    register_module("Normalizing_Flow", this->blocks);

}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> NormalizingFlowImpl::forward(torch::Tensor z){

    constexpr float eps = 1e-2;

    torch::Tensor z_, out, logdet, log_p, z_new, sum_logdet, sum_log_p;
    std::vector<torch::Tensor> z_list;

    z_ = eps + (1.0 - 2.0 * eps) * z;
    z = torch::log(z_) - torch::log1p(-z_);

    out = z;
    z_list = std::vector<torch::Tensor>(this->blocks->size());
    sum_logdet = (std::log(1.0 - 2.0 * eps) - torch::log(z_) - torch::log1p(-z_)).flatten(1).sum(1);
    sum_log_p = torch::zeros({z.size(0)}).to(z.device());
    for (long int i = 0; i < (long int)this->blocks->size(); i++){
        std::tie(out, logdet, log_p, z_new) = this->blocks[i]->as<Block>()->forward(out);
        z_list[i] = z_new;
        sum_logdet = sum_logdet + logdet;
        sum_log_p = sum_log_p + log_p;
    }

    return {z_list, sum_logdet, sum_log_p};

}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor NormalizingFlowImpl::inverse(std::vector<torch::Tensor> z_list){

    constexpr float eps = 1e-2;
    
    long int blocks_size;
    torch::Tensor x;

    blocks_size = z_list.size();
    for (long int i = blocks_size - 1; i >= 0; i--){
        if (i == blocks_size - 1){
            x = this->blocks[i]->as<Block>()->inverse(z_list[blocks_size - 1], z_list[blocks_size - 1]);
        }
        else{
            x = this->blocks[i]->as<Block>()->inverse(x, z_list[i]);
        }
    }

    x = torch::sigmoid(x);
    x = (x - eps) / (1.0 - 2.0 * eps);
    x = x.clamp(0.0, 1.0);

    return x;

}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    return;
}
