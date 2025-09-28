#include <vector>
#include <tuple>
#include <typeinfo>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;
namespace po = boost::program_options;


// ----------------------------------------------------------------------
// struct{RadialFlowImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
RadialFlowImpl::RadialFlowImpl(long int dim){
    this->c = register_parameter("c", torch::zeros({dim}));
    this->a = register_parameter("a", torch::zeros({}));
    this->b = register_parameter("b", torch::zeros({}));
}


// ----------------------------------------------------------------------
// struct{RadialFlowImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> RadialFlowImpl::forward(torch::Tensor z){

    constexpr float eps = 1e-8;
    
    torch::Tensor alpha, beta, delta, r, h, scale, z_new, term1, term2, logdet;

    alpha = torch::softplus(this->a) + eps;
    beta = -alpha + torch::softplus(this->b);
    delta = z - this->c.unsqueeze(0);  // {N,D}
    r = delta.norm(2, /*dim=*/1);  // {N}
    h = 1.0 / (alpha + r);  // {N}
    scale = 1.0 + beta * h;  // {N}

    z_new = z + (beta * h).unsqueeze(1) * delta;  // {N,D}

    term1 = (this->c.size(0) - 1) * torch::log(scale + eps);  // {N}
    term2 = torch::log(scale - beta * r * h * h + eps);  // {N}
    logdet = term1 + term2;  // {N}

    return {z_new, logdet};

}


// ----------------------------------------------------------------------
// struct{RadialFlowImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor RadialFlowImpl::inverse(torch::Tensor z){

    constexpr long int max_iters = 50;
    constexpr float eps = 1e-8;

    torch::Tensor alpha, beta, delta_p, r_p, r, denom, f, fp, delta, factor, z_new;
    torch::NoGradGuard no_grad;

    alpha = torch::softplus(this->a) + eps;
    beta = -alpha + torch::softplus(this->b);
    delta_p = (z - this->c.unsqueeze(0)); // {N,D}
    r_p = delta_p.norm(2, /*dim=*/1);  // {N}

    r = r_p;
    for (long int i = 0; i < max_iters; i++){
        denom = (alpha + r);
        f = r + beta * r / denom - r_p;
        fp = 1.0 + beta * alpha / (denom * denom);
        delta = (f / fp).clamp(-0.5, 0.5);
        r = (r - delta).clamp_min(0.0);
        if (torch::max(torch::abs(delta)).item<float>() < eps) break;
    }

    factor = (r / (r_p + eps)).unsqueeze(1);  // {N,1}
    z_new = this->c.unsqueeze(0) + factor * delta_p;  // {N,D}

    return z_new;

}


// ----------------------------------------------------------------------
// struct{MaskedConv2dImpl}(nn::Module) -> function{pretty_print}
// ----------------------------------------------------------------------
void RadialFlowImpl::pretty_print(std::ostream& stream) const{
    stream << "RadialFlow(dim=" << this->c.size(0) << ")";
    return;
}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
NormalizingFlowImpl::NormalizingFlowImpl(po::variables_map &vm){
    for (size_t i = 0; i < vm["n_flow"].as<size_t>(); i++){
        this->model->push_back(RadialFlow(vm["nc"].as<size_t>() * vm["size"].as<size_t>() * vm["size"].as<size_t>()));
    }
    register_module("Normalizing_Flow", this->model);
}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> NormalizingFlowImpl::forward(torch::Tensor z){

    torch::Tensor sum_logdet, z_new, logdet;

    sum_logdet = torch::zeros({z.size(0)}).to(z.device());
    for (size_t i = 0; i < this->model->size(); i++){
        std::tie(z_new, logdet) = this->model[i]->as<RadialFlow>()->forward(z);
        z = z_new;
        sum_logdet = sum_logdet + logdet;
    }

    return {z, sum_logdet};

}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor NormalizingFlowImpl::inverse(torch::Tensor z){
    for (long int i = this->model->size() - 1; i >= 0; i--){
        z = this->model[i]->as<RadialFlow>()->inverse(z);
    }
    return z;
}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}
