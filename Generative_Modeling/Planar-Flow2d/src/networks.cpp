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
// struct{PlanarFlowImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
PlanarFlowImpl::PlanarFlowImpl(long int dim){
    this->u = register_parameter("u", torch::randn({dim}) * 0.01);
    this->w = register_parameter("w", torch::randn({dim}) * 0.01);
    this->b = register_parameter("b", torch::zeros({}));
}


// ----------------------------------------------------------------------
// struct{PlanarFlowImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> PlanarFlowImpl::forward(torch::Tensor z){

    constexpr float eps = 1e-8;
    
    torch::Tensor wu, uh, lin, h, z_new, hp, logdet;

    wu = torch::dot(this->w, this->u);  // {}
    uh = this->u + ((-1.0 + torch::softplus(wu) - wu) * this->w) / (torch::dot(this->w, this->w) + eps);  // {D}
    lin = torch::matmul(z, this->w) + this->b;  // {N}
    h = torch::tanh(lin);  // {N}
    z_new = z + uh.unsqueeze(0) * h.unsqueeze(1);  // {N,D}

    hp = 1.0 - torch::tanh(lin).pow(2.0);  // {N}
    logdet = torch::log(torch::abs(1.0 + hp * torch::dot(this->w, uh)) + eps);  // {N}

    return {z_new, logdet};

}


// ----------------------------------------------------------------------
// struct{MaskedConv2dImpl}(nn::Module) -> function{pretty_print}
// ----------------------------------------------------------------------
void PlanarFlowImpl::pretty_print(std::ostream& stream) const{
    stream << "PlanarFlow(dim=" << this->u.size(0) << ")";
    return;
}


// ----------------------------------------------------------------------
// struct{PlanarFlowImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor PlanarFlowImpl::inverse(torch::Tensor z){

    constexpr long int max_iters = 50;
    constexpr float eps = 1e-8;

    torch::Tensor wu, uh, s, c, a, h, hp, f, fp, delta, z_new;
    torch::NoGradGuard no_grad;

    wu = torch::dot(this->w, this->u);  // {}
    uh = this->u + ((-1.0 + torch::softplus(wu) - wu) * this->w) / (torch::dot(this->w, this->w) + eps);  // {D}
    s = torch::dot(this->w, uh);  // {}
    c = torch::matmul(z, this->w) + this->b;  // {N}

    a = c.clamp(-3.0, 3.0);  // {N}
    for (long int i = 0; i < max_iters; i++){
        h = torch::tanh(a);  // {N}
        hp = 1.0 - h.pow(2.0);  // {N}
        f = a + s * h - c;  // {N}
        fp = (1.0 + s * hp).clamp_min(1e-4);  // {N}
        delta = (f / fp).clamp(-0.5, 0.5);  // {N}
        a = (a - delta).clamp(-10.0, 10.0);  // {N}
        if (torch::max(torch::abs(delta)).item<float>() < eps) break;
    }

    z_new = z - uh.unsqueeze(0) * torch::tanh(a).unsqueeze(1);  // {N,D}

    return z_new;

}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
NormalizingFlowImpl::NormalizingFlowImpl(po::variables_map &vm){
    for (size_t i = 0; i < vm["n_flow"].as<size_t>(); i++){
        this->model->push_back(PlanarFlow(vm["nc"].as<size_t>() * vm["size"].as<size_t>() * vm["size"].as<size_t>()));
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
        std::tie(z_new, logdet) = this->model[i]->as<PlanarFlow>()->forward(z);
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
        z = this->model[i]->as<PlanarFlow>()->inverse(z);
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
