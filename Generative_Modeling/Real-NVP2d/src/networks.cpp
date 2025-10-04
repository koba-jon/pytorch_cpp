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
// struct{RealNVP_CouplingLayerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
RealNVP_CouplingLayerImpl::RealNVP_CouplingLayerImpl(long int dim_, long int h_dim_){

    this->dim = dim_;
    this->h_dim = h_dim_;
    this->split = this->dim / 2;

    this->s_net = nn::Sequential(
        nn::Linear(this->split, this->h_dim),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Linear(this->h_dim, this->dim - this->split),
        nn::Tanh()
    );
    register_module("s_net", this->s_net);

    this->t_net = nn::Sequential(
        nn::Linear(this->split, this->h_dim),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Linear(this->h_dim, this->dim - this->split)
    );
    register_module("t_net", this->t_net);

    auto linear_s = this->s_net->ptr<nn::LinearImpl>(this->s_net->size() - 2);
    nn::init::constant_(linear_s->weight, 0.0);

    auto linear_t = this->t_net->ptr<nn::LinearImpl>(this->t_net->size() - 1);
    nn::init::constant_(linear_t->weight, 0.0);

}


// ----------------------------------------------------------------------
// struct{RealNVP_CouplingLayerImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> RealNVP_CouplingLayerImpl::forward(torch::Tensor x){

    torch::Tensor x_a, x_b, s, t, y_a, y_b, y, logdet;

    x_a = x.slice(/*dim=*/1, /*start=*/0, /*end=*/this->split);
    x_b = x.slice(/*dim=*/1, /*start=*/this->split, /*end=*/this->dim);

    s = this->s_net->forward(x_a);
    t = this->t_net->forward(x_a);

    y_a = x_a;
    y_b = x_b * torch::exp(s) + t;
    y = torch::cat({y_a, y_b}, 1);

    logdet = s.sum(1);

    return {y, logdet};

}


// ----------------------------------------------------------------------
// struct{RealNVP_CouplingLayerImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor RealNVP_CouplingLayerImpl::inverse(torch::Tensor y){

    torch::Tensor y_a, y_b, s, t, x_a, x_b, x;

    y_a = y.slice(/*dim=*/1, /*start=*/0, /*end=*/this->split);
    y_b = y.slice(/*dim=*/1, /*start=*/this->split, /*end=*/this->dim);

    s = this->s_net->forward(y_a);
    t = this->t_net->forward(y_a);

    x_a = y_a;
    x_b = (y_b - t) * torch::exp(-s);
    x = torch::cat({x_a, x_b}, 1);

    return x;

}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
NormalizingFlowImpl::NormalizingFlowImpl(po::variables_map &vm){
    for (size_t i = 0; i < vm["n_flow"].as<size_t>(); i++){
        this->model->push_back(RealNVP_CouplingLayer(vm["nc"].as<size_t>() * vm["size"].as<size_t>() * vm["size"].as<size_t>(), vm["h_dim"].as<size_t>()));
    }
    register_module("Normalizing_Flow", this->model);
}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor NormalizingFlowImpl::swap(torch::Tensor z, long int split){
    torch::Tensor a, b, out;
    a = z.slice(/*dim=*/1, /*start=*/0, /*end=*/split);
    b = z.slice(/*dim=*/1, /*start=*/split, /*end=*/z.size(1));
    out = torch::cat({b, a}, 1);
    return out;
}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> NormalizingFlowImpl::forward(torch::Tensor z){

    constexpr float eps = 1e-2;

    torch::Tensor z_, sum_logdet, z_new, logdet;

    z_ = eps + (1.0 - 2.0 * eps) * z;
    z = torch::log(z_) - torch::log1p(-z_);

    sum_logdet = (std::log(1.0 - 2.0 * eps) - torch::log(z_) - torch::log1p(-z_)).sum(1);
    for (size_t i = 0; i < this->model->size(); i++){
        if (i % 2 == 1) z = this->swap(z, z.size(1) / 2);
        std::tie(z_new, logdet) = this->model[i]->as<RealNVP_CouplingLayer>()->forward(z);
        z = z_new;
        sum_logdet = sum_logdet + logdet;
        if (i % 2 == 1) z = this->swap(z, z.size(1) - z.size(1) / 2);
    }

    return {z, sum_logdet};

}


// ----------------------------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module) -> function{inverse}
// ----------------------------------------------------------------------
torch::Tensor NormalizingFlowImpl::inverse(torch::Tensor z){

    constexpr float eps = 1e-2;

    for (long int i = this->model->size() - 1; i >= 0; i--){
        if (i % 2 == 1) z = this->swap(z, z.size(1) / 2);
        z = this->model[i]->as<RealNVP_CouplingLayer>()->inverse(z);
        if (i % 2 == 1) z = this->swap(z, z.size(1) - z.size(1) / 2);
    }

    z = torch::sigmoid(z);
    z = (z - eps) / (1.0 - 2.0 * eps);
    z = z.clamp(0.0, 1.0);

    return z;

}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    return;
}
