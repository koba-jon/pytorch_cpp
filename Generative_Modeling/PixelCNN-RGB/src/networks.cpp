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
using Slice = torch::indexing::Slice;



// -----------------------------------------------------------------------------
// struct{MaskedConv2dImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
MaskedConv2dImpl::MaskedConv2dImpl(char mask_type_, long int in_nc, long int out_nc, long int kernel){

    this->mask_type = mask_type_;
    this->padding = kernel / 2;

    torch::nn::Conv2d conv = torch::nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, kernel).bias(false));
    this->weight = register_parameter("weight", conv->weight.detach().clone());

    this->mask = torch::ones_like(this->weight);
    if (this->mask_type == 'A'){
        this->mask.index_put_({Slice(), Slice(), kernel / 2, Slice(kernel / 2, torch::indexing::None)}, 0.0);
        this->mask.index_put_({Slice(), Slice(), Slice(kernel / 2 + 1, torch::indexing::None), Slice()}, 0.0);
    }
    else{
        this->mask.index_put_({Slice(), Slice(), kernel / 2, Slice(kernel / 2 + 1, torch::indexing::None)}, 0.0);
        this->mask.index_put_({Slice(), Slice(), Slice(kernel / 2 + 1, torch::indexing::None), Slice()}, 0.0);
    }
    register_buffer("mask", this->mask);

}


// -----------------------------------------------------------------------------
// struct{MaskedConv2dImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor MaskedConv2dImpl::forward(torch::Tensor x){
    torch::Tensor w, out;
    w = this->weight * this->mask;
    out = F::conv2d(x, w, F::Conv2dFuncOptions().stride(1).padding(this->padding));
    return out;
}


// ----------------------------------------------------------------------
// struct{MaskedConv2dImpl}(nn::Module) -> function{pretty_print}
// ----------------------------------------------------------------------
void MaskedConv2dImpl::pretty_print(std::ostream& stream) const{
    stream << "MaskedConv2d(" << this->weight.size(1) << ", " << this->weight.size(0) << ", ";
    stream << "kernel_size=[" << this->weight.size(2) << ", " << this->weight.size(3) << "], ";
    stream << "stride=[1, 1], ";
    stream << "padding=[" << this->padding << ", " << this->padding << "], ";
    stream << "bias=false, ";
    stream << "mask=" << this->mask_type << ")";
    return;
}


// -----------------------------------------------------------------------------
// struct{MaskedConv2dBlockImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
MaskedConv2dBlockImpl::MaskedConv2dBlockImpl(char mask_type, long int dim, bool residual_){
    this->residual = residual_;
    this->model->push_back(MaskedConv2d(mask_type, dim, dim, 7));
    this->model->push_back(nn::BatchNorm2d(dim));
    this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(dim, dim, 1)));
    this->model->push_back(nn::BatchNorm2d(dim));
    this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    register_module("model", this->model);
}


// -----------------------------------------------------------------------------
// struct{MaskedConv2dBlockImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor MaskedConv2dBlockImpl::forward(torch::Tensor x){
    torch::Tensor out = this->residual ? (this->model->forward(x) + x) : this->model->forward(x);
    return out;
}


// -----------------------------------------------------------------------------
// struct{PixelCNNImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
PixelCNNImpl::PixelCNNImpl(po::variables_map &vm){

    this->dim = vm["dim"].as<size_t>();
    this->level = vm["L"].as<size_t>();

    this->layers->push_back(MaskedConv2d('A', 3, this->dim, 7));
    for (size_t i = 1; i < vm["n_layers"].as<size_t>(); i++){
        this->layers->push_back(MaskedConv2dBlock('B', this->dim, true));
    }
    register_module("layers", this->layers);

    this->output_conv = nn::Conv2d(nn::Conv2dOptions(this->dim, 3 * this->level, 1));
    register_module("output_conv", this->output_conv);

}


// ----------------------------------------------------------------------
// struct{PixelCNNImpl}(nn::Module) -> function{forward_z}
// ----------------------------------------------------------------------
torch::Tensor PixelCNNImpl::sampling(const std::vector<long int> x_shape, torch::Device device){

    torch::Tensor out, logits, probs, sampled;

    out = torch::zeros(x_shape).to(device);
    for (long int j = 0; j < x_shape[2]; j++){
        for (long int i = 0; i < x_shape[3]; i++){

            logits = this->forward(out);

            // R
            probs = torch::softmax(logits.index({Slice(), Slice(0, this->level), j, i}), /*dim=*/1);
            sampled = probs.multinomial(1).squeeze(1) / (this->level - 1);
            out.index_put_({Slice(), 0, j, i}, sampled);

            // G
            probs = torch::softmax(logits.index({Slice(), Slice(this->level, 2 * this->level), j, i}), /*dim=*/1);
            sampled = probs.multinomial(1).squeeze(1) / (this->level - 1);
            out.index_put_({Slice(), 1, j, i}, sampled);

            // B
            probs = torch::softmax(logits.index({Slice(), Slice(2 * this->level, 3 * this->level), j, i}), /*dim=*/1);
            sampled = probs.multinomial(1).squeeze(1) / (this->level - 1);
            out.index_put_({Slice(), 2, j, i}, sampled);

        }
    }

    return out;

}


// -----------------------------------------------------------------------------
// struct{PixelCNNImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor PixelCNNImpl::forward(torch::Tensor x){
    torch::Tensor out;
    x = this->layers->forward(x);
    out = this->output_conv->forward(x);
    return out;
}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl)) || (typeid(m) == typeid(nn::ConvTranspose2d)) || (typeid(m) == typeid(nn::ConvTranspose2dImpl))) {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::BatchNorm2d)) || (typeid(m) == typeid(nn::BatchNorm2dImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/1.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}

