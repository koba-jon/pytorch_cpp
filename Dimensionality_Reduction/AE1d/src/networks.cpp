#include <vector>
#include <typeinfo>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;


// ----------------------------------------------------------------------
// struct{AutoEncoder1dImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
AutoEncoder1dImpl::AutoEncoder1dImpl(po::variables_map &vm){
    
    constexpr size_t max_downs = 5;
    
    size_t input_dim, z_dim, dim, dim_size;
    std::vector<size_t> dim_list;

    input_dim = vm["nd"].as<size_t>();
    z_dim = vm["nz"].as<size_t>();
    dim = input_dim;
    for (size_t i = 0; i < max_downs; i++){
        dim_list.push_back(dim);
        dim = dim / 3;
        if (dim <= z_dim) break;
    }
    dim_list.push_back(z_dim);
    dim_size = dim_list.size();

    // Encoder
    for (size_t i = 0; i < dim_size - 2; i++){
        LinearLayer(this->encoder, dim_list.at(i), dim_list.at(i + 1), /*ReLU*/true);
    }
    LinearLayer(this->encoder, dim_list.at(dim_size - 2), dim_list.at(dim_size - 1), /*ReLU*/false);
    register_module("encoder", this->encoder);

    // Decoder
    for (size_t i = 0; i < dim_size - 2; i++){
        LinearLayer(this->decoder, dim_list.at(dim_size - i - 1), dim_list.at(dim_size - i - 2), /*ReLU*/true);
    }
    LinearLayer(this->decoder, dim_list.at(1), dim_list.at(0), /*ReLU*/false);
    // this->decoder->push_back(nn::Sigmoid());
    register_module("decoder", this->decoder);

}


// ----------------------------------------------------------------------
// struct{AutoEncoder1dImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor AutoEncoder1dImpl::forward(torch::Tensor x){
    torch::Tensor z = this->encoder->forward(x);    // {D} ===> {Z}
    torch::Tensor out = this->decoder->forward(z);  // {Z} ===> {D}
    return out;
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


// ----------------------------
// function{LinearLayer}
// ----------------------------
void LinearLayer(nn::Sequential &sq, const size_t in_dim, const size_t out_dim, const bool ReLU){
    sq->push_back(nn::Linear(in_dim, out_dim));
    if (ReLU){
        sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    return;
}
