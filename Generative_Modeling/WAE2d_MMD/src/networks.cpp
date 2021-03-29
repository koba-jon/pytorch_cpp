#include <iostream>
#include <vector>
#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;


// ---------------------------------------------------
// struct{WAE_EncoderImpl}(nn::Module) -> constructor
// ---------------------------------------------------
WAE_EncoderImpl::WAE_EncoderImpl(po::variables_map &vm){
    
    size_t feature = vm["nf"].as<size_t>();
    size_t ns = vm["size"].as<size_t>() / (size_t)(std::pow(2, 6));

    DownSampling(this->model, vm["nc"].as<size_t>(), feature, /*BN=*/false, /*ReLU=*/true);  // {C,256,256} ===> {F,128,128}
    DownSampling(this->model, feature, feature*2, /*BN=*/true, /*ReLU=*/true);               // {F,128,128} ===> {2F,64,64}
    DownSampling(this->model, feature*2, feature*4, /*BN=*/true, /*ReLU=*/true);             // {2F,64,64}  ===> {4F,32,32}
    DownSampling(this->model, feature*4, feature*8, /*BN=*/true, /*ReLU=*/true);             // {4F,32,32}  ===> {8F,16,16}
    DownSampling(this->model, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);             // {8F,16,16}  ===> {8F,8,8}
    DownSampling(this->model, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);             // {8F,8,8}    ===> {8F,4,4}
    this->model->push_back(ViewImpl({-1, (long int)(feature*8*ns*ns)}));                     // {8F,4,4}    ===> {8F*4*4}
    this->model->push_back(nn::Linear(feature*8*ns*ns, vm["nz"].as<size_t>()));              // {8F*4*4}    ===> {Z}
    register_module("Encoder", this->model);

}


// ---------------------------------------------------------
// struct{WAE_EncoderImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor WAE_EncoderImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);    // {C,256,256} ===> {Z}
    return out;
}


// ---------------------------------------------------
// struct{WAE_DecoderImpl}(nn::Module) -> constructor
// ---------------------------------------------------
WAE_DecoderImpl::WAE_DecoderImpl(po::variables_map &vm){
    
    size_t feature = vm["nf"].as<size_t>();
    size_t ns = vm["size"].as<size_t>() / (size_t)(std::pow(2, 6));

    this->model->push_back(nn::Linear(vm["nz"].as<size_t>(), feature*8*ns*ns));                 // {Z}         ===> {8F*4*4}
    this->model->push_back(ViewImpl({-1, (long int)(feature*8), (long int)ns, (long int)ns}));  // {8F*4*4}    ===> {8F,4,4}
    UpSampling(this->model, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);                  // {8F,4,4}    ===> {8F,8,8}
    UpSampling(this->model, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);                  // {8F,8,8}    ===> {8F,16,16}
    UpSampling(this->model, feature*8, feature*4, /*BN=*/true, /*ReLU=*/true);                  // {8F,16,16}  ===> {4F,32,32}
    UpSampling(this->model, feature*4, feature*2, /*BN=*/true, /*ReLU=*/true);                  // {4F,32,32}  ===> {2F,64,64}
    UpSampling(this->model, feature*2, feature, /*BN=*/true, /*ReLU=*/true);                    // {2F,64,64}  ===> {F,128,128}
    UpSampling(this->model, feature, vm["nc"].as<size_t>(), /*BN=*/false, /*ReLU=*/false);      // {F,128,128} ===> {C,256,256}
    this->model->push_back(nn::Tanh());                                                         // [-inf,+inf] ===> [-1,1]
    register_module("Decoder", this->model);

}


// ---------------------------------------------------------
// struct{WAE_DecoderImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor WAE_DecoderImpl::forward(torch::Tensor z){
    torch::Tensor out = this->model->forward(z);  // {Z} ===> {C,256,256}
    return out;
}


// --------------------------------------------
// struct{ViewImpl}(nn::Module) -> constructor
// --------------------------------------------
ViewImpl::ViewImpl(std::vector<long int> shape_){
    this->shape = shape_;
}


// --------------------------------------------------
// struct{ViewImpl}(nn::Module) -> function{forward}
// --------------------------------------------------
torch::Tensor ViewImpl::forward(torch::Tensor x){
    return x.view(this->shape);
}


// --------------------------------------------------------
// struct{ViewImpl}(nn::Module) -> function{pretty_print}
// --------------------------------------------------------
void ViewImpl::pretty_print(std::ostream& stream) const{
    size_t shape_size = this->shape.size();
    stream << "View(shape=[";
    for (size_t i = 0; i < shape_size - 1; i++){
        stream << this->shape.at(i) << ", ";
    }
    stream << this->shape.at(shape_size - 1) << "])";
    return;
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
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
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


// ----------------------------
// function{DownSampling}
// ----------------------------
void DownSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias){
    sq->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 4).stride(2).padding(1).bias(bias)));
    if (BN){
        sq->push_back(nn::BatchNorm2d(out_nc));
    }
    if (ReLU){
        sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    return;
}


// ----------------------------
// function{UpSampling}
// ----------------------------
void UpSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias){
    sq->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(in_nc, out_nc, 4).stride(2).padding(1).bias(bias)));
    if (BN){
        sq->push_back(nn::BatchNorm2d(out_nc));
    }
    if (ReLU){
        sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    return;
}