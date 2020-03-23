#include <typeinfo>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
using namespace torch;


// ----------------------------------------------------------------------
// struct{ConvolutionalAutoEncoderImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ConvolutionalAutoEncoderImpl::ConvolutionalAutoEncoderImpl(po::variables_map &vm){
    
    size_t feature = vm["nf"].as<size_t>();

    DownSampling(this->encoder, vm["nc"].as<size_t>(), feature, /*BN=*/false, /*ReLU=*/true);     // {C,256,256} ===> {F,128,128}
    DownSampling(this->encoder, feature, feature*2, /*BN=*/true, /*ReLU=*/true);                  // {F,128,128} ===> {2F,64,64}
    DownSampling(this->encoder, feature*2, feature*4, /*BN=*/true, /*ReLU=*/true);                // {2F,64,64}  ===> {4F,32,32}
    DownSampling(this->encoder, feature*4, feature*8, /*BN=*/true, /*ReLU=*/true);                // {4F,32,32}  ===> {8F,16,16}
    DownSampling(this->encoder, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);                // {8F,16,16}  ===> {8F,8,8}
    DownSampling(this->encoder, feature*8, vm["nz"].as<size_t>(), /*BN=*/false, /*ReLU=*/false);  // {8F,8,8}    ===> {Z,4,4}
    register_module("encoder", this->encoder);

    UpSampling(this->decoder, vm["nz"].as<size_t>(), feature*8, /*BN=*/true, /*ReLU=*/true);      // {Z,4,4}     ===> {8F,8,8}
    UpSampling(this->decoder, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);                  // {8F,8,8}    ===> {8F,16,16}
    UpSampling(this->decoder, feature*8, feature*4, /*BN=*/true, /*ReLU=*/true);                  // {8F,16,16}  ===> {4F,32,32}
    UpSampling(this->decoder, feature*4, feature*2, /*BN=*/true, /*ReLU=*/true);                  // {4F,32,32}  ===> {2F,64,64}
    UpSampling(this->decoder, feature*2, feature, /*BN=*/true, /*ReLU=*/true);                    // {2F,64,64}  ===> {F,128,128}
    UpSampling(this->decoder, feature, vm["nc"].as<size_t>(), /*BN=*/false, /*ReLU=*/false);      // {F,128,128} ===> {C,256,256}
    this->decoder->push_back(nn::Tanh());
    register_module("decoder", this->decoder);

}


// ----------------------------------------------------------------------
// struct{ConvolutionalAutoEncoderImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor ConvolutionalAutoEncoderImpl::forward(torch::Tensor x){
    torch::Tensor z = this->encoder->forward(x);    // {C,256,256} ===> {Z,4,4}
    torch::Tensor out = this->decoder->forward(z);  // {Z,4,4} ===> {C,256,256}
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