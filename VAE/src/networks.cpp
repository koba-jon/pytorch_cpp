#include <typeinfo>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>
// For Original Header
#include "networks.hpp"

// Define Namespace
using namespace torch;
namespace F = torch::nn::functional;
namespace po = boost::program_options;


// ----------------------------------------------------------------------
// struct{VariationalAutoEncoderImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
VariationalAutoEncoderImpl::VariationalAutoEncoderImpl(po::variables_map &vm){
    
    size_t feature = vm["nf"].as<size_t>();

    DownSampling(this->encoder, vm["nc"].as<size_t>(), feature, /*BN=*/false, /*ReLU=*/true);     // {C,256,256} ===> {F,128,128}
    DownSampling(this->encoder, feature, feature*2, /*BN=*/true, /*ReLU=*/true);                  // {F,128,128} ===> {2F,64,64}
    DownSampling(this->encoder, feature*2, feature*4, /*BN=*/true, /*ReLU=*/true);                // {2F,64,64}  ===> {4F,32,32}
    DownSampling(this->encoder, feature*4, feature*8, /*BN=*/true, /*ReLU=*/true);                // {4F,32,32}  ===> {8F,16,16}
    DownSampling(this->encoder, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);                // {8F,16,16}  ===> {8F,8,8}
    register_module("encoder", this->encoder);

    DownSampling(this->encoder_mean, feature*8, vm["nz"].as<size_t>(), /*BN=*/false, /*ReLU=*/false);  // {8F,8,8}    ===> {Z,4,4}
    DownSampling(this->encoder_var, feature*8, vm["nz"].as<size_t>(), /*BN=*/false, /*ReLU=*/false);   // {8F,8,8}    ===> {Z,4,4}
    register_module("encoder_mean", this->encoder_mean);
    register_module("encoder_var", this->encoder_var);

    UpSampling(this->decoder, vm["nz"].as<size_t>(), feature*8, /*BN=*/true, /*ReLU=*/true);      // {Z,4,4}     ===> {8F,8,8}
    UpSampling(this->decoder, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);                  // {8F,8,8}    ===> {8F,16,16}
    UpSampling(this->decoder, feature*8, feature*4, /*BN=*/true, /*ReLU=*/true);                  // {8F,16,16}  ===> {4F,32,32}
    UpSampling(this->decoder, feature*4, feature*2, /*BN=*/true, /*ReLU=*/true);                  // {4F,32,32}  ===> {2F,64,64}
    UpSampling(this->decoder, feature*2, feature, /*BN=*/true, /*ReLU=*/true);                    // {2F,64,64}  ===> {F,128,128}
    UpSampling(this->decoder, feature, vm["nc"].as<size_t>(), /*BN=*/false, /*ReLU=*/false);      // {F,128,128} ===> {C,256,256}
    this->decoder->push_back(nn::Tanh());                                                         // [-inf,+inf] ===> [-1,1]
    register_module("decoder", this->decoder);

}


// ----------------------------------------------------------------------
// struct{VariationalAutoEncoderImpl}(nn::Module) -> function{sampling}
// ----------------------------------------------------------------------
torch::Tensor VariationalAutoEncoderImpl::sampling(torch::Tensor &mean, torch::Tensor &var){
    torch::Tensor eps = torch::randn({mean.size(0), mean.size(1), mean.size(2), mean.size(3)}).to(mean.device());
    torch::Tensor z = mean + torch::sqrt(var) * eps;
    return z;
}


// ----------------------------------------------------------------------
// struct{VariationalAutoEncoderImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor VariationalAutoEncoderImpl::forward(torch::Tensor x){
    torch::Tensor mid = this->encoder->forward(x);    // {C,256,256} ===> {Z,8,8}
    this->mean_keep = this->encoder_mean->forward(mid);
    this->var_keep = F::softplus(this->encoder_var->forward(mid));
    this->z_keep = this->sampling(this->mean_keep, this->var_keep);
    torch::Tensor out = this->decoder->forward(this->z_keep);  // {Z,4,4} ===> {C,256,256}
    return out;
}


// -----------------------------------------------------------------------------
// struct{VariationalAutoEncoderImpl}(nn::Module) -> function{kld_just_before}
// -----------------------------------------------------------------------------
torch::Tensor VariationalAutoEncoderImpl::kld_just_before(){
    torch::Tensor kld = - 0.5 * torch::mean(1.0 + torch::log(this->var_keep) - this->mean_keep * this->mean_keep - this->var_keep);
    return kld;
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