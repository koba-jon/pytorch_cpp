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
MaskedConv2dImpl::MaskedConv2dImpl(char mask_type, long int in_nc, long int out_nc, long int kernel){
    
    this->padding = kernel / 2;

    this->conv = torch::nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, kernel).bias(false));
    register_module("conv", this->conv);

    this->weight = register_parameter("weight", conv->weight.detach().clone().set_requires_grad(true));

    this->mask = torch::ones_like(this->weight);
    if (mask_type == 'A'){
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


// -----------------------------------------------------------------------------
// struct{MaskedConv2dBlockImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
MaskedConv2dBlockImpl::MaskedConv2dBlockImpl(char mask_type, long int dim){
    this->model->push_back(MaskedConv2d(mask_type, dim, dim, 7));
    this->model->push_back(nn::BatchNorm2d(dim));
    this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    register_module("model", this->model);
}


// -----------------------------------------------------------------------------
// struct{MaskedConv2dBlockImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor MaskedConv2dBlockImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);
    return out;
}


// -----------------------------------------------------------------------------
// struct{PixelCNNImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
PixelCNNImpl::PixelCNNImpl(po::variables_map &vm){

    this->dim = vm["dim_pix"].as<size_t>();

    this->token_emb = nn::Embedding(nn::EmbeddingOptions(vm["K"].as<size_t>(), this->dim));
    register_module("token_emb", this->token_emb);

    this->layers->push_back(MaskedConv2dBlock('A', this->dim));
    for (size_t i = 1; i < vm["n_layers"].as<size_t>(); i++){
        this->layers->push_back(MaskedConv2dBlock('B', this->dim));
    }
    register_module("layers", this->layers);

    this->output_conv = nn::Conv2d(nn::Conv2dOptions(this->dim, vm["K"].as<size_t>(), 1));
    register_module("output_conv", this->output_conv);

}


// -----------------------------------------------------------------------------
// struct{PixelCNNImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor PixelCNNImpl::forward(torch::Tensor x){
    torch::Tensor out;
    x = this->token_emb->forward(x.view({-1})).view({x.size(0), x.size(1), x.size(2), this->dim}).permute({0, 3, 1, 2}).contiguous();
    x = this->layers->forward(x);
    out = this->output_conv->forward(x);
    return out;
}


// ----------------------------------------------------------------------
// struct{VectorQuantizerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
VectorQuantizerImpl::VectorQuantizerImpl(const size_t K, const size_t nz){
    this->e = register_parameter("Embedding Feature", torch::randn({(long int)K, (long int)nz}));
}


// ----------------------------------------------------------------------
// struct{VectorQuantizerImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> VectorQuantizerImpl::forward(torch::Tensor z_e){
    torch::Tensor z_e_flat, dist, idx, z_q, z_q_;
    z_e_flat = z_e.permute({0, 2, 3, 1}).contiguous().view({-1, z_e.size(1)});  // {N,Z,ZH,ZW} ===> {N,ZH,ZW,Z} ===> {N*ZH*ZW,Z}
    dist = torch::sum(z_e_flat.pow(2.0), /*dim=*/1, /*keepdim=*/true) + torch::sum(this->e.pow(2.0), /*dim=*/1).unsqueeze(0) - 2.0 * z_e_flat.mm(this->e.t());  // {N*ZH*ZW,K}
    idx = torch::argmin(dist, /*dim=*/1);  // {N*ZH*ZW,K} ===> {N*ZH*ZW}
    z_q = this->e.index_select(/*dim=*/0, idx).view({z_e.size(0), z_e.size(2), z_e.size(3), z_e.size(1)}).permute({0, 3, 1, 2});  // {N*ZH*ZW,Z} ===> {N,ZH,ZW,Z} ===> {N,Z,ZH,ZW}
    z_q_ = z_e + (z_q - z_e).detach();  // {N,Z,ZH,ZW}
    return {z_q_, z_q, idx.view({z_e.size(0), z_e.size(2), z_e.size(3)})};
}


// ----------------------------------------------------------------------
// struct{ResidualLayerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ResidualLayerImpl::ResidualLayerImpl(const size_t dim, const size_t h_dim){
    this->model = nn::Sequential(
        nn::BatchNorm2d(dim),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(dim, h_dim, 3).stride(1).padding(1).bias(false)),
        nn::BatchNorm2d(h_dim),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(h_dim, dim, 3).stride(1).padding(1).bias(true))
    );
    register_module("Residual Layer", this->model);
}


// ----------------------------------------------------------------------
// struct{ResidualLayerImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor ResidualLayerImpl::forward(torch::Tensor x){
    x = x + this->model->forward(x);
    return x;
}


// ----------------------------------------------------------------------
// struct{VQVAEImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
VQVAEImpl::VQVAEImpl(po::variables_map &vm){
    
    size_t feature = vm["nf"].as<size_t>();

    this->encoder = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(vm["nc"].as<size_t>(), feature, 4).stride(2).padding(1).bias(true)),     // {C,256,256} ===> {F,128,128}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(feature, feature*2, 4).stride(2).padding(1).bias(false)),                // {F,128,128} ===> {2F,64,64}
        nn::BatchNorm2d(feature*2),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(feature*2, vm["nz"].as<size_t>(), 3).stride(1).padding(1).bias(false)),  // {2F,64,64}  ===> {Z,64,64}
        ResidualLayer(vm["nz"].as<size_t>(), vm["res_h"].as<size_t>()),                                       // {Z,64,64}   ===> {Z,64,64}
        ResidualLayer(vm["nz"].as<size_t>(), vm["res_h"].as<size_t>())                                        // {Z,64,64}   ===> {Z,64,64}
    );
    register_module("encoder", this->encoder);

    this->vq = VectorQuantizer(vm["K"].as<size_t>(), vm["nz"].as<size_t>());
    register_module("Vector Quantizer", this->vq);

    this->decoder = nn::Sequential(
        nn::ConvTranspose2d(nn::ConvTranspose2dOptions(vm["nz"].as<size_t>(), feature*2, 3).stride(1).padding(1).bias(false)),  // {Z,64,64}   ===> {2F,64,64}
        ResidualLayer(feature*2, vm["res_h"].as<size_t>()),                                                                     // {Z,64,64}   ===> {Z,64,64}
        ResidualLayer(feature*2, vm["res_h"].as<size_t>()),                                                                     // {Z,64,64}   ===> {Z,64,64}
        nn::BatchNorm2d(feature*2),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::ConvTranspose2d(nn::ConvTranspose2dOptions(feature*2, feature, 4).stride(2).padding(1).bias(false)),                // {2F,64,64}  ===> {F,128,128}
        nn::BatchNorm2d(feature),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::ConvTranspose2d(nn::ConvTranspose2dOptions(feature, vm["nc"].as<size_t>(), 4).stride(2).padding(1).bias(true)),     // {F,128,128} ===> {C,256,256}
        nn::Tanh()                                                                                                              // [-inf,+inf] ===> [-1,1]
    );
    register_module("decoder", this->decoder);

}


// ----------------------------------------------------------------------
// struct{VQVAEImpl}(nn::Module) -> function{forward_z}
// ----------------------------------------------------------------------
torch::Tensor VQVAEImpl::sampling(const std::vector<long int> z_shape, PixelCNN pixelcnn, torch::Device device){

    torch::Tensor idx, logits, probs, sampled, z_q, out;

    idx = torch::zeros({z_shape[0], z_shape[2], z_shape[3]}).to(torch::kLong).to(device);
    for (long int j = 0; j < z_shape[2]; j++){
        for (long int i = 0; i < z_shape[3]; i++){
            logits = pixelcnn->forward(idx);
            probs = torch::softmax(logits.index({Slice(), Slice(), j, i}), /*dim=*/1);
            sampled = probs.multinomial(1).squeeze(1);
            idx.index_put_({Slice(), j, i}, sampled);
        }
    }

    z_q = this->vq->e.index_select(/*dim=*/0, idx.view(-1)).view({z_shape[0], z_shape[2], z_shape[3], z_shape[1]}).permute({0, 3, 1, 2});
    out = this->decoder->forward(z_q);

    return out;

}


// ----------------------------------------------------------------------
// struct{VQVAEImpl}(nn::Module) -> function{forward_z}
// ----------------------------------------------------------------------
torch::Tensor VQVAEImpl::synthesis(torch::Tensor x, torch::Tensor y, const float alpha){
    torch::Tensor zx_e, zy_e, z_e, out;
    zx_e = this->encoder->forward(x);    // {C,256,256} ===> {Z,64,64}
    zy_e = this->encoder->forward(y);    // {C,256,256} ===> {Z,64,64}
    z_e = zx_e * alpha + zy_e * (1.0 - alpha);
    auto [z_q_, z_q, idx] = this->vq->forward(z_e);
    out = this->decoder->forward(z_q_);  // {Z,64,64} ===> {C,256,256}
    return out;
}


// ----------------------------------------------------------------------
// struct{VQVAEImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> VQVAEImpl::forward(torch::Tensor x){
    torch::Tensor z_e = this->encoder->forward(x);     // {C,256,256} ===> {Z,64,64}
    auto [z_q_, z_q, idx] = this->vq->forward(z_e);
    torch::Tensor out = this->decoder->forward(z_q_);  // {Z,64,64} ===> {C,256,256}
    return {out, z_e, z_q};
}


// ----------------------------------------------------------------------
// struct{VQVAEImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor VQVAEImpl::forward_idx(torch::Tensor x){
    torch::Tensor z_e = this->encoder->forward(x);     // {C,256,256} ===> {Z,64,64}
    auto [z_q_, z_q, idx] = this->vq->forward(z_e);
    return idx;
}


// -----------------------------------------------------------------------------
// struct{VQVAEImpl}(nn::Module) -> function{get_z_shape}
// -----------------------------------------------------------------------------
std::vector<long int> VQVAEImpl::get_z_shape(const std::vector<long int> x_shape, torch::Device &device){
    torch::Tensor x = torch::full(x_shape, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    torch::Tensor z = this->encoder->forward(x);    // {C,256,256} ===> {Z,64,64}
    std::vector<long int> z_shape = {z.size(0), z.size(1), z.size(2), z.size(3)};
    return z_shape;
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

