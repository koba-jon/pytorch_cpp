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
// struct{GatedActivationImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor GatedActivationImpl::forward(torch::Tensor x){
    std::vector<torch::Tensor> chunks = x.chunk(2, /*dim=*/1);
    torch::Tensor out = torch::tanh(chunks[0]) * torch::sigmoid(chunks[1]);
    return out;
}


// -----------------------------------------------------------------------------
// struct{GatedMaskedConv2dImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
GatedMaskedConv2dImpl::GatedMaskedConv2dImpl(char mask_type, long int dim, long int kernel, bool residual_){

    this->residual = residual_;

    this->vert_stack = nn::Conv2d(nn::Conv2dOptions(dim, 2 * dim, {kernel / 2 + 1, kernel}).stride(1).padding({kernel / 2, kernel / 2}));
    register_module("vert_stack", this->vert_stack);

    this->vert_to_horiz = nn::Conv2d(nn::Conv2dOptions(dim, 2 * dim, 1));
    register_module("vert_to_horiz", this->vert_to_horiz);

    this->horiz_stack = nn::Conv2d(nn::Conv2dOptions(dim, 2 * dim, {1, kernel / 2 + 1}).stride(1).padding({0, kernel / 2}));
    register_module("horiz_stack", this->horiz_stack);

    this->horiz_resid = nn::Conv2d(nn::Conv2dOptions(dim, dim, 1));
    register_module("horiz_resid", this->horiz_resid);

    this->gate = GatedActivation();
    register_module("gate", this->gate);

    this->vmask = torch::zeros_like(this->vert_stack->weight);
    this->vmask.index_put_({Slice(), Slice(), Slice(0, kernel / 2), Slice()}, 1.0);
    if (mask_type == 'B'){
        this->vmask.index_put_({Slice(), Slice(), kernel / 2, Slice()}, 1.0);
    }
    register_buffer("vmask", this->vmask);

    this->hmask = torch::zeros_like(this->horiz_stack->weight);
    this->hmask.index_put_({Slice(), Slice(), 0, Slice(0, kernel / 2)}, 1.0);
    if (mask_type == 'B'){
        this->hmask.index_put_({Slice(), Slice(), 0, kernel / 2}, 1.0);
    }
    register_buffer("hmask", this->hmask);

}


// -----------------------------------------------------------------------------
// struct{GatedMaskedConv2dImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> GatedMaskedConv2dImpl::forward(torch::Tensor x_v, torch::Tensor x_h){

    long int vkh, vkw, hw;
    torch::Tensor xvp, wv, hv, out_v, xhp, wh, hh, v2h, out, out_h;

    vkh = this->vmask.size(2);
    vkw = this->vmask.size(3);
    xvp = F::pad(x_v, F::PadFuncOptions({vkw / 2, vkw / 2, vkh - 1, 0}));
    wv = this->vert_stack->weight * this->vmask;
    hv = F::conv2d(xvp, wv, F::Conv2dFuncOptions().stride(1).padding(0).bias(this->vert_stack->bias));
    out_v = this->gate->forward(hv);

    hw = this->hmask.size(3);
    xhp = F::pad(x_h, F::PadFuncOptions({hw - 1, 0, 0, 0}));
    wh = this->horiz_stack->weight * this->hmask;
    hh = F::conv2d(xhp, wh, F::Conv2dFuncOptions().stride(1).padding(0).bias(this->horiz_stack->bias));
    v2h = this->vert_to_horiz->forward(out_v);
    out = this->gate->forward(v2h + hh);
    out_h = this->residual ? (this->horiz_resid->forward(out) + x_h) : this->horiz_resid->forward(out);

    return {out_v, out_h};

}


// -----------------------------------------------------------------------------
// struct{GatedPixelSnailImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
GatedPixelSnailImpl::GatedPixelSnailImpl(po::variables_map &vm){

    this->dim = vm["dim_pix"].as<size_t>();

    this->token_emb = nn::Embedding(nn::EmbeddingOptions(vm["K"].as<size_t>(), this->dim));
    register_module("token_emb", this->token_emb);

    this->layers->push_back(GatedMaskedConv2d('A', this->dim, /*kernel_size=*/7, /*residual_=*/false));
    for (size_t i = 1; i < vm["n_layers"].as<size_t>(); i++){
        this->layers->push_back(GatedMaskedConv2d('B', this->dim, /*kernel_size=*/3, /*residual_=*/true));
    }
    register_module("layers", this->layers);

    this->output_conv = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(this->dim, 512, 1)),
        nn::ReLU(nn::ReLUOptions(true)),
        nn::Conv2d(nn::Conv2dOptions(512, vm["K"].as<size_t>(), 1))
    );
    register_module("output_conv", this->output_conv);

}


// -----------------------------------------------------------------------------
// struct{GatedPixelSnailImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor GatedPixelSnailImpl::forward(torch::Tensor x){

    torch::Tensor emb, x_v, x_h, out;
    std::tuple<torch::Tensor, torch::Tensor> x_vh;

    emb = this->token_emb->forward(x.view({-1})).view({x.size(0), x.size(1), x.size(2), this->dim}).permute({0, 3, 1, 2}).contiguous();
    x_v = emb.clone();
    x_h = emb;

    for (size_t i = 0; i < this->layers->size(); i++){
        x_vh = this->layers[i]->as<GatedMaskedConv2d>()->forward(x_v, x_h);
        x_v = std::get<0>(x_vh);
        x_h = std::get<1>(x_vh);
    }
    out = this->output_conv->forward(x_h);

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
// struct{VQVAE2Impl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
VQVAE2Impl::VQVAE2Impl(po::variables_map &vm){
    
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
// struct{VQVAE2Impl}(nn::Module) -> function{forward_z}
// ----------------------------------------------------------------------
torch::Tensor VQVAE2Impl::sampling(const std::vector<long int> z_shape, GatedPixelSnail pixelsnail, torch::Device device){

    torch::Tensor idx, logits, probs, sampled, z_q, out;

    idx = torch::randint(0, this->vq->e.size(0), {z_shape[0], z_shape[2], z_shape[3]}).to(device);
    // idx = torch::zeros({z_shape[0], z_shape[2], z_shape[3]}).to(torch::kLong).to(device);
    for (long int j = 0; j < z_shape[2]; j++){
        for (long int i = 0; i < z_shape[3]; i++){
            logits = pixelsnail->forward(idx);
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
// struct{VQVAE2Impl}(nn::Module) -> function{forward_z}
// ----------------------------------------------------------------------
torch::Tensor VQVAE2Impl::synthesis(torch::Tensor x, torch::Tensor y, const float alpha){
    torch::Tensor zx_e, zy_e, z_e, out;
    zx_e = this->encoder->forward(x);    // {C,256,256} ===> {Z,64,64}
    zy_e = this->encoder->forward(y);    // {C,256,256} ===> {Z,64,64}
    z_e = zx_e * alpha + zy_e * (1.0 - alpha);
    auto [z_q_, z_q, idx] = this->vq->forward(z_e);
    out = this->decoder->forward(z_q_);  // {Z,64,64} ===> {C,256,256}
    return out;
}


// ----------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> VQVAE2Impl::forward(torch::Tensor x){
    torch::Tensor z_e = this->encoder->forward(x);     // {C,256,256} ===> {Z,64,64}
    auto [z_q_, z_q, idx] = this->vq->forward(z_e);
    torch::Tensor out = this->decoder->forward(z_q_);  // {Z,64,64} ===> {C,256,256}
    return {out, z_e, z_q};
}


// ----------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor VQVAE2Impl::forward_idx(torch::Tensor x){
    torch::Tensor z_e = this->encoder->forward(x);     // {C,256,256} ===> {Z,64,64}
    auto [z_q_, z_q, idx] = this->vq->forward(z_e);
    return idx;
}


// -----------------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{get_z_shape}
// -----------------------------------------------------------------------------
std::vector<long int> VQVAE2Impl::get_z_shape(const std::vector<long int> x_shape, torch::Device &device){
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

