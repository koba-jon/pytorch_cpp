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
// struct{PixelSnailImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
PixelSnailImpl::PixelSnailImpl(po::variables_map &vm){

    this->dim = vm["dim_pix"].as<size_t>();

    this->token_emb = nn::Embedding(nn::EmbeddingOptions(vm["K"].as<size_t>(), this->dim));
    register_module("token_emb", this->token_emb);

    this->cond_resnet = nn::Sequential(
        nn::ConvTranspose2d(nn::ConvTranspose2dOptions(this->dim, this->dim, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(this->dim),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(this->dim, this->dim, 3).padding(1).bias(false)),
        nn::BatchNorm2d(this->dim),
        nn::ReLU(nn::ReLUOptions().inplace(true))
    );
    register_module("cond_resnet", this->cond_resnet);

    this->layers->push_back(MaskedConv2dBlock('A', this->dim, false));
    for (size_t i = 1; i < vm["n_layers"].as<size_t>(); i++){
        this->layers->push_back(MaskedConv2dBlock('B', this->dim, true));
    }
    register_module("layers", this->layers);

    this->output_conv = nn::Conv2d(nn::Conv2dOptions(this->dim, vm["K"].as<size_t>(), 1));
    register_module("output_conv", this->output_conv);

}


// -----------------------------------------------------------------------------
// struct{PixelSnailImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor PixelSnailImpl::forward(torch::Tensor x, std::vector<torch::Tensor> condition){
    torch::Tensor x_emb, c_emb, c_res, last, out;
    x_emb = this->token_emb->forward(x.view({-1})).view({x.size(0), x.size(1), x.size(2), this->dim}).permute({0, 3, 1, 2}).contiguous();
    if (condition.size() > 0){
        c_emb = this->token_emb->forward(condition[0].view({-1})).view({condition[0].size(0), condition[0].size(1), condition[0].size(2), this->dim}).permute({0, 3, 1, 2}).contiguous();
        c_res = this->cond_resnet->forward(c_emb);
        x_emb = x_emb + c_res;
    }
    last = this->layers->forward(x_emb);
    out = this->output_conv->forward(last);
    return out;
}


// ----------------------------------------------------------------------
// struct{VectorQuantizerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
VectorQuantizerImpl::VectorQuantizerImpl(const size_t D, const size_t K, const float decay_, const float eps_){
    this->decay = decay_;
    this->eps = eps_;
    this->e = register_buffer("Embedding Feature", torch::randn({(long int)D, (long int)K}));
    this->cluster_size = register_buffer("Cluster Size", torch::zeros({(long int)K}));
    this->e_avg = register_buffer("Embedding Feature Average", this->e.detach().clone());
}


// ----------------------------------------------------------------------
// struct{VectorQuantizerImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> VectorQuantizerImpl::forward(torch::Tensor z_e){

    torch::Tensor z_e_flat, x2, e2, dist, idx, z_q, z_q_st, one_hot, one_hot_sum, embed_sum, n, cluster_size_norm, denom, new_embed, diff;

    z_e_flat = z_e.view({-1, z_e.size(3)});  // {N,ZH,ZW,Z} ===> {N*ZH*ZW,Z}

    x2 = z_e_flat.pow(2.0).sum(/*dim=*/1, /*keepdim=*/true);  // {N*ZH*ZW,Z} ===> {N*ZH*ZW,1}
    e2 = this->e.pow(2.0).sum(/*dim=*/0, /*keepdim=*/true);  // {Z,K} ===> {1,K}
    dist = x2 + e2 - 2.0 * z_e_flat.matmul(this->e);  // {N*ZH*ZW,Z} ===> {N*ZH*ZW,K}

    idx = std::get<1>((-dist).max(1));  // {N*ZH*ZW}
    z_q = F::embedding(idx.view({z_e.size(0), z_e.size(1), z_e.size(2)}), this->e.transpose(0, 1));  // {N,ZH,ZW,Z}
    z_q_st = z_e + (z_q - z_e).detach();

    if (this->is_training()){
        one_hot = F::one_hot(idx, /*num_classes=*/this->e.size(1)).to(torch::kFloat);  // {N*ZH*ZW,K}
        one_hot_sum = one_hot.sum(0);  // {K}
        embed_sum = z_e_flat.transpose(0, 1).matmul(one_hot);  // {Z,K}

        this->cluster_size.mul_(this->decay).add_(one_hot_sum.detach(), 1.0 - this->decay);  // {K}
        this->e_avg.mul_(this->decay).add_(embed_sum.detach(), 1.0 - this->decay);  // {Z,K}

        n = this->cluster_size.sum(); // {}
        cluster_size_norm = (this->cluster_size + this->eps) / (n + this->cluster_size.size(0) * this->eps) * n;  // {K}
        denom = cluster_size_norm.unsqueeze(0);  // {1,K}
        new_embed = this->e_avg / denom.clamp_min(1e-8);  // {Z,K}
        this->e.data().copy_(new_embed);
    }

    diff = (z_q.detach() - z_e).pow(2.0).mean();

    return {z_q_st, diff, idx.view({z_e.size(0), z_e.size(1), z_e.size(2)})};

}


// ----------------------------------------------------------------------
// struct{ResidualLayerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ResidualLayerImpl::ResidualLayerImpl(const size_t dim, const size_t h_dim){
    this->model = nn::Sequential(
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(dim, h_dim, 3).stride(1).padding(1).bias(true)),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Conv2d(nn::Conv2dOptions(h_dim, dim, 1).bias(true))
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
// struct{EncoderImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
EncoderImpl::EncoderImpl(long int in_nc, long int out_nc, long int n_res_block, long int n_res_nc, long int stride){

    if (stride == 4){
        this->model->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc / 2, 4).stride(2).padding(1).bias(true)));
        this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
        this->model->push_back(nn::Conv2d(nn::Conv2dOptions(out_nc / 2, out_nc, 4).stride(2).padding(1).bias(true)));
        this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
        this->model->push_back(nn::Conv2d(nn::Conv2dOptions(out_nc, out_nc, 3).stride(1).padding(1).bias(true)));
    }
    else if (stride == 2){
        this->model->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc / 2, 4).stride(2).padding(1).bias(true)));
        this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
        this->model->push_back(nn::Conv2d(nn::Conv2dOptions(out_nc / 2, out_nc, 3).stride(1).padding(1).bias(true)));
    }

    for (long int i = 0; i < n_res_block; i++){
        this->model->push_back(ResidualLayer(out_nc, n_res_nc));
    }
    this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));

    register_module("encoder", this->model);

}


// ----------------------------------------------------------------------
// struct{EncoderImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor EncoderImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);
    return out;
}


// ----------------------------------------------------------------------
// struct{DecoderImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
DecoderImpl::DecoderImpl(long int in_nc, long int out_nc, long int mid_nc, long int res_block, long int res_nc, long int stride){

    this->model->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, mid_nc, 3).stride(1).padding(1).bias(true)));

    for (long int i = 0; i < res_block; i++){
        this->model->push_back(ResidualLayer(mid_nc, res_nc));
    }
    this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));

    if (stride == 4){
        this->model->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(mid_nc, mid_nc / 2, 4).stride(2).padding(1).bias(true)));
        this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
        this->model->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(mid_nc / 2, out_nc, 4).stride(2).padding(1).bias(true)));
    }
    else if (stride == 2){
        this->model->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(mid_nc, out_nc, 4).stride(2).padding(1).bias(true)));
    }

    register_module("decoder", this->model);

}


// ----------------------------------------------------------------------
// struct{DecoderImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor DecoderImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);
    return out;
}


// ----------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
VQVAE2Impl::VQVAE2Impl(po::variables_map &vm){

    size_t nc = vm["nc"].as<size_t>();
    size_t feature = vm["nf"].as<size_t>();
    size_t res_block = vm["res_block"].as<size_t>();
    size_t res_nc = vm["res_nc"].as<size_t>();
    size_t nz = vm["nz"].as<size_t>();
    size_t K = vm["K"].as<size_t>();

    this->enc_b = Encoder(nc, feature, res_block, res_nc, /*stride=*/4);
    register_module("enc_b", this->enc_b);

    this->enc_t = Encoder(feature, feature, res_block, res_nc, /*stride=*/2);
    register_module("enc_t", this->enc_t);

    this->quantize_conv_t = nn::Conv2d(nn::Conv2dOptions(feature, nz, 1));
    register_module("quantize_conv_t", this->quantize_conv_t);

    this->quantize_t = VectorQuantizer(nz, K);
    register_module("quantize_t", this->quantize_t);

    this->dec_t = Decoder(nz, nz, feature, res_block, res_nc, /*stride*/2);
    register_module("dec_t", this->dec_t);

    this->quantize_conv_b = nn::Conv2d(nn::Conv2dOptions(nz + feature, nz, 1));
    register_module("quantize_conv_b", this->quantize_conv_b);

    this->quantize_b = VectorQuantizer(nz, K);
    register_module("quantize_b", this->quantize_b);

    this->upsample_t = nn::ConvTranspose2d(nn::ConvTranspose2dOptions(nz, nz, 4).stride(2).padding(1));
    register_module("upsample_t", this->upsample_t);

    this->dec = Decoder(nz + nz, nc, feature, res_block, res_nc, /*stride*/4);
    register_module("dec", this->dec);

}


// ----------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{forward_z}
// ----------------------------------------------------------------------
torch::Tensor VQVAE2Impl::sampling(const std::tuple<std::vector<long int>, std::vector<long int>> idx_shape, PixelSnail pixelsnail_t, PixelSnail pixelsnail_b, torch::Device device){

    std::vector<long int> idx_t_shape, idx_b_shape;
    torch::Tensor idx_t, idx_b, logits, probs, sampled, quant_t, quant_b, up_t, quant, out;

    idx_t_shape = std::get<0>(idx_shape);
    idx_t = torch::zeros(idx_t_shape).to(torch::kLong).to(device);
    for (long int j = 0; j < idx_t_shape[1]; j++){
        for (long int i = 0; i < idx_t_shape[2]; i++){
            logits = pixelsnail_t->forward(idx_t);
            probs = torch::softmax(logits.index({Slice(), Slice(), j, i}), /*dim=*/1);
            sampled = probs.multinomial(1).squeeze(1);
            idx_t.index_put_({Slice(), j, i}, sampled);
        }
    }

    idx_b_shape = std::get<1>(idx_shape);
    idx_b = torch::zeros(idx_b_shape).to(torch::kLong).to(device);
    for (long int j = 0; j < idx_b_shape[1]; j++){
        for (long int i = 0; i < idx_b_shape[2]; i++){
            logits = pixelsnail_b->forward(idx_b, {idx_t});
            probs = torch::softmax(logits.index({Slice(), Slice(), j, i}), /*dim=*/1);
            sampled = probs.multinomial(1).squeeze(1);
            idx_b.index_put_({Slice(), j, i}, sampled);
        }
    }

    quant_t = F::embedding(idx_t, this->quantize_t->e.transpose(0, 1)).permute({0, 3, 1, 2}).contiguous();
    quant_b = F::embedding(idx_b, this->quantize_b->e.transpose(0, 1)).permute({0, 3, 1, 2}).contiguous();
    up_t = this->upsample_t->forward(quant_t);
    quant = torch::cat({up_t, quant_b}, /*dim=*/1);
    out = this->dec->forward(quant);

    return out;

}


// ----------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{forward_z}
// ----------------------------------------------------------------------
torch::Tensor VQVAE2Impl::synthesis(torch::Tensor x, torch::Tensor y, const float alpha){

    torch::Tensor enc_b_out, enc_t_out, enc_b_x_out, enc_t_x_out, enc_b_y_out, enc_t_y_out, quant_t_, dec_t_out, quant_b_, out;

    enc_b_x_out = this->enc_b->forward(x);
    enc_t_x_out = this->enc_t->forward(enc_b_x_out);

    enc_b_y_out = this->enc_b->forward(x);
    enc_t_y_out = this->enc_t->forward(enc_b_y_out);

    enc_b_out = enc_b_x_out * alpha + enc_b_y_out * (1.0 - alpha);
    enc_t_out = enc_t_x_out * alpha + enc_t_y_out * (1.0 - alpha);

    quant_t_ = this->quantize_conv_t->forward(enc_t_out).permute({0, 2, 3, 1}).contiguous();
    auto [quant_t, diff_t, idx_t] = this->quantize_t->forward(quant_t_);
    quant_t = quant_t.permute({0, 3, 1, 2}).contiguous();

    dec_t_out = this->dec_t->forward(quant_t);
    enc_b_out = torch::cat({dec_t_out, enc_b_out}, /*dim=*/1);

    quant_b_ = this->quantize_conv_b->forward(enc_b_out).permute({0, 2, 3, 1}).contiguous();
    auto [quant_b, diff_b, idx_b] = this->quantize_b->forward(quant_b_);
    quant_b = quant_b.permute({0, 3, 1, 2}).contiguous();

    out = this->decode(quant_t, quant_b);

    return out;

}


// ----------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{encode}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> VQVAE2Impl::encode(torch::Tensor x){

    torch::Tensor enc_b_out, enc_t_out, quant_t_, dec_t_out, quant_b_;

    enc_b_out = this->enc_b->forward(x);
    enc_t_out = this->enc_t->forward(enc_b_out);

    quant_t_ = this->quantize_conv_t->forward(enc_t_out).permute({0, 2, 3, 1}).contiguous();
    auto [quant_t, diff_t, idx_t] = this->quantize_t->forward(quant_t_);
    quant_t = quant_t.permute({0, 3, 1, 2}).contiguous();
    diff_t = diff_t.unsqueeze(0);

    dec_t_out = this->dec_t->forward(quant_t);
    enc_b_out = torch::cat({dec_t_out, enc_b_out}, /*dim=*/1);

    quant_b_ = this->quantize_conv_b->forward(enc_b_out).permute({0, 2, 3, 1}).contiguous();
    auto [quant_b, diff_b, idx_b] = this->quantize_b->forward(quant_b_);
    quant_b = quant_b.permute({0, 3, 1, 2}).contiguous();
    diff_b = diff_b.unsqueeze(0);

    return {quant_t, quant_b, diff_t + diff_b, idx_t, idx_b};

}


// ----------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{decode}
// ----------------------------------------------------------------------
torch::Tensor VQVAE2Impl::decode(torch::Tensor quant_t, torch::Tensor quant_b){
    torch::Tensor up_t, quant, out;
    up_t = this->upsample_t->forward(quant_t);
    quant = torch::cat({up_t, quant_b}, /*dim=*/1);
    out = this->dec->forward(quant);
    return out;
}


// ----------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> VQVAE2Impl::forward(torch::Tensor x){
    auto [quant_t, quant_b, diff, _, __] = this->encode(x);
    auto out = this->decode(quant_t, quant_b);
    return {out, diff};
}


// ----------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> VQVAE2Impl::forward_idx(torch::Tensor x){
    auto [_, __, ___, idx_t, idx_b] = this->encode(x);
    return {idx_t, idx_b};
}


// -----------------------------------------------------------------------------
// struct{VQVAE2Impl}(nn::Module) -> function{get_z_shape}
// -----------------------------------------------------------------------------
std::tuple<std::vector<long int>, std::vector<long int>> VQVAE2Impl::get_idx_shape(const std::vector<long int> x_shape, torch::Device &device){
    torch::Tensor x = torch::full(x_shape, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    auto [_, __, ___, idx_t, idx_b] = this->encode(x);
    std::vector<long int> z_t_shape = {idx_t.size(0), idx_t.size(1), idx_t.size(2)};
    std::vector<long int> z_b_shape = {idx_b.size(0), idx_b.size(1), idx_b.size(2)};
    return {z_t_shape, z_b_shape};
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

