#include <cmath>
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
// struct{BNConv2dImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
BNConv2dImpl::BNConv2dImpl(long int in_nc, long int out_nc, std::vector<long int> kernel, long int stride, std::vector<long int> padding, bool bias){

    this->conv = torch::nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, kernel).stride(stride).padding(padding).bias(bias));
    register_module("conv", conv);

    this->bn = torch::nn::BatchNorm2d(out_nc);
    register_module("bn", bn);

}


// -----------------------------------------------------------------------------
// struct{BNConv2dImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor BNConv2dImpl::forward(torch::Tensor x){
    torch::Tensor out;
    out = this->conv->forward(x);
    out = this->bn->forward(out);
    return out;
}


// -----------------------------------------------------------------------------
// struct{CausalConv2dImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
CausalConv2dImpl::CausalConv2dImpl(long int in_nc, long int out_nc, std::vector<long int> kernel, long int stride, std::string padding){

    std::vector<long int> pad;

    if (padding == "downright"){
        pad = {kernel[1] - 1, 0, kernel[0] - 1, 0};
    }
    else if ((padding == "down") || (padding == "causal")){
        pad = {kernel[1] / 2, kernel[1] / 2, kernel[0] - 1, 0};
    }

    this->causal = 0;
    if (padding == "causal"){
        this->causal = kernel[1] / 2;
    }

    this->zero_pad = nn::ZeroPad2d(nn::ZeroPad2dOptions(pad));
    register_module("zero_pad", zero_pad);

    this->conv = BNConv2d(in_nc, out_nc, kernel, stride, /*padding=*/std::vector<long int>{0, 0});
    register_module("conv", this->conv);

}


// -----------------------------------------------------------------------------
// struct{CausalConv2dImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor CausalConv2dImpl::forward(torch::Tensor x){
    torch::Tensor out;
    out = this->zero_pad(x);
    if (this->causal > 0) {
        torch::NoGradGuard no_grad;
        this->conv->conv->weight.index_put_({Slice(), Slice(), -1, Slice(this->causal, torch::indexing::None)}, 0.0);
    }
    out = this->conv->forward(out);
    return out;
}


// -----------------------------------------------------------------------------
// struct{GatedResBlockImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
GatedResBlockImpl::GatedResBlockImpl(long int in_nc, long int nc, std::vector<long int> kernel, std::string conv, std::string act, float droprate, long int aux_nc, long int cond_dim){

    if (conv == "bnconv2d"){
        this->conv1->push_back(BNConv2d(in_nc, nc, kernel, /*stride=*/1, /*padding=*/std::vector<long int>{kernel[0] / 2, kernel[1] / 2}));
        this->conv2->push_back(BNConv2d(nc, in_nc * 2, kernel, /*stride=*/1, /*padding=*/std::vector<long int>{kernel[0] / 2, kernel[1] / 2}));
    }
    else if (conv == "causal_downright"){
        this->conv1->push_back(CausalConv2d(in_nc, nc, kernel, /*stride=*/1, /*padding=*/"downright"));
        this->conv2->push_back(CausalConv2d(nc, in_nc * 2, kernel, /*stride=*/1, /*padding=*/"downright"));
    }
    else if (conv == "causal"){
        this->conv1->push_back(CausalConv2d(in_nc, nc, kernel, /*stride=*/1, /*padding=*/"causal"));
        this->conv2->push_back(CausalConv2d(nc, in_nc * 2, kernel, /*stride=*/1, /*padding=*/"causal"));
    }
    register_module("conv1", this->conv1);
    register_module("conv2", this->conv2);

    if (act == "ELU"){
        this->activation = nn::ELU();
    }
    register_module("activation", this->activation);

    if (aux_nc > 0){
        this->aux_conv->push_back(BNConv2d(aux_nc, nc, /*kernel*/std::vector<long int>{1, 1}));
    }
    else{
        this->aux_conv->push_back(nn::Identity());
    }
    register_module("aux_conv", this->aux_conv);

    this->dropout = nn::Dropout(droprate);
    register_module("dropout", this->dropout);

    if (cond_dim > 0){
        this->cond->push_back(BNConv2d(cond_dim, in_nc * 2, /*kernel*/std::vector<long int>{1, 1}, /*stride=*/1, /*padding=*/std::vector<long int>{0, 0}, /*bias=*/false));
    }
    else{
        this->cond->push_back(nn::Identity());
    }
    register_module("cond", this->cond);

    this->gate = nn::GLU(nn::GLUOptions(1));
    register_module("gate", this->gate);

}


// -----------------------------------------------------------------------------
// struct{GatedResBlockImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor GatedResBlockImpl::forward(torch::Tensor x, torch::Tensor aux, torch::Tensor condition){
    
    torch::Tensor out;

    out = this->conv1->forward(this->activation->forward(x));

    if (aux.numel() > 0){
        out = out + this->aux_conv->forward(this->activation->forward(aux));
    }

    out = this->activation->forward(out);
    out = this->dropout->forward(out);
    out = this->conv2->forward(out);

    if (condition.numel() > 0){
        condition = this->cond->forward(condition);
        out = out + condition;
    }

    out = this->gate->forward(out);
    out = out + x;

    return out;

}


// -----------------------------------------------------------------------------
// struct{CausalAttentionImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
CausalAttentionImpl::CausalAttentionImpl(long int query_nc, long int key_nc, long int nc, long int n_head_, float droprate){

    this->dim_head = nc / n_head_;
    this->n_head = n_head_;

    this->query_linear = nn::Linear(query_nc, nc);
    register_module("query_linear", this->query_linear);

    this->key_linear = nn::Linear(key_nc, nc);
    register_module("key_linear", this->key_linear);

    this->value_linear = nn::Linear(key_nc, nc);
    register_module("value_linear", this->value_linear);

    this->dropout = nn::Dropout(droprate);
    register_module("dropout", this->dropout);

}


// -----------------------------------------------------------------------------
// struct{CausalAttentionImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor CausalAttentionImpl::forward(torch::Tensor query, torch::Tensor key){

    long int N, H, W;
    torch::Tensor query_flat, key_flat, query_map, key_map, value_map, attn, mask, start_mask, out;

    N = key.size(0);
    H = key.size(2);
    W = key.size(3);

    query_flat = query.view({N, query.size(1), -1}).transpose(1, 2);
    key_flat = key.view({N, key.size(1), -1}).transpose(1, 2);
    query_map = this->query_linear->forward(query_flat).view({N, -1, this->n_head, this->dim_head}).transpose(1, 2);
    key_map = this->key_linear->forward(key_flat).view({N, -1, this->n_head, this->dim_head}).transpose(1, 2).transpose(2, 3);
    value_map = this->value_linear(key_flat).view({N, -1, this->n_head, this->dim_head}).transpose(1, 2);

    attn = query_map.matmul(key_map) / std::sqrt(this->dim_head);
    mask = torch::ones({H * W, H * W}).triu(1).t().unsqueeze(0).to(query.device());
    start_mask = torch::ones({H * W}).to(query.device());
    start_mask[0] = 0.0;
    start_mask = start_mask.unsqueeze(1);
    attn = attn.masked_fill(mask == 0, -1e4);
    attn = torch::softmax(attn, 3) * start_mask;
    attn = this->dropout->forward(attn);

    out = attn.matmul(value_map);
    out = out.transpose(1, 2).view({N, H, W, this->dim_head * this->n_head}).contiguous();
    out = out.permute({0, 3, 1, 2}).contiguous();

    return out;

}


// -----------------------------------------------------------------------------
// struct{PixelBlockImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
PixelBlockImpl::PixelBlockImpl(long int in_nc, long int nc, long int kernel, long int res_block_, bool attention_, float droprate, long int cond_dim){
    
    this->res_block = res_block_;
    this->attention = attention_;

    for (long int i = 0; i < this->res_block; i++){
        this->resblocks->push_back(GatedResBlock(in_nc, nc, std::vector<long int>{kernel, kernel}, /*conv=*/"causal", /*act=*/"ELU", /*dropout=*/droprate, /*aux_nc=*/0, /*cond_dim=*/cond_dim));
    }
    register_module("resblocks", this->resblocks);

    if (this->attention){
        this->key_resblock = GatedResBlock(in_nc * 2 + 2, in_nc, std::vector<long int>{1, 1}, /*conv=*/"bnconv2d", /*act=*/"ELU", /*dropout=*/droprate);
        this->query_resblock = GatedResBlock(in_nc + 2, in_nc, std::vector<long int>{1, 1}, /*conv=*/"bnconv2d", /*act=*/"ELU", /*dropout=*/droprate);
        this->causal_attention = GatedResBlock(in_nc + 2, in_nc * 2 + 2, std::vector<long int>{in_nc / 2, in_nc / 2}, /*conv=*/"bnconv2d", /*act=*/"ELU", /*dropout=*/droprate);
        this->out_resblock = GatedResBlock(in_nc, in_nc, std::vector<long int>{1, 1}, /*conv=*/"bnconv2d", /*act=*/"ELU", /*dropout=*/droprate, /*aux_nc=*/in_nc + 2);
        register_module("key_resblock", this->key_resblock);
        register_module("query_resblock", this->query_resblock);
        register_module("causal_attention", this->causal_attention);
        register_module("out_resblock", this->out_resblock);
    }
    else{
        this->out_conv = BNConv2d(in_nc + 2, in_nc, std::vector<long int>{1, 1});
        register_module("out_conv", this->out_conv);
    }

}


// -----------------------------------------------------------------------------
// struct{PixelBlockImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor PixelBlockImpl::forward(torch::Tensor x, torch::Tensor background, torch::Tensor condition){

    torch::Tensor out, key_cat, key, query_cat, query, attn_out, bg_cat;

    out = x;
    for (long int i = 0; i < this->res_block; i++){
        out = this->resblocks[i]->as<GatedResBlock>()->forward(out, torch::Tensor(), condition);
    }

    if (this->attention){
        key_cat = torch::cat({x, out, background}, 1);
        key = this->key_resblock->forward(key_cat);
        query_cat = torch::cat({out, background}, 1);
        query = this->query_resblock->forward(query_cat);
        attn_out = this->causal_attention->forward(query, key);
        out = this->out_resblock->forward(out, attn_out);
    }
    else{
        bg_cat = torch::cat({out, background}, 1);
        out = this->out_conv->forward(bg_cat);
    }

    return out;

}


// -----------------------------------------------------------------------------
// struct{CondResNetImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
CondResNetImpl::CondResNetImpl(long int in_nc, long int nc, long int kernel, long int res_block_){
    this->res_block = res_block_;
    this->blocks->push_back(BNConv2d(in_nc, nc, std::vector<long int>{kernel, kernel}, /*stride=*/1, /*padding=*/std::vector<long int>{kernel / 2, kernel / 2}));
    for (long int i = 0; i < this->res_block; i++){
        this->blocks->push_back(GatedResBlock(nc, nc, std::vector<long int>{kernel, kernel}));
    }
    register_module("blocks", this->blocks);
}


// -----------------------------------------------------------------------------
// struct{CondResNetImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor CondResNetImpl::forward(torch::Tensor x){
    torch::Tensor out;
    out = this->blocks[0]->as<BNConv2d>()->forward(x);
    for (long int i = 0; i < this->res_block; i++){
        out = this->blocks[i + 1]->as<GatedResBlock>()->forward(out);
    }
    return out;
}


// -----------------------------------------------------------------------------
// struct{PixelSnailImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
PixelSnailImpl::PixelSnailImpl(std::vector<long int> shape, long int n_class_, long int nc, long int kernel, long int block_, long int res_block, long int res_nc, bool attention, float droprate, long int cond_res_block, long int cond_res_nc, long int cond_res_kernel, long int out_res_block){

    this->n_class = n_class_;
    this->block = block_;
    torch::Tensor coord_x, coord_y;

    if (kernel % 2 == 0) kernel = kernel + 1;
    this->horizontal = CausalConv2d(this->n_class, nc, std::vector<long int>{kernel / 2, kernel}, 1, /*padding=*/"down");
    this->vertical = CausalConv2d(this->n_class, nc, std::vector<long int>{(kernel + 1) / 2, kernel / 2}, 1, /*padding=*/"downright");
    register_module("horizontal", this->horizontal);
    register_module("vertical", this->vertical);

    coord_x = (torch::arange(shape[0]).to(torch::kFloat) - shape[0] * 0.5) / shape[0];
    coord_x = coord_x.view({1, 1, shape[0], 1}).expand({1, 1, shape[0], shape[1]});
    coord_y = (torch::arange(shape[1]).to(torch::kFloat) - shape[1] * 0.5) / shape[1];
    coord_y = coord_y.view({1, 1, 1, shape[0]}).expand({1, 1, shape[0], shape[1]});
    this->background = register_buffer("background", torch::cat({coord_x, coord_y}, 1));

    for (long int i = 0; i < this->block; i++){
        this->blocks->push_back(PixelBlock(nc, res_nc, kernel, res_block, attention, droprate, cond_res_nc));
    }
    register_module("blocks", this->blocks);

    if (cond_res_block > 0){
        this->cond_resnet = CondResNet(this->n_class, cond_res_nc, cond_res_kernel, cond_res_block);
        register_module("cond_resnet", this->cond_resnet);
    }

    for (long int i = 0; i < out_res_block; i++){
        this->out_module->push_back(GatedResBlock(nc, res_nc, std::vector<long int>{1, 1}));
    }
    this->out_module->push_back(nn::ELU(nn::ELUOptions().inplace(true)));
    this->out_module->push_back(BNConv2d(nc, n_class, std::vector<long int>{1, 1}));
    register_module("out_module", this->out_module);

}


// -----------------------------------------------------------------------------
// struct{PixelSnailImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor PixelSnailImpl::forward(torch::Tensor x, torch::Tensor condition){

    torch::Tensor x_, horiz, vert, out, back;

    x_ = F::one_hot(x, this->n_class).permute({0, 3, 1, 2}).to(torch::kFloat);
    horiz = this->horizontal->forward(x_);
    horiz = F::pad(horiz, std::vector<long int>{0, 0, 1, 0}).index({Slice(), Slice(), Slice(0, horiz.size(2)), Slice()});
    vert = this->vertical->forward(x_);
    vert = F::pad(vert, std::vector<long int>{1, 0, 0, 0}).index({Slice(), Slice(), Slice(), Slice(0, vert.size(3))});
    out = horiz + vert;

    back = this->background.index({Slice(), Slice(), Slice(0, x.size(1)), Slice()}).expand({x.size(0), 2, x.size(1), x.size(2)});

    if (condition.numel() > 0){
        condition = F::one_hot(condition, this->n_class).permute({0, 3, 1, 2}).to(torch::kFloat);
        condition = this->cond_resnet->forward(condition);
        condition = F::interpolate(condition, F::InterpolateFuncOptions().scale_factor(std::vector<double>{2.0, 2.0}));
        condition = condition.index({Slice(), Slice(), Slice(0, x.size(1)), Slice()});
    }
    
    for (long int i = 0; i < this->block; i++){
        out = this->blocks[i]->as<PixelBlock>()->forward(out, back, condition);
    }
    out = this->out_module->forward(out);

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

    enc_b_y_out = this->enc_b->forward(y);
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

