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
// struct{WNConv2dImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
WNConv2dImpl::WNConv2dImpl(long int in_nc, long int out_nc, std::vector<long int> kernel, long int stride, std::vector<long int> padding, bool bias){

    this->conv = torch::nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, kernel).stride(stride).padding(padding).bias(bias));
    register_module("conv", conv);

    this->v = register_parameter("v", this->conv->weight.detach().clone());
    this->g = register_parameter("g", this->conv->weight.norm(2, /*dim=*/{1, 2, 3}, /*keepdim=*/true).detach().clone());

}


// -----------------------------------------------------------------------------
// struct{WNConv2dImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor WNConv2dImpl::forward(torch::Tensor x){
    torch::Tensor v_norm, out;
    v_norm = this->v.norm(2, /*dim=*/{1, 2, 3}, /*keepdim=*/true).clamp(1e-10);
    this->conv->weight = this->g * this->v / v_norm;
    out = this->conv->forward(x);
    return out;
}


// -----------------------------------------------------------------------------
// struct{WNLinearImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
WNLinearImpl::WNLinearImpl(long int in_nc, long int out_nc){

    this->linear = torch::nn::Linear(in_nc, out_nc);
    register_module("linear", linear);

    this->v = register_parameter("v", this->linear->weight.detach().clone());
    this->g = register_parameter("g", this->linear->weight.norm(2, /*dim=*/{1}, /*keepdim=*/true).detach().clone());

}


// -----------------------------------------------------------------------------
// struct{WNLinearImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor WNLinearImpl::forward(torch::Tensor x){
    torch::Tensor v_norm, out;
    v_norm = this->v.norm(2, /*dim=*/{1}, /*keepdim=*/true).clamp(1e-10);
    this->linear->weight = this->g * this->v / v_norm;
    out = this->linear->forward(x);
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

    this->conv = WNConv2d(in_nc, out_nc, kernel, stride, /*padding=*/std::vector<long int>{0, 0});
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
        this->conv->v.index_put_({Slice(), Slice(), -1, Slice(this->causal, torch::indexing::None)}, 0.0);
    }
    out = this->conv->forward(out);
    return out;
}


// -----------------------------------------------------------------------------
// struct{GatedResBlockImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
GatedResBlockImpl::GatedResBlockImpl(long int in_nc, long int nc, std::vector<long int> kernel, std::string conv, std::string act, float droprate, long int aux_nc){

    if (conv == "wnconv2d"){
        this->conv1->push_back(WNConv2d(in_nc, nc, kernel, /*stride=*/1, /*padding=*/std::vector<long int>{kernel[0] / 2, kernel[1] / 2}));
        this->conv2->push_back(WNConv2d(nc, in_nc * 2, kernel, /*stride=*/1, /*padding=*/std::vector<long int>{kernel[0] / 2, kernel[1] / 2}));
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
        this->aux_conv->push_back(WNConv2d(aux_nc, nc, /*kernel*/std::vector<long int>{1, 1}));
    }
    else{
        this->aux_conv->push_back(nn::Identity());
    }
    register_module("aux_conv", this->aux_conv);

    this->dropout = nn::Dropout(droprate);
    register_module("dropout", this->dropout);

    this->gate = nn::GLU(nn::GLUOptions(1));
    register_module("gate", this->gate);

}


// -----------------------------------------------------------------------------
// struct{GatedResBlockImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor GatedResBlockImpl::forward(torch::Tensor x, torch::Tensor aux){
    
    torch::Tensor out;

    out = this->conv1->forward(this->activation->forward(x));

    if (aux.numel() > 0){
        out = out + this->aux_conv->forward(this->activation->forward(aux));
    }

    out = this->activation->forward(out);
    out = this->dropout->forward(out);
    out = this->conv2->forward(out);

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

    this->query_linear = WNLinear(query_nc, nc);
    register_module("query_linear", this->query_linear);

    this->key_linear = WNLinear(key_nc, nc);
    register_module("key_linear", this->key_linear);

    this->value_linear = WNLinear(key_nc, nc);
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

    query_flat = query.view({N, query.size(1), -1}).transpose(1, 2).contiguous();
    key_flat = key.view({N, key.size(1), -1}).transpose(1, 2).contiguous();
    query_map = this->query_linear->forward(query_flat).view({N, -1, this->n_head, this->dim_head}).transpose(1, 2).contiguous();
    key_map = this->key_linear->forward(key_flat).view({N, -1, this->n_head, this->dim_head}).transpose(1, 2).transpose(2, 3).contiguous();
    value_map = this->value_linear(key_flat).view({N, -1, this->n_head, this->dim_head}).transpose(1, 2).contiguous();

    attn = query_map.matmul(key_map) / std::sqrt(this->dim_head);
    mask = torch::ones({H * W, H * W}).triu(1).t().unsqueeze(0).to(query.device());
    start_mask = torch::ones({H * W}).to(query.device());
    start_mask[0] = 0.0;
    start_mask = start_mask.unsqueeze(1);
    attn = attn.masked_fill(mask == 0, -1e4);
    attn = torch::softmax(attn, 3) * start_mask;
    attn = this->dropout->forward(attn);

    out = attn.matmul(value_map);
    out = out.transpose(1, 2).contiguous().view({N, H, W, this->dim_head * this->n_head}).contiguous();
    out = out.permute({0, 3, 1, 2}).contiguous();

    return out;

}


// -----------------------------------------------------------------------------
// struct{PixelBlockImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
PixelBlockImpl::PixelBlockImpl(long int in_nc, long int nc, long int kernel, long int res_block_, bool attention_, float droprate){
    
    this->res_block = res_block_;
    this->attention = attention_;

    for (long int i = 0; i < this->res_block; i++){
        this->resblocks->push_back(GatedResBlock(in_nc, nc, std::vector<long int>{kernel, kernel}, /*conv=*/"causal", /*act=*/"ELU", /*dropout=*/droprate, /*aux_nc=*/0));
    }
    register_module("resblocks", this->resblocks);

    if (this->attention){
        this->key_resblock = GatedResBlock(in_nc * 2 + 2, in_nc, std::vector<long int>{1, 1}, /*conv=*/"wnconv2d", /*act=*/"ELU", /*dropout=*/droprate);
        this->query_resblock = GatedResBlock(in_nc + 2, in_nc, std::vector<long int>{1, 1}, /*conv=*/"wnconv2d", /*act=*/"ELU", /*dropout=*/droprate);
        this->causal_attention = CausalAttention(in_nc + 2, in_nc * 2 + 2, in_nc / 2, 8, droprate);
        this->out_resblock = GatedResBlock(in_nc, in_nc, std::vector<long int>{1, 1}, /*conv=*/"wnconv2d", /*act=*/"ELU", /*dropout=*/droprate, /*aux_nc=*/in_nc / 2);
        register_module("key_resblock", this->key_resblock);
        register_module("query_resblock", this->query_resblock);
        register_module("causal_attention", this->causal_attention);
        register_module("out_resblock", this->out_resblock);
    }
    else{
        this->out_conv = WNConv2d(in_nc + 2, in_nc, std::vector<long int>{1, 1});
        register_module("out_conv", this->out_conv);
    }

}


// -----------------------------------------------------------------------------
// struct{PixelBlockImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor PixelBlockImpl::forward(torch::Tensor x, torch::Tensor background){

    torch::Tensor out, key_cat, key, query_cat, query, attn_out, bg_cat;

    out = x;
    for (long int i = 0; i < this->res_block; i++){
        out = this->resblocks[i]->as<GatedResBlock>()->forward(out, torch::Tensor());
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
// struct{PixelSNAILImpl}(nn::Module) -> constructor
// -----------------------------------------------------------------------------
PixelSNAILImpl::PixelSNAILImpl(po::variables_map &vm){

    long int kernel = 5;
    long int nc = 1;
    long int dim = vm["dim"].as<size_t>();
    this->block = vm["pix_block"].as<size_t>();
    this->level = vm["L"].as<size_t>();

    this->horizontal = CausalConv2d(nc, dim, std::vector<long int>{kernel / 2, kernel}, 1, /*padding=*/"down");
    this->vertical = CausalConv2d(nc, dim, std::vector<long int>{(kernel + 1) / 2, kernel / 2}, 1, /*padding=*/"downright");
    register_module("horizontal", this->horizontal);
    register_module("vertical", this->vertical);

    for (long int i = 0; i < this->block; i++){
        this->blocks->push_back(PixelBlock(dim, vm["res_nc"].as<size_t>(), kernel, vm["res_block"].as<size_t>(), true, vm["droprate"].as<float>()));
    }
    register_module("blocks", this->blocks);

    for (size_t i = 0; i < vm["out_res_block"].as<size_t>(); i++){
        this->out_module->push_back(GatedResBlock(dim, vm["res_nc"].as<size_t>(), std::vector<long int>{1, 1}));
    }
    this->out_module->push_back(nn::ELU(nn::ELUOptions().inplace(true)));
    this->out_module->push_back(WNConv2d(dim, nc * this->level, std::vector<long int>{1, 1}));
    register_module("out_module", this->out_module);

}


// ----------------------------------------------------------------------
// struct{PixelSNAILImpl}(nn::Module) -> function{forward_z}
// ----------------------------------------------------------------------
torch::Tensor PixelSNAILImpl::sampling(const std::vector<long int> x_shape, torch::Device device){

    torch::Tensor out, logits, probs, sampled;

    out = torch::zeros(x_shape).to(device);
    for (long int j = 0; j < x_shape[2]; j++){
        for (long int i = 0; i < x_shape[3]; i++){
            logits = this->forward(out);
            probs = torch::softmax(logits.index({Slice(), Slice(), j, i}), /*dim=*/1);
            sampled = probs.multinomial(1).squeeze(1) / (this->level - 1);
            out.index_put_({Slice(), Slice(), j, i}, sampled);
        }
    }

    return out;

}


// -----------------------------------------------------------------------------
// struct{PixelSNAILImpl}(nn::Module) -> function{forward}
// -----------------------------------------------------------------------------
torch::Tensor PixelSNAILImpl::forward(torch::Tensor x){

    torch::Tensor horiz, vert, out, coord_x, coord_y, back;

    horiz = this->horizontal->forward(x);
    horiz = F::pad(horiz, std::vector<long int>{0, 0, 1, 0}).index({Slice(), Slice(), Slice(0, horiz.size(2)), Slice()});
    vert = this->vertical->forward(x);
    vert = F::pad(vert, std::vector<long int>{1, 0, 0, 0}).index({Slice(), Slice(), Slice(), Slice(0, vert.size(3))});
    out = horiz + vert;

    coord_x = (torch::arange(x.size(3)).to(torch::kFloat).to(x.device()) - x.size(3) * 0.5) / x.size(3);
    coord_x = coord_x.view({1, 1, 1, x.size(3)}).expand({x.size(0), 1, x.size(2), x.size(3)});
    coord_y = (torch::arange(x.size(2)).to(torch::kFloat).to(x.device()) - x.size(2) * 0.5) / x.size(2);
    coord_y = coord_y.view({1, 1, x.size(2), 1}).expand({x.size(0), 1, x.size(2), x.size(3)});
    back = torch::cat({coord_x, coord_y}, 1);

    for (long int i = 0; i < this->block; i++){
        out = this->blocks[i]->as<PixelBlock>()->forward(out, back);
    }
    out = this->out_module->forward(out);

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

