#include <cmath>
#include <tuple>
#include <vector>
#include <typeinfo>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;


// ----------------------------------------------------------------------
// struct{FeedForwardImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
FeedForwardImpl::FeedForwardImpl(const size_t dim, const size_t hidden_dim){
    this->mlp = nn::Sequential(
        nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)dim})),
        nn::Linear(dim, hidden_dim),
        nn::GELU(),
        nn::Dropout(0.0),
        nn::Linear(hidden_dim, dim),
        nn::Dropout(0.0)
    );
    register_module("Feed Forward", this->mlp);
}


// ---------------------------------------------------------
// struct{FeedForwardImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor FeedForwardImpl::forward(torch::Tensor x){
    torch::Tensor out = this->mlp->forward(x);
    return out;
}


// ----------------------------------------------------------------------
// struct{AttentionImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
AttentionImpl::AttentionImpl(const size_t dim, const size_t heads_){
    
    this->heads = heads_;

    this->norm = nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)dim}));
    register_module("Layer Normalization", this->norm);

    this->attend = nn::Softmax(/*dim=*/-1);
    register_module("Softmax", this->attend);

    this->dropout = nn::Dropout(0.0);
    register_module("Dropout", this->dropout);

    this->to_qkv = nn::Linear(nn::LinearOptions(dim, dim * 3).bias(false));
    register_module("Linear", this->to_qkv);

    if (this->heads == 1){
        this->to_out = nn::Sequential(nn::Identity());
    }
    else{
        this->to_out = nn::Sequential(
            nn::Linear(dim, dim),
            nn::Dropout(0.0)
        );
    }
    register_module("Output Function", this->to_out);

}


// ---------------------------------------------------------
// struct{AttentionImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor AttentionImpl::forward(torch::Tensor x){
    
    torch::Tensor q, k, v, dots, attn, out;
    std::vector<torch::Tensor> qkv;
    
    x = this->norm->forward(x);

    qkv = this->to_qkv->forward(x).chunk(3, /*dim=*/-1);
    q = qkv[0].view({x.size(0), x.size(1), (long int)this->heads, -1}).permute({0, 2, 1, 3});
    k = qkv[1].view({x.size(0), x.size(1), (long int)this->heads, -1}).permute({0, 2, 1, 3});
    v = qkv[2].view({x.size(0), x.size(1), (long int)this->heads, -1}).permute({0, 2, 1, 3});

    dots = torch::matmul(q, k.transpose(-1, -2));
    
    attn = this->attend->forward(dots);
    attn = this->dropout->forward(attn);

    out = torch::matmul(attn, v);
    out = out.permute({0, 2, 1, 3}).contiguous().view({x.size(0), x.size(1), -1});
    out = this->to_out->forward(out);
    
    return out;

}


// ----------------------------------------------------------------------
// struct{TransformerImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
TransformerImpl::TransformerImpl(size_t dim, size_t heads, float mlp_ratio){
    this->layer->push_back(Attention(dim, heads));
    this->layer->push_back(FeedForward(dim, size_t(dim * mlp_ratio + 0.5)));
    register_module("Layer", this->layer);
}


// ---------------------------------------------------------
// struct{TransformerImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor TransformerImpl::forward(torch::Tensor x){
    torch::Tensor out;
    x = this->layer[0]->as<Attention>()->forward(x) + x;
    out = this->layer[1]->as<FeedForward>()->forward(x) + x;
    return out;
}


// ----------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MaskedAutoEncoderImpl::MaskedAutoEncoderImpl(po::variables_map &vm){

    size_t nc = vm["nc"].as<size_t>();
    this->split = vm["split"].as<size_t>();
    size_t np = vm["split"].as<size_t>() * vm["split"].as<size_t>();
    this->image_size = vm["size"].as<size_t>() - vm["size"].as<size_t>() % vm["split"].as<size_t>();
    this->patch_size = this->image_size / vm["split"].as<size_t>();
    this->enc_dim = vm["enc_dim"].as<size_t>();
    this->dec_dim = vm["dec_dim"].as<size_t>();
    this->keep_num = size_t(np * (1.0 - vm["mask_ratio"].as<float>()));
    size_t enc_depth = vm["enc_depth"].as<size_t>();
    size_t dec_depth = vm["dec_depth"].as<size_t>();
    
    // Convolutional Patch Embedding
    this->conv = nn::Conv2d(nn::Conv2dOptions(nc, this->enc_dim, this->patch_size).stride(this->patch_size).padding(0).bias(true));
    register_module("Convolution", this->conv);

    // Class/Mask Token and Positional Encoding
    this->class_token = register_parameter("Class Token", torch::randn({1, 1, (long int)this->enc_dim}));
    this->mask_token = register_parameter("Mask Token", torch::randn({1, 1, (long int)this->dec_dim}));
    this->enc_pos_encoding = register_buffer("Encoder Positional Encoding", torch::randn({1, (long int)np + 1, (long int)this->enc_dim}));
    this->dec_pos_encoding = register_buffer("Decoder Positional Encoding", torch::randn({1, (long int)np + 1, (long int)this->dec_dim}));

    // Normalization for Encoder
    this->enc_norm = nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)this->enc_dim}));
    register_module("Encoder Layer Normalization", this->enc_norm);

    // Normalization for Decoder
    this->dec_norm = nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)this->dec_dim}));
    register_module("Decoder Layer Normalization", this->dec_norm);

    // Transformer for Encoder
    for (size_t i = 0; i < enc_depth; i++){
        this->enc_block->push_back(
            Transformer(this->enc_dim, vm["enc_heads"].as<size_t>(), vm["mlp_ratio"].as<float>())
        );
    }
    register_module("Encoder Transformer", this->enc_block);

    // Transformer for Decoder
    for (size_t i = 0; i < dec_depth; i++){
        this->dec_block->push_back(
            Transformer(this->dec_dim, vm["dec_heads"].as<size_t>(), vm["mlp_ratio"].as<float>())
        );
    }
    register_module("Decoder Transformer", this->dec_block);

    // Linear for latent space
    this->latent = nn::Linear(this->enc_dim, this->dec_dim);
    register_module("Latent Linear", this->latent);

    // Linear for output
    this->pred = nn::Linear(this->dec_dim, this->patch_size * this->patch_size * nc);
    register_module("Prediction Linear", this->pred);

}


// ------------------------------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{get_1d_sincos_pos_embed_from_grid}
// ------------------------------------------------------------------------------------------
torch::Tensor MaskedAutoEncoderImpl::get_1d_sincos_pos_embed_from_grid(size_t dim, torch::Tensor pos){

    size_t half;
    torch::Tensor omega, emb_out, emb_sin, emb_cos, out;

    pos = pos.view({-1});  // {1,NPH,NPW} ===> {NP}
    half = dim / 2;
    omega = torch::arange(half).to(torch::kFloat);  // {D/4}
    omega = omega / float(half);  // {D/4}
    omega = torch::exp((-std::log(10000.0)) * omega);  // {D/4}

    emb_out = pos.unsqueeze(1) * omega.unsqueeze(0);  // {NP,D/4}
    emb_sin = torch::sin(emb_out);  // {NP,D/4}
    emb_cos = torch::cos(emb_out);  // {NP,D/4}
    out = torch::cat({emb_sin, emb_cos}, /*dim=*/1);  // {NP,D/2}

    return out;

}


// -----------------------------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{get_2d_sincos_pos_embed_for_grid}
// -----------------------------------------------------------------------------------------
torch::Tensor MaskedAutoEncoderImpl::get_2d_sincos_pos_embed_for_grid(size_t dim, torch::Tensor grid){
    torch::Tensor emb0, emb1, out;
    emb0 = this->get_1d_sincos_pos_embed_from_grid(dim / 2, grid.index({0}));  // {1,NPH,NPW} ===> {NP,D/2}
    emb1 = this->get_1d_sincos_pos_embed_from_grid(dim / 2, grid.index({1}));  // {1,NPH,NPW} ===> {NP,D/2}
    out = torch::cat({emb0, emb1}, /*dim=*/1);  // {NP,D}
    return out;
}


// --------------------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{get_2d_sincos_pos_embed}
// --------------------------------------------------------------------------------
torch::Tensor MaskedAutoEncoderImpl::get_2d_sincos_pos_embed(size_t dim, size_t grid_size){

    torch::Tensor grid_h, grid_w, grid, class_token, pos_encoding, out;
    std::vector<torch::Tensor> grids;

    grid_h = torch::arange(grid_size).to(torch::kFloat);  // {NPH}
    grid_w = torch::arange(grid_size).to(torch::kFloat);  // {NPW}

    grids = torch::meshgrid({grid_w, grid_h}, "xy");  // {NPH,NPW}, {NPH,NPW}
    grid = torch::stack(grids, 0).view({2, 1, (long int)grid_size, (long int)grid_size}).to(torch::kFloat);  // {2,1,NPH,NPW}

    class_token = torch::zeros({1, (long int)dim}).to(torch::kFloat);  // {1,D}
    pos_encoding = this->get_2d_sincos_pos_embed_for_grid(dim, grid);  // {2,1,NPH,NPW} ===> {NP,D}
    out = torch::cat({class_token, pos_encoding}, /*dim=*/0);  // {1+NP,D}

    return out;

}


// -----------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{random_masking}
// -----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MaskedAutoEncoderImpl::random_masking(torch::Tensor x){

    torch::Tensor noise, ids_shuffle, ids_restore, ids_keep, x_masked, mask;

    noise = torch::rand({x.size(0), x.size(1)}).to(x.device());  // {N,NP}

    ids_shuffle = torch::argsort(noise, /*dim=*/1, /*descending=*/false);  // {N,NP}
    ids_restore = torch::argsort(ids_shuffle, /*dim=*/1, /*descending=*/false);  // {N,NP}

    ids_keep = ids_shuffle.index({Slice(), Slice(0, this->keep_num)});  // {N,K}
    x_masked = torch::gather(x, /*dim=*/1, /*index=*/ids_keep.unsqueeze(-1).expand({ids_keep.size(0), ids_keep.size(1), x.size(2)}));  // {N,K,ED}

    mask = torch::ones({x.size(0), x.size(1)}).to(x.device());
    mask.index_put_({Slice(), Slice(0, this->keep_num)}, 0.0);
    mask = torch::gather(mask, /*dim=*/1, /*index=*/ids_restore);  // {N,NP}
    
    return {x_masked, mask, ids_restore};

}


// ----------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{encoder}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MaskedAutoEncoderImpl::encoder(torch::Tensor x){

    torch::Tensor pf, mask, ids_restore, cf, z;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mask_set;

    // (1) Apply Convolutional Patch Embedding
    pf = this->conv->forward(x);  // {N,C,H,W} ===> {N,ED,NPH,NPW}
    pf = pf.view({pf.size(0), pf.size(1), -1}).contiguous().transpose(1, 2);  // {N,ED,NPH,NPW} ===> {N,ED,NP} ===> {N,NP,ED}

    // (2) Add positional encoding
    pf += this->enc_pos_encoding.index({Slice(), Slice(1, torch::indexing::None), Slice()});  // {N,NP,ED}

    // (3) Apply Random Masking
    mask_set = this->random_masking(pf);
    pf = std::get<0>(mask_set);  // {N,K,ED}
    mask = std::get<1>(mask_set);  // {N,NP}
    ids_restore = std::get<2>(mask_set);  // {N,NP}

    // (4) Add class token
    cf = this->class_token + this->enc_pos_encoding.index({Slice(), Slice(0, 1), Slice()});  // {1,1,ED}
    cf = cf.expand({pf.size(0), 1, -1});  // {N,1,ED}
    pf = torch::cat({cf, pf}, /*dim=*/1);  // {N,1+K,ED}

    // (5) Apply Transformer
    pf = this->enc_block->forward(pf);  // {N,1+K,ED}
    z = this->enc_norm->forward(pf);  // {N,1+K,ED}
    
    return {z, mask, ids_restore};

}


// ----------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{decoder}
// ----------------------------------------------------------------------
torch::Tensor MaskedAutoEncoderImpl::decoder(torch::Tensor z, torch::Tensor ids_restore){

    torch::Tensor pf, mask_tokens, out;

    // (1) Apply Linear for latent space
    z = this->latent->forward(z);  // {N,1+K,ED} ===> {N,1+K,DD}

    // (2) Add mask tokens to embeddings
    mask_tokens = this->mask_token.expand({z.size(0), ids_restore.size(1) + 1 - z.size(1), this->mask_token.size(2)});  // {1,1,DD} ===> {N,NP-K,DD}
    pf = torch::cat({z.index({Slice(), Slice(1, torch::indexing::None), Slice()}), mask_tokens}, /*dim=*/1);  // {N,NP,DD}
    pf = torch::gather(pf, /*dim=*/1, /*index=*/ids_restore.unsqueeze(-1).expand({ids_restore.size(0), ids_restore.size(1), z.size(2)}));  // {N,NP,DD}
    pf = torch::cat({z.index({Slice(), Slice(0, 1), Slice()}), pf}, /*dim=*/1);  // {N,1+NP,DD}

    // (3) Add positional encoding
    pf += this->dec_pos_encoding;  // {N,1+NP,DD}

    // (4) Apply Transformer
    pf = this->dec_block->forward(pf);  // {N,1+NP,DD}
    pf = this->dec_norm->forward(pf);  // {N,1+NP,DD}

    // (5) Apply Linear for output
    out = this->pred->forward(pf);  // {N,1+NP,C*P*P}
    out = out.index({Slice(), Slice(1, torch::indexing::None), Slice()});  // {N,NP,P*P*C}
    
    return out;

}


// -------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{init}
// -------------------------------------------------------------
void MaskedAutoEncoderImpl::init(){
    this->apply(weights_init);
    nn::init::normal_(this->class_token, /*mean=*/0.0, /*std=*/0.02);
    nn::init::normal_(this->mask_token, /*mean=*/0.0, /*std=*/0.02);
    this->enc_pos_encoding.copy_(this->get_2d_sincos_pos_embed(this->enc_dim, this->split));
    this->dec_pos_encoding.copy_(this->get_2d_sincos_pos_embed(this->dec_dim, this->split));
    return;
}


// ----------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{patchify}
// ----------------------------------------------------------------------
torch::Tensor MaskedAutoEncoderImpl::patchify(torch::Tensor x){
    torch::Tensor out;
    x = x.view({x.size(0), x.size(1), (long int)this->split, (long int)this->patch_size, (long int)this->split, (long int)this->patch_size}).contiguous();  // {N,C,H,W} ===> {N,C,NPH,P,NPW,P}
    x = x.permute({0, 2, 4, 3, 5, 1}).contiguous();  // {N,C,NPH,P,NPW,P} ===> {N,NPH,NPW,P,P,C}
    out = x.view({x.size(0), x.size(1) * x.size(2), x.size(3) * x.size(4) * x.size(5)}).contiguous();  // {N,NPH,NPW,P,P,C} ===> {N,NP,P*P*C}
    return out;
}


// ----------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{unpatchify}
// ----------------------------------------------------------------------
torch::Tensor MaskedAutoEncoderImpl::unpatchify(torch::Tensor x){
    torch::Tensor out;
    x = x.view({x.size(0), (long int)this->split, (long int)this->split, (long int)this->patch_size, (long int)this->patch_size, -1}).contiguous();  // {N,NP,P*P*C} ===> {N,NPH,NPW,P,P,C}
    x = x.permute({0, 5, 1, 3, 2, 4}).contiguous();  // {N,NPH,NPW,P,P,C} ===> {N,C,NPH,P,NPW,P}
    out = x.view({x.size(0), x.size(1), x.size(2) * x.size(3), x.size(4) * x.size(5)}).contiguous(); // {N,C,NPH,P,NPW,P} ===> {N,C,H,W}
    return out;
}


// ----------------------------------------------------------------------
// struct{MaskedAutoEncoderImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MaskedAutoEncoderImpl::forward(torch::Tensor x){

    torch::Tensor z, mask, ids_restore, out, patch;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> enc_set;

    x = F::interpolate(x, F::InterpolateFuncOptions().size(std::vector<long int>{(long int)this->image_size, (long int)this->image_size}).mode(torch::kBilinear).align_corners(false));  // {N,C,H,W}

    enc_set = this->encoder(x);  // {N,C,H,W}
    z = std::get<0>(enc_set);  // {N,1+K,ED}
    mask = std::get<1>(enc_set);  // {N,NP}
    ids_restore = std::get<2>(enc_set);  // {N,NP}

    patch = this->patchify(x);  // {N,C,H,W} ===> {N,NP,P*P*C}
    out = this->decoder(z, ids_restore);  // {N,1+K,ED}, {N,NP} ===> {N,NP,P*P*C}
    out = out * mask.unsqueeze(-1) + patch * (1.0 - mask.unsqueeze(-1));  // {N,NP,P*P*C}

    return {patch, out, mask};

}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl))) {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::LayerNorm)) || (typeid(m) == typeid(nn::LayerNormImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::constant_(*w, /*weight=*/1.0);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}
