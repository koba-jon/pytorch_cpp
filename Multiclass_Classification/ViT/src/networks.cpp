#include <cmath>
#include <vector>
#include <typeinfo>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;


// ----------------------------------------------------------------------
// struct{FeedForwardImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
FeedForwardImpl::FeedForwardImpl(const size_t dim, const size_t hidden_dim){
    this->mlp = nn::Sequential(
        nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)dim})),
        nn::Linear(dim, hidden_dim),
        nn::GELU(),
        nn::Dropout(0.1),
        nn::Linear(hidden_dim, dim),
        nn::Dropout(0.1)
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
AttentionImpl::AttentionImpl(const size_t dim, const size_t heads_, const size_t dim_head){
    
    size_t inner_dim = dim_head * heads_;
    this->heads = heads_;
    this->scale = std::pow(dim_head, -0.5);

    this->norm = nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)dim}));
    register_module("Layer Normalization", this->norm);

    this->attend = nn::Softmax(/*dim=*/-1);
    register_module("Softmax", this->attend);

    this->dropout = nn::Dropout(0.1);
    register_module("Dropout", this->dropout);

    this->to_qkv = nn::Linear(nn::LinearOptions(dim, inner_dim * 3).bias(false));
    register_module("Linear", this->to_qkv);

    if ((this->heads == 1) && (dim_head == dim)){
        this->to_out = nn::Sequential(nn::Identity());
    }
    else{
        this->to_out = nn::Sequential(
            nn::Linear(inner_dim, dim),
            nn::Dropout(0.1)
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

    dots = torch::matmul(q, k.transpose(-1, -2)) * this->scale;
    
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
TransformerImpl::TransformerImpl(size_t dim, size_t depth_, size_t heads, size_t dim_head, size_t mlp_dim){

    this->depth = depth_;
    this->norm = nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)dim}));
    register_module("Layer Normalization", this->norm);

    for (size_t i = 0; i < this->depth; i++){
        nn::ModuleList layer;
        layer->push_back(Attention(dim, heads, dim_head));
        layer->push_back(FeedForward(dim, mlp_dim));
        this->layers->push_back(layer);
    }
    register_module("Layers", this->layers);

}


// ---------------------------------------------------------
// struct{TransformerImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor TransformerImpl::forward(torch::Tensor x){

    torch::Tensor out;

    for (size_t i = 0; i < this->depth; i++){
        auto ml = this->layers[i]->as<nn::ModuleList>();
        x = (*ml)[0]->as<Attention>()->forward(x) + x;
        x = (*ml)[1]->as<FeedForward>()->forward(x) + x;
    }
    out = this->norm->forward(x);

    return out;

}


// ----------------------------------------------------------------------
// struct{ViTImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ViTImpl::ViTImpl(po::variables_map &vm){

    this->nc = vm["nc"].as<size_t>();
    this->np = vm["split"].as<size_t>() * vm["split"].as<size_t>();
    this->image_size = vm["size"].as<size_t>() - vm["size"].as<size_t>() % vm["split"].as<size_t>();
    this->patch_size = this->image_size / vm["split"].as<size_t>();
    this->dim = vm["dim"].as<size_t>();

    // Linear Projection of Flattened Patches
    this->linear_projection = nn::Sequential(
        nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)(this->patch_size*this->patch_size*this->nc)})),
        nn::Linear(this->patch_size*this->patch_size*this->nc, this->dim),
        nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)this->dim}))
    );
    register_module("Linear Projection", this->linear_projection);

    // Class Token and Positional Encoding
    this->class_token = register_parameter("Class Token", torch::randn({1, 1, (long int)this->dim}));
    this->pos_encoding = register_parameter("Positional Encoding", torch::randn({1, (long int)this->np + 1, (long int)this->dim}));
    this->dropout = nn::Dropout(0.1);
    register_module("Dropout", this->dropout);

    // Transformer
    this->transformer = Transformer(this->dim, vm["depth"].as<size_t>(), vm["heads"].as<size_t>(), vm["dim_head"].as<size_t>(), vm["mlp_dim"].as<size_t>());
    register_module("Transformer", this->transformer);

    // Multi Layer Perceptron (Head)
    this->mlp_head = nn::Linear(/*in_channels=*/this->dim, /*out_channels=*/vm["class_num"].as<size_t>());
    register_module("MLP Head", this->mlp_head);

}


// ---------------------------------------------------------
// struct{ViTImpl}(nn::Module) -> function{init}
// ---------------------------------------------------------
void ViTImpl::init(){
    this->apply(weights_init);
    return;
}


// ---------------------------------------------------------
// struct{ViTImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor ViTImpl::forward(torch::Tensor x){

    torch::Tensor p, pf, cf, out; 

    // (1) Resize
    x = F::interpolate(x, F::InterpolateFuncOptions().size(std::vector<long int>{(long int)this->image_size, (long int)this->image_size}).mode(torch::kBilinear).align_corners(false));  // {N,C,H,W}

    // (2) Flatten patches
    p = x.unfold(/*dim=*/2, this->patch_size, this->patch_size).unfold(/*dim=*/3, this->patch_size, this->patch_size);  // {N,C,H,W} ===> {N,C,NPH,NPW,PH,PW}
    p = p.permute({0, 2, 3, 4, 5, 1}).contiguous().view({x.size(0), (long int)this->np, (long int)(this->patch_size*this->patch_size*x.size(1))}); // {N,C,NPH,NPW,PH,PW} ===> {N,NPH,NPW,PH,PW,C} ===> {N,NP,PH*PW*C}

    // (3) Convert patches to features
    pf = this->linear_projection->forward(p);  // {N,NP,PH*PW*C} ===> {N,NP,D}

    // (4) Add class token and positional encoding
    pf = torch::cat({this->class_token.expand({pf.size(0), 1, (long int)this->dim}), pf}, /*dim=*/1);  // {N,1,D} + {N,NP,D} ===> {N,1+NP,D}
    pf += this->pos_encoding;  // {N,1+NP,D}
    pf = this->dropout(pf);  // {N,1+NP,D}

    // (5) Apply Transformer
    pf = this->transformer(pf);  // {N,1+NP,D}

    // (6) Apply MLP
    cf = pf.select(/*dim=*/1, /*index=*/0);  // {N,D}
    out = this->mlp_head->forward(cf);  // {N,D} ===> {N,CN}
    out = F::log_softmax(out, /*dim=*/1);

    return out;

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

