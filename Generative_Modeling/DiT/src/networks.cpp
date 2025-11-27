#include <algorithm>
#include <cmath>
#include <typeinfo>
// For External Library
#include <torch/torch.h>

// For Original Header
#include "networks.hpp"

namespace nn = torch::nn;
namespace F = torch::nn::functional;


// ----------------------------------------------------------------------
// struct{AutoEncoderImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
AutoEncoderImpl::AutoEncoderImpl(po::variables_map &vm){

    size_t nc = vm["nc"].as<size_t>();
    size_t nf = vm["nf"].as<size_t>();
    size_t nz = vm["nz"].as<size_t>();
    long int downsample_level = vm["ae_downsample"].as<size_t>();
    std::vector<size_t> channels;
    size_t mult;

    channels = std::vector<size_t>(downsample_level + 1);
    for (long int i = 0; i <= downsample_level; i++){
        mult = std::min<size_t>(std::pow(2, i), 8);
        channels[i] = nf * mult;
    }

    this->encoder->push_back(nn::Conv2d(nn::Conv2dOptions(nc, channels[0], 3).stride(1).padding(1).bias(true)));
    this->encoder->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    for (long int i = 0; i < downsample_level; i++){
        this->encoder->push_back(nn::Conv2d(nn::Conv2dOptions(channels[i], channels[i + 1], 4).stride(2).padding(1).bias(false)));
        this->encoder->push_back(nn::BatchNorm2d(channels[i + 1]));
        this->encoder->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    this->encoder->push_back(nn::Conv2d(nn::Conv2dOptions(channels[downsample_level], nz, 3).stride(1).padding(1).bias(true)));
    register_module("Encoder", this->encoder);

    this->decoder->push_back(nn::Conv2d(nn::Conv2dOptions(nz, channels[downsample_level], 3).stride(1).padding(1).bias(true)));
    this->decoder->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    for (long int i = downsample_level; i > 0; i--){
        this->decoder->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(channels[i], channels[i - 1], 4).stride(2).padding(1).bias(false)));
        this->decoder->push_back(nn::BatchNorm2d(channels[i - 1]));
        this->decoder->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    this->decoder->push_back(nn::Conv2d(nn::Conv2dOptions(channels[0], nc, 3).stride(1).padding(1).bias(true)));
    this->decoder->push_back(nn::Tanh());
    register_module("Decoder", this->decoder);

}


// ----------------------------------------------------------------------
// struct{AutoEncoderImpl}(nn::Module) -> function{encode}
// ----------------------------------------------------------------------
torch::Tensor AutoEncoderImpl::encode(torch::Tensor x){
    torch::Tensor z;
    z = this->encoder->forward(x);
    return z;
}


// ----------------------------------------------------------------------
// struct{AutoEncoderImpl}(nn::Module) -> function{decode}
// ----------------------------------------------------------------------
torch::Tensor AutoEncoderImpl::decode(torch::Tensor z){
    torch::Tensor x;
    x = this->decoder->forward(z);
    return x;
}


// ----------------------------------------------------------------------
// struct{AutoEncoderImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor AutoEncoderImpl::forward(torch::Tensor x){
    torch::Tensor z, out;
    z = this->encode(x);
    out = this->decode(z);
    return out;
}


// ----------------------------------------------------------------------
// struct{FeedForwardImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
FeedForwardImpl::FeedForwardImpl(const size_t dim, const size_t hidden_dim){

    this->mlp = nn::Sequential(
        nn::Linear(dim, hidden_dim),
        nn::GELU(),
        nn::Dropout(0.1),
        nn::Linear(hidden_dim, dim),
        nn::Dropout(0.1)
    );
    register_module("Feed Forward", this->mlp);

    auto linear1 = this->mlp->ptr<nn::LinearImpl>(0);
    nn::init::normal_(linear1->weight, /*mean=*/0.0, /*std=*/0.01);
    nn::init::constant_(linear1->bias, 0.0);

    auto linear2 = this->mlp->ptr<nn::LinearImpl>(3);
    nn::init::normal_(linear2->weight, /*mean=*/0.0, /*std=*/0.01);
    nn::init::constant_(linear2->bias, 0.0);

}


// ---------------------------------------------------------
// struct{FeedForwardImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor FeedForwardImpl::forward(torch::Tensor x){
    return this->mlp->forward(x);
}


// ----------------------------------------------------------------------
// struct{AttentionImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
AttentionImpl::AttentionImpl(const size_t dim, const size_t heads_, const size_t dim_head){

    size_t inner_dim = dim_head * heads_;
    this->heads = heads_;
    this->scale = std::pow(dim_head, -0.5);

    this->attend = nn::Softmax(/*dim=*/-1);
    register_module("Softmax", this->attend);

    this->dropout = nn::Dropout(0.1);
    register_module("Dropout", this->dropout);

    this->to_qkv = nn::Linear(nn::LinearOptions(dim, inner_dim * 3).bias(false));
    nn::init::normal_(this->to_qkv->weight, /*mean=*/0.0, /*std=*/0.01);
    register_module("Linear", this->to_qkv);

    if ((this->heads == 1) && (dim_head == dim)){
        this->to_out = nn::Sequential(nn::Identity());
    }
    else{
        this->to_out = nn::Sequential(
            nn::Linear(inner_dim, dim),
            nn::Dropout(0.1)
        );
        auto linear = this->to_out->ptr<nn::LinearImpl>(0);
        nn::init::normal_(linear->weight, /*mean=*/0.0, /*std=*/0.01);
        nn::init::constant_(linear->bias, 0.0);
    }
    register_module("Output Function", this->to_out);

}


// ---------------------------------------------------------
// struct{AttentionImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor AttentionImpl::forward(torch::Tensor x){

    torch::Tensor q, k, v, dots, attn, out;
    std::vector<torch::Tensor> qkv;

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
// struct{DiTBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
DiTBlockImpl::DiTBlockImpl(const size_t dim, const size_t heads, const size_t dim_head, const size_t mlp_dim){

    this->norm1 = nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)dim}));
    register_module("LayerNorm1", this->norm1);

    this->attn = Attention(dim, heads, dim_head);
    register_module("Attention", this->attn);

    this->norm2 = nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)dim}));
    register_module("LayerNorm2", this->norm2);

    this->ff = FeedForward(dim, mlp_dim);
    register_module("FeedForward", this->ff);

    this->adaln = nn::Sequential(
        nn::SiLU(),
        nn::Linear(dim, dim * 6)
    );
    register_module("AdaLN", this->adaln);

    auto linear = this->adaln->ptr<nn::LinearImpl>(1);
    nn::init::constant_(linear->weight, 0.0);
    nn::init::constant_(linear->bias, 0.0);

}


// ---------------------------------------------------------
// struct{DiTBlockImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor DiTBlockImpl::forward(torch::Tensor x, torch::Tensor t){

    torch::Tensor h, shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp;
    std::vector<torch::Tensor> mod_params;

    mod_params = this->adaln->forward(t).chunk(6, /*dim=*/1);
    shift_attn = mod_params[0];
    scale_attn = mod_params[1];
    gate_attn = mod_params[2];
    shift_mlp = mod_params[3];
    scale_mlp = mod_params[4];
    gate_mlp = mod_params[5];

    h = this->norm1->forward(x);
    h = h * (1.0 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1);
    h = this->attn->forward(h);
    x = x + gate_attn.unsqueeze(1) * h;

    h = this->norm2->forward(x);
    h = h * (1.0 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1);
    h = this->ff->forward(h);
    x = x + gate_mlp.unsqueeze(1) * h;

    return x;

}


// ----------------------------------------------------------------------
// struct{DiTBackboneImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
DiTBackboneImpl::DiTBackboneImpl(po::variables_map &vm, std::vector<long int> z_shape){

    size_t nz = vm["nz"].as<size_t>();
    this->z_size = z_shape[2];
    this->patch_size = vm["patch"].as<size_t>();
    this->dim = vm["dim"].as<size_t>();
    this->time_embed_dim = vm["nt"].as<size_t>();
    size_t depth = vm["depth"].as<size_t>();
    size_t heads = vm["heads"].as<size_t>();
    size_t dim_head = vm["dim_head"].as<size_t>();
    size_t mlp_dim = vm["mlp_dim"].as<size_t>();

    size_t grid = this->z_size / this->patch_size;
    this->num_patches = grid * grid;

    this->patch_embed = nn::Conv2d(nn::Conv2dOptions(nz, this->dim, this->patch_size).stride(this->patch_size).padding(0).bias(true));
    register_module("Patch Embedding", this->patch_embed);

    this->pos_embedding = register_parameter("Positional Encoding", torch::randn({1, (long int)this->num_patches, (long int)this->dim}));
    this->dropout = nn::Dropout(0.1);
    register_module("Dropout", this->dropout);

    this->time_mlp = nn::Sequential(
        nn::Linear(this->time_embed_dim, this->dim),
        nn::SiLU(),
        nn::Linear(this->dim, this->dim)
    );
    register_module("Time MLP", this->time_mlp);

    auto linear1 = this->time_mlp->ptr<nn::LinearImpl>(0);
    nn::init::normal_(linear1->weight, /*mean=*/0.0, /*std=*/0.01);
    nn::init::constant_(linear1->bias, 0.0);

    auto linear2 = this->time_mlp->ptr<nn::LinearImpl>(2);
    nn::init::normal_(linear2->weight, /*mean=*/0.0, /*std=*/0.01);
    nn::init::constant_(linear2->bias, 0.0);

    for (size_t i = 0; i < depth; i++){
        this->blocks->push_back(DiTBlock(this->dim, heads, dim_head, mlp_dim));
    }
    register_module("Blocks", this->blocks);

    this->final_norm = nn::LayerNorm(nn::LayerNormOptions(std::vector<long int>{(long int)this->dim}));
    register_module("FinalNorm", this->final_norm);

    this->head = nn::ConvTranspose2d(nn::ConvTranspose2dOptions(this->dim, nz, this->patch_size).stride(this->patch_size).padding(0));
    register_module("Head", this->head);

}


// ---------------------------------------------------------
// struct{DiTBackboneImpl}(nn::Module) -> function{timestep_embedding}
// ---------------------------------------------------------
torch::Tensor DiTBackboneImpl::timestep_embedding(torch::Tensor t, long int dim_){

    torch::Tensor device, half, emb;

    device = torch::ones({1}, torch::TensorOptions().device(t.device()));
    half = torch::arange(dim_ / 2, torch::TensorOptions().device(device.device()).dtype(torch::kFloat)) / (float)(dim_ / 2);
    half = torch::exp(-std::log(10000.0) * half);
    half = t.to(torch::kFloat).unsqueeze(1) * half.unsqueeze(0);
    emb = torch::cat({torch::sin(half), torch::cos(half)}, /*dim=*/1);

    if (dim_ % 2 == 1){
        emb = torch::cat({emb, torch::zeros({emb.size(0), 1}, torch::TensorOptions().device(emb.device()))}, /*dim=*/1);
    }

    return emb;

}


// ---------------------------------------------------------
// struct{DiTBackboneImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
torch::Tensor DiTBackboneImpl::forward(torch::Tensor x, torch::Tensor t){

    torch::Tensor pf, t_embed, out;

    pf = this->patch_embed->forward(x);  // {N,C,H,W} ===> {N,D,H/P,W/P}
    pf = pf.view({pf.size(0), pf.size(1), -1}).transpose(1, 2);  // {N,D,H/P,W/P} ===> {N,NP,D}

    pf = this->dropout(this->pos_embedding + pf);

    t_embed = this->timestep_embedding(t, (long int)this->time_embed_dim);
    t_embed = this->time_mlp->forward(t_embed);  // {N,D}

    for (size_t i = 0; i < this->blocks->size(); i++){
        pf = this->blocks[i]->as<DiTBlock>()->forward(pf, t_embed);
    }

    pf = this->final_norm->forward(pf);
    pf = pf.transpose(1, 2).view({pf.size(0), pf.size(2), (long int)this->z_size / (long int)this->patch_size, (long int)this->z_size / (long int)this->patch_size});

    out = this->head->forward(pf);

    return out;

}


// -----------------------------------------------------
// struct{DiTImpl}(nn::Module) -> constructor
// -----------------------------------------------------
DiTImpl::DiTImpl(po::variables_map &vm, torch::Device device){

    this->pred = vm["pred"].as<char>();
    this->nc = vm["nc"].as<size_t>();
    this->size = vm["size"].as<size_t>();
    this->timesteps = vm["timesteps"].as<size_t>();
    this->timesteps_inf = vm["timesteps_inf"].as<size_t>();
    this->eta = vm["eta"].as<float>();
    this->betas = torch::linspace(vm["beta_start"].as<float>(), vm["beta_end"].as<float>(), this->timesteps).to(device);
    this->betas = torch::cat({torch::zeros({1}).to(device), this->betas}, /*dim=*/0);
    this->alphas = 1.0 - this->betas;
    this->alpha_bars = torch::cumprod(this->alphas, /*dim=*/0);

    this->ae = AutoEncoder(vm);
    this->ae->to(device);
    register_module("AutoEncoder", this->ae);

    this->model = DiTBackbone(vm, this->get_z_shape(device));
    register_module("Diffusion Transformer", this->model);

}


// -----------------------------------------------------
// struct{DiTImpl}(nn::Module) -> function{add_noise}
// -----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> DiTImpl::add_noise(torch::Tensor z_0, torch::Tensor t){

    torch::Tensor alpha_bar, noise, z_t, v;
    
    alpha_bar = this->alpha_bars.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});  // {N,1,1,1}
    noise = torch::randn_like(z_0);
    z_t = torch::sqrt(alpha_bar) * z_0 + torch::sqrt(1.0 - alpha_bar) * noise;
    if (this->pred == 'e') return {z_t, noise};
    v = torch::sqrt(alpha_bar) * noise - torch::sqrt(1.0 - alpha_bar) * z_0;

    return {z_t, v};

}


// ---------------------------------------------------
// struct{DiTImpl}(nn::Module) -> function{denoise}
// ---------------------------------------------------
torch::Tensor DiTImpl::denoise(torch::Tensor z_t, torch::Tensor t, torch::Tensor t_prev){

    bool is_training;
    torch::Tensor alpha, alpha_bar, alpha_bar_prev, v, eps, noise, mu, sigma, out;
    torch::Tensor z_0;

    alpha = this->alphas.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});
    alpha_bar = this->alpha_bars.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});
    alpha_bar_prev = this->alpha_bars.index_select(/*dim=*/0, t_prev).view({-1, 1, 1, 1});

    is_training = this->model->is_training();
    this->model->eval();
    if (this->pred == 'e'){
        eps = this->model->forward(z_t, t);
    }
    else {
        v = this->model->forward(z_t, t);
        eps = torch::sqrt(alpha_bar) * v + torch::sqrt(1.0 - alpha_bar) * z_t;
    }
    if (is_training) this->model->train();
    
    noise = torch::randn_like(z_t).to(z_t.device());
    noise.masked_fill_(t.view({-1, 1, 1, 1}).expand_as(noise) == 1, 0.0);
    z_0 = (z_t - torch::sqrt(1.0 - alpha_bar) * eps) / torch::sqrt(alpha_bar);
    sigma = this->eta * torch::sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar)) * torch::sqrt(1.0 - alpha_bar / alpha_bar_prev);
    out = torch::sqrt(alpha_bar_prev) * z_0 + torch::sqrt(1.0 - alpha_bar_prev - sigma * sigma) * eps + sigma * noise;

    return out;

}


// -----------------------------------------------------
// struct{DiTImpl}(nn::Module) -> function{denoise_t}
// -----------------------------------------------------
torch::Tensor DiTImpl::denoise_t(torch::Tensor z_t, torch::Tensor t){

    bool is_training;
    torch::Tensor alpha_bar, eps, v, out;

    alpha_bar = this->alpha_bars.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});

    is_training = this->model->is_training();
    this->model->eval();
    if (this->pred == 'e'){
        eps = this->model->forward(z_t, t);
        out = (z_t - torch::sqrt(1.0 - alpha_bar) * eps) / torch::sqrt(alpha_bar);
    }
    else{
        v = this->model->forward(z_t, t);
        out = torch::sqrt(alpha_bar) * z_t - torch::sqrt(1.0 - alpha_bar) * v;
    }
    if (is_training) this->model->train();

    return out;

}


// ------------------------------------------------------
// struct{DiTImpl}(nn::Module) -> function{get_z_shape}
// ------------------------------------------------------
std::vector<long int> DiTImpl::get_z_shape(torch::Device &device){
    torch::Tensor x, z;
    std::vector<long int> z_shape;
    x = torch::randn({1, this->nc, this->size, this->size}).to(device);
    z = this->ae->encode(x);
    z_shape = {z.size(0), z.size(1), z.size(2), z.size(3)};
    return z_shape;
}


// ---------------------------------------------------
// struct{DiTImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> DiTImpl::forward(torch::Tensor x_0, torch::Tensor t){
    torch::Tensor z_0, z_t, target, pred, rec;
    z_0 = this->ae->encode(x_0);
    std::tie(z_t, target) = this->add_noise(z_0.detach(), t);
    pred = this->model->forward(z_t, t);
    rec = this->ae->decode(z_0);
    return {pred, target, rec, z_t};
}


// -----------------------------------------------------
// struct{DiTImpl}(nn::Module) -> function{forward_z}
// -----------------------------------------------------
torch::Tensor DiTImpl::forward_z(torch::Tensor z){
    torch::Tensor x, t, t_prev;
    for (long int i = this->timesteps_inf; i > 0; i--){   
        t = torch::full({z.size(0)}, /*value=*/(size_t)(float(i) / this->timesteps_inf * this->timesteps + 0.5), torch::TensorOptions().dtype(torch::kLong)).to(z.device());
        t_prev = torch::full({z.size(0)}, /*value=*/(size_t)(float(i - 1) / this->timesteps_inf * this->timesteps + 0.5), torch::TensorOptions().dtype(torch::kLong)).to(z.device());
        z = this->denoise(z, t, t_prev);  // {C,256,256} ===> {C,256,256}
    }
    x = this->ae->decode(z);
    return x;
}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl)) || (typeid(m) == typeid(nn::ConvTranspose2d)) || (typeid(m) == typeid(nn::ConvTranspose2dImpl))) {
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

