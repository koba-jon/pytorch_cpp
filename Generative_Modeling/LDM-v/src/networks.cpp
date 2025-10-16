#include <vector>
#include <tuple>
#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;


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
// struct{ResBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
ResBlockImpl::ResBlockImpl(const size_t in_nc, const size_t out_nc, const size_t time_embed){

    this->conv1 = nn::Sequential(
        nn::GroupNorm(nn::GroupNormOptions(32, in_nc)),
        nn::SiLU(),
        nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 3).stride(1).padding(1).bias(true))
    );
    register_module("conv1", this->conv1);

    this->time_emb_proj = nn::Sequential(
        nn::SiLU(),
        nn::Linear(nn::LinearOptions(time_embed, out_nc))
    );
    register_module("time_emb_proj", this->time_emb_proj);

    this->conv2 = nn::Sequential(
        nn::GroupNorm(nn::GroupNormOptions(32, out_nc)),
        nn::SiLU(),
        nn::Conv2d(nn::Conv2dOptions(out_nc, out_nc, 3).stride(1).padding(1).bias(true))
    );
    register_module("conv2", this->conv2);

    this->use_skip = (in_nc != out_nc);
    if (this->use_skip){
        this->skip_conv = nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 1).stride(1).padding(0).bias(true));
        register_module("skip_conv", this->skip_conv);
    }

}


// ----------------------------------------------------------------------
// struct{ResBlockImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor ResBlockImpl::forward(torch::Tensor x, torch::Tensor v){
    torch::Tensor h, x_skip;
    h = this->conv1->forward(x);
    h = h + this->time_emb_proj->forward(v).unsqueeze(-1).unsqueeze(-1);
    h = this->conv2->forward(h);
    x_skip = this->use_skip ? this->skip_conv->forward(x) : x;
    return h + x_skip;
}


// ----------------------------------------------------------------------
// struct{AttentionBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
AttentionBlockImpl::AttentionBlockImpl(const size_t nc, const size_t heads_){

    this->heads = heads_;

    this->norm = nn::GroupNorm(nn::GroupNormOptions(32, nc));
    register_module("norm", this->norm);

    this->to_qkv = nn::Conv2d(nn::Conv2dOptions(nc, nc * 3, 1).stride(1).padding(0).bias(true));
    register_module("to_qkv", this->to_qkv);

    this->proj_out = nn::Conv2d(nn::Conv2dOptions(nc, nc, 1).stride(1).padding(0).bias(true));
    register_module("proj_out", this->proj_out);

}


// ----------------------------------------------------------------------
// struct{AttentionBlockImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor AttentionBlockImpl::forward(torch::Tensor x){

    torch::Tensor q, k, v, attn, out;
    std::vector<torch::Tensor> qkv;

    qkv = this->to_qkv->forward(this->norm->forward(x)).chunk(3, 1);
    q = qkv[0].contiguous().view({x.size(0), (long int)this->heads, (long int)(x.size(1) / this->heads), x.size(2) * x.size(3)}).permute({0, 1, 3, 2});
    k = qkv[1].contiguous().view({x.size(0), (long int)this->heads, (long int)(x.size(1) / this->heads), x.size(2) * x.size(3)});
    v = qkv[2].contiguous().view({x.size(0), (long int)this->heads, (long int)(x.size(1) / this->heads), x.size(2) * x.size(3)}).permute({0, 1, 3, 2});
    
    attn = torch::matmul(q, k);
    attn = attn / std::sqrt((double)(x.size(1) / this->heads));
    attn = torch::softmax(attn, -1);
    out = torch::matmul(attn, v);

    out = out.permute({0, 1, 3, 2}).view(x.sizes());
    out = this->proj_out->forward(out);

    return x + out;

}


// ----------------------------------------------------------------------
// struct{UNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
UNetImpl::UNetImpl(po::variables_map &vm){

    this->max_level = 4;
    this->max_block = 2;
    this->time_embed = vm["nt"].as<size_t>();
    long int nf = vm["nf"].as<size_t>();
    long int nz = vm["nz"].as<size_t>();

    size_t in_nc, out_nc, mult;
    std::vector<size_t> input_block_nc;

    this->time_mlp = nn::Sequential(
        nn::Linear(this->time_embed, this->time_embed),
        nn::SiLU(),
        nn::Linear(this->time_embed, this->time_embed)
    );
    register_module("time_mlp", this->time_mlp);

    this->input_conv = nn::Conv2d(nn::Conv2dOptions(nz, nf, 3).stride(1).padding(1).bias(true));
    register_module("input_conv", this->input_conv);

    mult = 1;
    in_nc = nf;
    input_block_nc.push_back(in_nc);
    for (long int level = 0; level < max_level; level++){

        out_nc = nf * mult;
        for (long int block = 0; block < max_block; block++){

            ResBlock resblock = ResBlock(in_nc, out_nc, this->time_embed);
            register_module("down_resblock_" + std::to_string(this->down_resblocks.size()), resblock);
            this->down_resblocks.push_back(resblock);

            if (level != max_level - 1){
                AttentionBlock attn = AttentionBlock(out_nc, 1);
                register_module("down_attn_" + std::to_string(this->down_attentions.size()), attn);
                this->down_attentions.push_back(attn);
            }

            in_nc = out_nc;
            input_block_nc.push_back(out_nc);

        }

        if (level != max_level - 1){
            nn::Conv2d down = nn::Conv2d(nn::Conv2dOptions(out_nc, out_nc, 4).stride(2).padding(1).bias(true));
            register_module("downsample_" + std::to_string(this->downsample_layers.size()), down);
            this->downsample_layers.push_back(down);
            input_block_nc.push_back(out_nc);
        }
        mult *= 2;

    }

    this->mid_block1 = ResBlock(out_nc, out_nc, this->time_embed);
    register_module("mid_block1", this->mid_block1);
    this->mid_attn = AttentionBlock(out_nc, 1);
    register_module("mid_attn", this->mid_attn);
    this->mid_block2 = ResBlock(out_nc, out_nc, this->time_embed);
    register_module("mid_block2", this->mid_block2);

    mult /= 2;
    for (long int level = max_level - 1; level >= 0; level--){

        out_nc = nf * mult;
        for (long int block = 0; block < max_block + 1; block++){

            ResBlock resblock = ResBlock(in_nc + input_block_nc.back(), out_nc, this->time_embed);
            register_module("up_resblock_" + std::to_string(this->up_resblocks.size()), resblock);
            this->up_resblocks.push_back(resblock);

            if (level != max_level - 1){
                AttentionBlock attn = AttentionBlock(out_nc, 1);
                register_module("up_attn_" + std::to_string(this->up_attentions.size()), attn);
                this->up_attentions.push_back(attn);
            }

            in_nc = out_nc;
            input_block_nc.pop_back();

        }

        if (level != 0){
            nn::ConvTranspose2d up = nn::ConvTranspose2d(nn::ConvTranspose2dOptions(out_nc, out_nc, 4).stride(2).padding(1).bias(true));
            register_module("upsample_" + std::to_string(this->upsample_layers.size()), up);
            this->upsample_layers.push_back(up);
        }
        mult /= 2;

    }

    this->out_conv = nn::Sequential(
        nn::GroupNorm(nn::GroupNormOptions(32, out_nc)),
        nn::SiLU(),
        nn::Conv2d(nn::Conv2dOptions(out_nc, nz, 3).stride(1).padding(1).bias(true))
    );
    register_module("out_conv", this->out_conv);

}


// --------------------------------------------------------------
// struct{UNetImpl}(nn::Module) -> function{timestep_embedding}
// --------------------------------------------------------------
torch::Tensor UNetImpl::timestep_embedding(torch::Tensor t, long int dim){
    torch::Tensor freq, args, v;
    freq = torch::exp(torch::arange(0, dim / 2, torch::kFloat).to(t.device()) * (-std::log(10000.0) / (dim / 2 - 1)));
    args = t.unsqueeze(1) * freq.unsqueeze(0);
    v = torch::cat({torch::cos(args), torch::sin(args)}, -1);
    if (dim % 2 == 1) v = torch::cat({v, torch::zeros({v.size(0), 1}).to(v.device())}, -1);
    return v;
}


// ----------------------------------------------------------------------
// struct{UNetImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor UNetImpl::forward(torch::Tensor x, torch::Tensor t){

    torch::Tensor v, h, out;
    std::vector<torch::Tensor> hs;
    size_t down_res_idx, down_attn_idx, downsample_idx, up_res_idx, up_attn_idx, upsample_idx;

    v = this->timestep_embedding(t, this->time_embed);
    v = this->time_mlp->forward(v);
    h = this->input_conv->forward(x);
    hs.push_back(h);

    down_res_idx = 0;
    down_attn_idx = 0;
    downsample_idx = 0;
    for (long int level = 0; level < this->max_level; level++){
        for (long int block = 0; block < this->max_block; block++){
            h = this->down_resblocks[down_res_idx++]->forward(h, v);
            if (level != max_level - 1){
                h = this->down_attentions[down_attn_idx++]->forward(h);
            }
            hs.push_back(h);
        }
        if (level != max_level - 1){
            h = this->downsample_layers[downsample_idx++]->forward(h);
            hs.push_back(h);
        }
    }

    h = this->mid_block1->forward(h, v);
    h = this->mid_attn->forward(h);
    h = this->mid_block2->forward(h, v);

    up_res_idx = 0;
    up_attn_idx = 0;
    upsample_idx = 0;
    for (long int level = this->max_level - 1; level >= 0; level--){
        for (long int block = 0; block < this->max_block + 1; block++){
            h = torch::cat({h, hs.back()}, 1);
            h = this->up_resblocks[up_res_idx++]->forward(h, v);
            if (level != max_level - 1){
                h = this->up_attentions[up_attn_idx++]->forward(h);
            }
            hs.pop_back();
        }
        if (level != 0){
            h = this->upsample_layers[upsample_idx++]->forward(h);
        }
    }

    out = this->out_conv->forward(h);

    return out;

}


// ---------------------------------------------
// struct{LDMImpl}(nn::Module) -> constructor
// ---------------------------------------------
LDMImpl::LDMImpl(po::variables_map &vm, torch::Device device){

    this->nc = vm["nc"].as<size_t>();
    this->size = vm["size"].as<size_t>();
    this->timesteps = vm["timesteps"].as<size_t>();
    this->timesteps_inf = vm["timesteps_inf"].as<size_t>();
    this->eta = vm["eta"].as<float>();
    this->betas = torch::linspace(vm["beta_start"].as<float>(), vm["beta_end"].as<float>(), this->timesteps).to(device);  // {T} (0.0001, 0.00012, 0.00014, ..., 0.01996, 0.01998, 0.02)
    this->betas = torch::cat({torch::zeros({1}).to(device), this->betas}, /*dim=*/0);  // {T+1} (0.0, 0.0001, 0.00012, 0.00014, ..., 0.01996, 0.01998, 0.02)
    this->alphas = 1.0 - this->betas;  // {T+1} (1.0, 0.9999, 0.99988, 0.99986, ..., 0.98004, 0.98002, 0.98)
    this->alpha_bars = torch::cumprod(this->alphas, /*dim=*/0);  // {T+1} (1.0, 0.9999, 0.99978, 0.99964, ..., 0.000042, 0.000041, 0.00004)

    this->ae = AutoEncoder(vm);
    register_module("AutoEncoder", this->ae);

    this->model = UNet(vm);
    register_module("U-Net", this->model);

}


// -----------------------------------------------------
// struct{LDMImpl}(nn::Module) -> function{add_noise}
// -----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> LDMImpl::add_noise(torch::Tensor z_0, torch::Tensor t){

    torch::Tensor alpha_bar, noise, z_t, v;
    
    alpha_bar = this->alpha_bars.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});  // {N,1,1,1}
    noise = torch::randn_like(z_0);
    z_t = torch::sqrt(alpha_bar) * z_0 + torch::sqrt(1.0 - alpha_bar) * noise;
    v = torch::sqrt(alpha_bar) * noise - torch::sqrt(1.0 - alpha_bar) * z_0;

    return {z_t, v};

}


// ---------------------------------------------------
// struct{LDMImpl}(nn::Module) -> function{denoise}
// ---------------------------------------------------
torch::Tensor LDMImpl::denoise(torch::Tensor z_t, torch::Tensor t, torch::Tensor t_prev){

    bool is_training;
    torch::Tensor alpha, alpha_bar, alpha_bar_prev, v, eps, noise, mu, sigma, out;
    torch::Tensor z_0;

    alpha = this->alphas.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});
    alpha_bar = this->alpha_bars.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});
    alpha_bar_prev = this->alpha_bars.index_select(/*dim=*/0, t_prev).view({-1, 1, 1, 1});

    is_training = this->model->is_training();
    this->model->eval();
    v = this->model->forward(z_t, t);
    if (is_training) this->model->train();
    
    eps = torch::sqrt(alpha_bar) * v + torch::sqrt(1.0 - alpha_bar) * z_t;
    noise = torch::randn_like(z_t).to(z_t.device());
    noise.masked_fill_(t.view({-1, 1, 1, 1}).expand_as(noise) == 1, 0.0);
    z_0 = (z_t - torch::sqrt(1.0 - alpha_bar) * eps) / torch::sqrt(alpha_bar);
    sigma = this->eta * torch::sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar)) * torch::sqrt(1.0 - alpha_bar / alpha_bar_prev);
    out = torch::sqrt(alpha_bar_prev) * z_0 + torch::sqrt(1.0 - alpha_bar_prev - sigma * sigma) * eps + sigma * noise;

    return out;

}


// -----------------------------------------------------
// struct{LDMImpl}(nn::Module) -> function{denoise_t}
// -----------------------------------------------------
torch::Tensor LDMImpl::denoise_t(torch::Tensor z_t, torch::Tensor t){

    bool is_training;
    torch::Tensor alpha_bar, v, out;

    is_training = this->model->is_training();
    this->model->eval();
    v = this->model->forward(z_t, t);
    if (is_training) this->model->train();

    alpha_bar = this->alpha_bars.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});
    out = torch::sqrt(alpha_bar) * z_t - torch::sqrt(1.0 - alpha_bar) * v;

    return out;

}


// ------------------------------------------------------
// struct{LDMImpl}(nn::Module) -> function{get_z_shape}
// ------------------------------------------------------
std::vector<long int> LDMImpl::get_z_shape(torch::Device &device){
    torch::Tensor x, z;
    std::vector<long int> z_shape;
    x = torch::randn({1, this->nc, this->size, this->size}).to(device);
    z = this->ae->encode(x);
    z_shape = {z.size(0), z.size(1), z.size(2), z.size(3)};
    return z_shape;
}


// ---------------------------------------------------
// struct{LDMImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LDMImpl::forward(torch::Tensor x_0, torch::Tensor t){
    torch::Tensor z_0, z_t, v, pred_v, rec;
    z_0 = this->ae->encode(x_0);
    std::tie(z_t, v) = this->add_noise(z_0.detach(), t);
    pred_v = this->model->forward(z_t, t);
    rec = this->ae->decode(z_0);
    return {pred_v, v, rec, z_t};
}


// -----------------------------------------------------
// struct{LDMImpl}(nn::Module) -> function{forward_z}
// -----------------------------------------------------
torch::Tensor LDMImpl::forward_z(torch::Tensor z){
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
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::BatchNorm2d)) || (typeid(m) == typeid(nn::BatchNorm2dImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::constant_(*w, /*weight=*/1.0);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}

