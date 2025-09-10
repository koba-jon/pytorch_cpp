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
// struct{DownSamplingImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
DownSamplingImpl::DownSamplingImpl(const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const size_t time_embed){

    this->convs->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 3).stride(1).padding(1).bias(!BN)));
    if (BN){
        this->convs->push_back(nn::BatchNorm2d(out_nc));
    }
    if (ReLU){
        this->convs->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    this->convs->push_back(nn::Conv2d(nn::Conv2dOptions(out_nc, out_nc, 4).stride(2).padding(1).bias(false)));
    this->convs->push_back(nn::BatchNorm2d(out_nc));
    this->convs->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    register_module("DownConvolutions", this->convs);

    this->mlp = nn::Sequential(
        nn::Linear(time_embed, in_nc),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Linear(in_nc, in_nc)
    );
    register_module("MLP", this->mlp);

}


// ----------------------------------------------------------------------
// struct{DownSamplingImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor DownSamplingImpl::forward(torch::Tensor x, torch::Tensor v){
    torch::Tensor y;
    v = this->mlp->forward(v).unsqueeze(/*dim=*/-1).unsqueeze(/*dim=*/-1);
    y = this->convs->forward(x + v);
    return y;
}


// ----------------------------------------------------------------------
// struct{UpSamplingImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
UpSamplingImpl::UpSamplingImpl(const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const size_t time_embed){

    this->convs->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, in_nc, 3).stride(1).padding(1).bias(false)));
    this->convs->push_back(nn::BatchNorm2d(in_nc));
    this->convs->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    this->convs->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(in_nc, out_nc, 4).stride(2).padding(1).bias(!BN)));
    if (BN){
        this->convs->push_back(nn::BatchNorm2d(out_nc));
    }
    if (ReLU){
        this->convs->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    register_module("UpConvolutions", this->convs);

    this->mlp = nn::Sequential(
        nn::Linear(time_embed, in_nc),
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Linear(in_nc, in_nc)
    );
    register_module("MLP", this->mlp);

}


// ----------------------------------------------------------------------
// struct{UpSamplingImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor UpSamplingImpl::forward(torch::Tensor x, torch::Tensor v){
    torch::Tensor y;
    v = this->mlp->forward(v).unsqueeze(/*dim=*/-1).unsqueeze(/*dim=*/-1);
    y = this->convs->forward(x + v);
    return y;
}


// ----------------------------------------------------------------------
// struct{UNetBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
UNetBlockImpl::UNetBlockImpl(const std::pair<size_t, size_t> outside_nc, const size_t inside_nc, UNetBlockImpl &submodule, const bool outermost_, const bool innermost_, const size_t time_embed){

    this->outermost = outermost_;
    this->innermost = innermost_;

    if (this->outermost){  // {IC,256,256} ===> {F,128,128} ===> ... ===> {2F,128,128} ===> {OC,256,256}
        this->down->push_back(DownSampling(outside_nc.first, inside_nc, /*BN=*/false, /*ReLU=*/true, /*time_embed=*/time_embed));  // {IC,256,256} ===> {F,128,128}
        this->sub->push_back(submodule);
        this->up->push_back(UpSampling(inside_nc*2, inside_nc, /*BN=*/true, /*ReLU=*/true, /*time_embed=*/time_embed));  // {2F,128,128} ===> {OC,256,256}
        this->up->push_back(nn::Conv2d(nn::Conv2dOptions(inside_nc, outside_nc.second, 3).stride(1).padding(1).bias(true)));
    }
    else if (this->innermost){   // {8F,2,2} ===> {Z,1,1} ===> {8F,2,2}
        this->down->push_back(DownSampling(outside_nc.first, inside_nc, /*BN=*/false, /*ReLU=*/false, /*time_embed=*/time_embed));  // {8F,2,2} ===> {Z,1,1}
        this->sub->push_back(nn::Identity());
        this->up->push_back(UpSampling(inside_nc, outside_nc.second, /*BN=*/true, /*ReLU=*/true, /*time_embed=*/time_embed));     // {Z,1,1} ===> {8F,2,2}
    }
    else{                  // {NF,H,W} ===> {NF,H/2,W/2} ===> ... ===> {2NF,H/2,W/2} ===> {NF,H,W}
        this->down->push_back(DownSampling(outside_nc.first, inside_nc, /*BN=*/true, /*ReLU=*/true, /*time_embed=*/time_embed));    // {NF,H,W} ===> {NF,H/2,W/2}
        this->sub->push_back(submodule);
        this->up->push_back(UpSampling(inside_nc*2, outside_nc.second, /*BN=*/true, /*ReLU=*/true, /*time_embed=*/time_embed));   // {2NF,H/2,W/2} ===> {NF,H,W}
    }
    register_module("DownSampling", this->down);
    register_module("SubModule", this->sub);
    register_module("UpSampling", this->up);

}


// ----------------------------------------------------------------------
// struct{UNetBlockImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor UNetBlockImpl::forward(torch::Tensor x, torch::Tensor v){
    torch::Tensor mid1, mid2, out;
    if (this->outermost){
        mid1 = this->down->forward(x, v);
        mid2 = this->sub->forward(mid1, v);
        mid2 = torch::cat({mid1, mid2}, /*dim=*/1);
        out = this->up->forward(mid2, v);
    }
    else if (this->innermost){
        mid1 = this->down->forward(x, v);
        mid2 = this->sub->forward(mid1);
        out = this->up->forward(mid2, v);
    }
    else{
        mid1 = this->down->forward(x, v);
        mid2 = this->sub->forward(mid1, v);
        mid2 = torch::cat({mid1, mid2}, /*dim=*/1);
        out = this->up->forward(mid2, v);
    }
    return out;
}


// ----------------------------------------------------------------------
// struct{UNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
UNetImpl::UNetImpl(po::variables_map &vm){
    
    size_t feature = vm["nf"].as<size_t>();
    size_t num_downs = (size_t)(std::log2(vm["size"].as<size_t>()));
    this->time_embed = vm["nt"].as<size_t>();

    UNetBlockImpl blocks, fake;
    blocks = UNetBlockImpl({feature*8, feature*8}, feature*8, /*submodule_=*/fake, /*outermost_=*/false, /*innermost_=*/true, /*time_embed=*/this->time_embed);
    for (size_t i = 0; i < num_downs - 5; i++){
        blocks = UNetBlockImpl({feature*8, feature*8}, feature*8, /*submodule_=*/blocks, /*outermost_=*/false, /*innermost_=*/false, /*time_embed=*/this->time_embed);
    }
    blocks = UNetBlockImpl({feature*4, feature*4}, feature*8, /*submodule_=*/blocks, /*outermost_=*/false, /*innermost_=*/false, /*time_embed=*/this->time_embed);
    blocks = UNetBlockImpl({feature*2, feature*2}, feature*4, /*submodule_=*/blocks, /*outermost_=*/false, /*innermost_=*/false, /*time_embed=*/this->time_embed);
    blocks = UNetBlockImpl({feature, feature}, feature*2, /*submodule_=*/blocks, /*outermost_=*/false, /*innermost_=*/false, /*time_embed=*/this->time_embed);
    blocks = UNetBlockImpl({vm["nc"].as<size_t>(), vm["nc"].as<size_t>()}, feature, /*submodule_=*/blocks, /*outermost_=*/true, /*innermost_=*/false, /*time_embed=*/this->time_embed);
    
    this->model->push_back(blocks);
    register_module("U-Net", this->model);

}


// --------------------------------------------------------
// struct{UNetImpl}(nn::Module) -> function{pos_encoding}
// --------------------------------------------------------
torch::Tensor UNetImpl::pos_encoding(torch::Tensor t, long int dim){
    long int batch_size;
    torch::Tensor v, tt, vv, ii, div_term, idx0, idx1;
    torch::Device device(t.device());
    batch_size = t.size(0);
    v = torch::zeros({batch_size, dim}).to(device);
    for (long int i = 0; i < batch_size; i++){
        tt = t[i];
        vv = torch::zeros(dim).to(device);
        ii = torch::arange(0, dim).to(device);
        div_term = torch::exp(ii / float(dim) * std::log(10000.0));
        idx0 = torch::arange(0, vv.size(0), 2, torch::kLong).to(device);
        idx1 = torch::arange(1, vv.size(0), 2, torch::kLong).to(device);
        vv.index_put_({idx0}, torch::sin(tt / div_term.index_select(/*dim=*/0, idx0)));
        vv.index_put_({idx1}, torch::cos(tt / div_term.index_select(/*dim=*/0, idx1)));
        v[i] = vv;
    }
    return v;
}


// ----------------------------------------------------------------------
// struct{UNetImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor UNetImpl::forward(torch::Tensor x, torch::Tensor t){
    torch::Tensor v, out;
    v = this->pos_encoding(t, this->time_embed);
    out = this->model->forward(x, v);  // {C,256,256} ===> {C,256,256}
    return out;
}


// ---------------------------------------------
// struct{DDPMImpl}(nn::Module) -> constructor
// ---------------------------------------------
DDPMImpl::DDPMImpl(po::variables_map &vm, torch::Device device){

    this->timesteps = vm["timesteps"].as<size_t>();
    this->betas = torch::linspace(vm["beta_start"].as<float>(), vm["beta_end"].as<float>(), this->timesteps + 1).to(device);  // {T+1} (0.0001, 0.00012, 0.00014, ..., 0.01996, 0.01998, 0.02)
    this->alphas = 1.0 - this->betas;  // {T+1} (0.9999, 0.99988, 0.99986, ..., 0.98004, 0.98002, 0.98)
    this->alpha_bars = torch::cumprod(this->alphas, /*dim=*/0);  // {T+1} (0.9999, 0.99978, 0.99964, ..., 0.000042, 0.000041, 0.00004)

    this->model = UNet(vm);
    register_module("U-Net", this->model);

}


// -----------------------------------------------------
// struct{DDPMImpl}(nn::Module) -> function{add_noise}
// -----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> DDPMImpl::add_noise(torch::Tensor x_0, torch::Tensor t){

    torch::Tensor alpha_bar, noise, x_t;
    
    alpha_bar = this->alpha_bars.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});  // {N,1,1,1}
    noise = torch::randn_like(x_0).to(x_0.device());
    x_t = torch::sqrt(alpha_bar) * x_0 + torch::sqrt(1.0 - alpha_bar) * noise;

    return {x_t, noise};

}


// ---------------------------------------------------
// struct{DDPMImpl}(nn::Module) -> function{denoise}
// ---------------------------------------------------
torch::Tensor DDPMImpl::denoise(torch::Tensor x_t, torch::Tensor t){

    bool is_training;
    torch::Tensor alpha, alpha_bar, alpha_bar_prev, eps, noise, mu, sigma, out;

    alpha = this->alphas.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});
    alpha_bar = this->alpha_bars.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});
    alpha_bar_prev = this->alpha_bars.index_select(/*dim=*/0, t - 1).view({-1, 1, 1, 1});

    is_training = this->model->is_training();
    this->model->eval();
    eps = this->model->forward(x_t, t);
    if (is_training) this->model->train();
    
    noise = torch::randn_like(x_t).to(x_t.device());
    noise.masked_fill_(t.view({-1, 1, 1, 1}).expand_as(noise) == 1, 0.0);
    mu = (x_t - ((1.0 - alpha) / torch::sqrt(1.0 - alpha_bar)) * eps) / torch::sqrt(alpha);
    sigma = torch::sqrt((1.0 - alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar));
    out = mu + noise * sigma;

    return out;
    
}


// -----------------------------------------------------
// struct{DDPMImpl}(nn::Module) -> function{denoise_t}
// -----------------------------------------------------
torch::Tensor DDPMImpl::denoise_t(torch::Tensor x_t, torch::Tensor t){

    bool is_training;
    torch::Tensor alpha_bar, eps, out;

    is_training = this->model->is_training();
    this->model->eval();
    eps = this->model->forward(x_t, t);
    if (is_training) this->model->train();

    alpha_bar = this->alpha_bars.index_select(/*dim=*/0, t).view({-1, 1, 1, 1});
    out = (x_t - torch::sqrt(1.0 - alpha_bar) * eps) / torch::sqrt(alpha_bar);    // {C,256,256} ===> {C,256,256}

    return out;

}


// ---------------------------------------------------
// struct{DDPMImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------
torch::Tensor DDPMImpl::forward(torch::Tensor x_t, torch::Tensor t){
    torch::Tensor out = this->model->forward(x_t, t);    // {C,256,256} ===> {C,256,256}
    return out;
}


// -----------------------------------------------------
// struct{DDPMImpl}(nn::Module) -> function{forward_z}
// -----------------------------------------------------
torch::Tensor DDPMImpl::forward_z(torch::Tensor z){
    torch::Tensor x, t;
    x = z;
    for (size_t i = this->timesteps; i > 0; i--){   
        t = torch::full({z.size(0)}, /*value=*/i, torch::TensorOptions().dtype(torch::kLong)).to(z.device());
        x = this->denoise(x, t);  // {C,256,256} ===> {C,256,256}
    }
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

