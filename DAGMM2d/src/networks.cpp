#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <iomanip>
#include <typeinfo>
#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Constant
#define PI 3.14159265358979

// Define Namespace
namespace nn = torch::nn;


// ----------------------------------------------------------------------
// struct{EncoderImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
EncoderImpl::EncoderImpl(po::variables_map &vm){
    
    size_t feature = vm["nf"].as<size_t>();
    size_t num_downs = (size_t)(std::log2(vm["size"].as<size_t>()));

    DownSampling(this->model, vm["nc"].as<size_t>(), feature, /*BN=*/false, /*ReLU=*/true);       // {C,256,256} ===> {F,128,128}
    DownSampling(this->model, feature, feature*2, /*BN=*/true, /*ReLU=*/true);                    // {F,128,128} ===> {2F,64,64}
    DownSampling(this->model, feature*2, feature*4, /*BN=*/true, /*ReLU=*/true);                  // {2F,64,64}  ===> {4F,32,32}
    DownSampling(this->model, feature*4, feature*8, /*BN=*/true, /*ReLU=*/true);                  // {4F,32,32}  ===> {8F,16,16}
    for (size_t i = 0; i < num_downs - 5; i++){
        DownSampling(this->model, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);              // {8F,16,16}  ===> {8F,2,2}
    }
    DownSampling(this->model, feature*8, vm["nz_c"].as<size_t>(), /*BN=*/false, /*ReLU=*/false);  // {8F,2,2}    ===> {ZC,1,1}
    register_module("Encoder", this->model);

}


// ----------------------------------------------------------------------
// struct{EncoderImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor EncoderImpl::forward(torch::Tensor x){
    torch::Tensor out = this->model->forward(x);    // {C,256,256} ===> {ZC,1,1}
    return out;
}


// ----------------------------------------------------------------------
// struct{DecoderImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
DecoderImpl::DecoderImpl(po::variables_map &vm){
    
    size_t feature = vm["nf"].as<size_t>();
    size_t num_downs = (size_t)(std::log2(vm["size"].as<size_t>()));

    UpSampling(this->model, vm["nz_c"].as<size_t>(), feature*8, /*BN=*/true, /*ReLU=*/true);    // {ZC,1,1}     ===> {8F,2,2}
    for (size_t i = 0; i < num_downs - 5; i++){
        UpSampling(this->model, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);              // {8F,2,2}    ===> {8F,16,16}
    }
    UpSampling(this->model, feature*8, feature*4, /*BN=*/true, /*ReLU=*/true);                  // {8F,16,16}  ===> {4F,32,32}
    UpSampling(this->model, feature*4, feature*2, /*BN=*/true, /*ReLU=*/true);                  // {4F,32,32}  ===> {2F,64,64}
    UpSampling(this->model, feature*2, feature, /*BN=*/true, /*ReLU=*/true);                    // {2F,64,64}  ===> {F,128,128}
    UpSampling(this->model, feature, vm["nc"].as<size_t>(), /*BN=*/false, /*ReLU=*/false);      // {F,128,128} ===> {C,256,256}
    this->model->push_back(nn::Tanh());                                                         // [-inf,+inf] ===> [-1,1]
    register_module("Decoder", this->model);

}


// ----------------------------------------------------------------------
// struct{DecoderImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor DecoderImpl::forward(torch::Tensor z_c){
    torch::Tensor out = this->model->forward(z_c);  // {ZC,1,1} ===> {C,256,256}
    return out;
}


// ----------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
EstimationNetworkImpl::EstimationNetworkImpl(po::variables_map &vm, const float eps_){

    this->eps = eps_;
    size_t nz = vm["nz_c"].as<size_t>() + vm["nz_r"].as<size_t>();
    size_t nk = vm["nk"].as<size_t>();
    size_t nr = vm["nr"].as<size_t>();
    size_t n_blocks = vm["n_blocks"].as<size_t>();
    bool use_dropout = !vm["no_dropout"].as<bool>();

    this->model->push_back(nn::Linear(nz, nr));
    for (size_t i = 0; i < n_blocks; i++){
        this->model->push_back(LinearResBlock(nr, use_dropout));
    }
    this->model->push_back(nn::Linear(nr, nk));
    this->model->push_back(nn::Softmax(nn::SoftmaxOptions(/*dim=*/1)));
    register_module("EstimationNetwork", this->model);

}


// ----------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor EstimationNetworkImpl::forward(torch::Tensor z){
    torch::Tensor out = this->model->forward(z) + this->eps;  // {Z} ===> {K}
    return out;
}


// ----------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{estimation}
// ----------------------------------------------------------------------
void EstimationNetworkImpl::estimation(torch::Tensor z, torch::Tensor gamma_ap){

    size_t nz = z.size(1);      // Z = the number of latent variables
    size_t nk = gamma_ap.size(1);  // K = the number of attribution probability

    torch::Tensor gamma_sum = torch::sum(gamma_ap, /*dim=*/0);  // gamma_ap{N,K} ===> gamma_sum{K}
    torch::Tensor mu = torch::sum(gamma_ap.unsqueeze(2) * z.unsqueeze(1), /*dim=*/0) / gamma_sum.unsqueeze(1);  // gamma_ap{N,K,1}, z{N,1,Z}, gamma_sum{K,1} ===> mu{K,Z}
    torch::Tensor z_dev = z.unsqueeze(1) - mu.unsqueeze(0);  // z{N,1,Z}, mu{1,K,Z} ===> z_dev{N,K,Z}
    torch::Tensor z_dev_mat = z_dev.unsqueeze(3) * z_dev.unsqueeze(2);  // z_dev{N,K,Z,1}, z_dev{N,K,1,Z} ===> z_dev_mat{N,K,Z,Z}
    torch::Tensor sigma = torch::sum(gamma_ap.unsqueeze(-1).unsqueeze(-1) * z_dev_mat, /*dim=*/0) / gamma_sum.unsqueeze(-1).unsqueeze(-1);  // gamma_ap{N,K,1,1}, z_dev_mat{N,K,Z,Z}, gamma_sum{K,1,1} ===> sigma{K,Z,Z}
    torch::Tensor phi = gamma_sum / gamma_ap.size(0);  // gamma_sum{K} ===> phi{K}

    torch::Tensor sigma_k = sigma[0] + (torch::eye(nz) * this->eps).detach().to(z.device());  // sigma[i]{Z,Z} ===> sigma_k{Z,Z}
    torch::Tensor precision = torch::sum(1.0 / sigma_k.diag());  // sigma_k.diag{Z} ===> precision{}
    torch::Tensor sigma_inv = torch::inverse(sigma_k).unsqueeze(0);  // sigma_k{Z,Z} ===> sigma_inv{1,Z,Z}
    torch::Tensor det_sigma = torch::det(2.0 * PI * sigma_k).unsqueeze(0) + this->eps;  // sigma_k{Z,Z} ===> det_sigma{1}
    for (size_t i = 1; i < nk; i++){
        sigma_k = sigma[i] + (torch::eye(nz) * this->eps).detach().to(z.device());  // sigma[i]{Z,Z} ===> sigma_k{Z,Z}
        precision = precision + torch::sum(1.0 / sigma_k.diag());  // sigma_k.diag{Z} ===> precision{}
        sigma_inv = torch::cat({sigma_inv, torch::inverse(sigma_k).unsqueeze(0)}, /*dim=*/0);  // sigma_inv{i,Z,Z} + {1,Z,Z} ===> sigma_inv{i+1,Z,Z}
        det_sigma = torch::cat({det_sigma, torch::det(2.0 * PI * sigma_k).unsqueeze(0) + this->eps}, /*dim=*/0);  // det_sigma{i} + {1} ===> det_sigma{i+1}
    }

    torch::Tensor exp_term_left = torch::sum(z_dev.unsqueeze(3) * sigma_inv.unsqueeze(0), /*dim=*/2);  // z_dev{N,K,Z,1}, sigma_inv{1,K,Z,Z} ===> exp_term_left{N,K,Z}
    torch::Tensor exp_term_base = -0.5 * torch::sum(exp_term_left * z_dev, /*dim=*/2);  // exp_term_left{N,K,Z}, z_dev{N,K,Z} ===> exp_term_base{N,K}
    torch::Tensor max_value = std::get<0>(torch::max(exp_term_base.clamp(/*min=*/0.0), /*dim=*/1, /*keepdim=*/true));  // exp_term_base{N,K} ===> max_value{N,1}  (Note: To prevent overflow in "logsumexp" calculation.)
    torch::Tensor exp_term = torch::exp(exp_term_base - max_value);  // exp_term_base{N,K}, max_value{N,1} ===> exp_term{N,K}
    torch::Tensor energy = - max_value.squeeze(1) - torch::log(torch::sum(phi.unsqueeze(0) * exp_term / torch::sqrt(det_sigma).unsqueeze(0), /*dim=*/1) + this->eps);  // max_value{N}, phi{1,K}, exp_term{N,K}, det_sigma{1,1} ===> energy{N}

    this->energy_keep = torch::mean(energy);  // energy{N} ===> energy_keep{}
    this->precision_keep = precision / (float)nz / (float)nk;  // precision{} ===> precision_keep{}

    return;

}


// ----------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{estimationNVI}
// ----------------------------------------------------------------------
void EstimationNetworkImpl::estimationNVI(torch::Tensor z, torch::Tensor gamma_ap){

    size_t mini_batch_size = z.size(0);  // N = mini batch size
    size_t nz = z.size(1);               // Z = the number of latent variables
    size_t nk = gamma_ap.size(1);           // K = the number of attribution probability

    torch::Tensor gamma_sum = torch::sum(gamma_ap, /*dim=*/0);  // gamma_ap{N,K} ===> gamma_sum{K}
    torch::Tensor mu = torch::sum(gamma_ap.unsqueeze(2) * z.unsqueeze(1), /*dim=*/0) / gamma_sum.unsqueeze(1);  // gamma_ap{N,K,1}, z{N,1,Z}, gamma_sum{K,1} ===> mu{K,Z}
    torch::Tensor z_dev = z.unsqueeze(1) - mu.unsqueeze(0);  // z{N,1,Z}, mu{1,K,Z} ===> z_dev{N,K,Z}
    torch::Tensor z_dev_mat = z_dev.unsqueeze(3) * z_dev.unsqueeze(2);  // z_dev{N,K,Z,1}, z_dev{N,K,1,Z} ===> z_dev_mat{N,K,Z,Z}
    torch::Tensor sigma = torch::sum(gamma_ap.unsqueeze(-1).unsqueeze(-1) * z_dev_mat, /*dim=*/0) / gamma_sum.unsqueeze(-1).unsqueeze(-1);  // gamma_ap{N,K,1,1}, z_dev_mat{N,K,Z,Z}, gamma_sum{K,1,1} ===> sigma{K,Z,Z}
    torch::Tensor phi = gamma_sum / gamma_ap.size(0);  // gamma_sum{K} ===> phi{K}

    torch::Tensor sigma_k = sigma[0] + (torch::eye(nz) * this->eps).detach().to(z.device());  // sigma[i]{Z,Z} ===> sigma_k{Z,Z}
    torch::Tensor precision = torch::sum(1.0 / sigma_k.diag());  // sigma_k.diag{Z} ===> precision{}
    for (size_t i = 1; i < nk; i++){
        sigma_k = sigma[i] + (torch::eye(nz) * this->eps).detach().to(z.device());  // sigma[i]{Z,Z} ===> sigma_k{Z,Z}
        precision = precision + torch::sum(1.0 / sigma_k.diag());  // sigma_k.diag{Z} ===> precision{}
    }

    static auto criterion = nn::KLDivLoss(nn::KLDivLossOptions().reduction(torch::kSum));
    torch::Tensor NVI = criterion(gamma_ap.log(), phi.expand({(long int)mini_batch_size, (long int)nk}));  // gamma_ap{N,K}, phi{N,K} ===> NVI{}

    this->NVI_keep = NVI / (float)nk;  // NVI{} ===> NVI_keep{}
    this->precision_keep = precision / (float)nz / (float)nk;  // precision{} ===> precision_keep{}

    return;

}


// ----------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{resetGMP}
// ----------------------------------------------------------------------
void EstimationNetworkImpl::resetGMP(torch::Device device){

    this->N = 0;
    this->gamma_sum_keep = torch::full({}, /*value*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    this->mu_sum_keep = torch::full({}, /*value*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    this->sigma_sum_keep = torch::full({}, /*value*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

    return;

}


// ----------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{estimationGMP}
// ----------------------------------------------------------------------
void EstimationNetworkImpl::estimationGMP(torch::Tensor z, torch::Tensor gamma_ap){

    size_t mini_batch_size = z.size(0);  // N = mini batch size

    torch::Tensor gamma_sum = torch::sum(gamma_ap, /*dim=*/0);  // gamma_ap{N,K} ===> gamma_sum{K}
    torch::Tensor mu = torch::sum(gamma_ap.unsqueeze(2) * z.unsqueeze(1), /*dim=*/0) / gamma_sum.unsqueeze(1);  // gamma_ap{N,K,1}, z{N,1,Z}, gamma_sum{K,1} ===> mu{K,Z}
    torch::Tensor z_dev = z.unsqueeze(1) - mu.unsqueeze(0);  // z{N,1,Z}, mu{1,K,Z} ===> z_dev{N,K,Z}
    torch::Tensor z_dev_mat = z_dev.unsqueeze(3) * z_dev.unsqueeze(2);  // z_dev{N,K,Z,1}, z_dev{N,K,1,Z} ===> z_dev_mat{N,K,Z,Z}
    torch::Tensor sigma = torch::sum(gamma_ap.unsqueeze(-1).unsqueeze(-1) * z_dev_mat, /*dim=*/0) / gamma_sum.unsqueeze(-1).unsqueeze(-1);  // gamma_ap{N,K,1,1}, z_dev_mat{N,K,Z,Z}, gamma_sum{K,1,1} ===> sigma{K,Z,Z}

    this->N += (size_t)mini_batch_size;  // mini_batch_size{} ===> N{}
    this->gamma_sum_keep = (this->gamma_sum_keep + gamma_sum).detach();  // gamma_sum{K} ===> gamma_sum_keep{K}
    this->mu_sum_keep = (this->mu_sum_keep + mu * gamma_sum.unsqueeze(-1)).detach();  // mu{K,Z}, gamma_sum{K,1} ===> mu_sum_keep{K,Z}
    this->sigma_sum_keep = (this->sigma_sum_keep + sigma * gamma_sum.unsqueeze(-1).unsqueeze(-1)).detach();  // sigma{K,Z,Z}, gamma_sum{K,1,1} ===> sigma_sum_keep{K,Z,Z}

    return;

}


// -----------------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{estimated_mu}
// -----------------------------------------------------------------------------
torch::Tensor EstimationNetworkImpl::estimated_mu(){
    torch::Tensor mu = this->mu_sum_keep / this->gamma_sum_keep.unsqueeze(-1);  // mu_sum_keep{K,Z}, gamma_sum_keep{K,1} ===> mu{K,Z}
    return mu.detach();
}


// -----------------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{estimated_sigma}
// -----------------------------------------------------------------------------
torch::Tensor EstimationNetworkImpl::estimated_sigma(){
    torch::Tensor sigma = this->sigma_sum_keep / this->gamma_sum_keep.unsqueeze(-1).unsqueeze(-1);  // sigma_sum_keep{K,Z,Z}, gamma_sum_keep{K,1,1} ===> sigma{K,Z,Z}
    return sigma.detach();
}


// -----------------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{estimated_phi}
// -----------------------------------------------------------------------------
torch::Tensor EstimationNetworkImpl::estimated_phi(){
    torch::Tensor phi = this->gamma_sum_keep / (float)this->N;  // gamma_sum_keep{K}, N{} ===> phi{K}
    return phi.detach();
}


// -----------------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{energy_just_before}
// -----------------------------------------------------------------------------
torch::Tensor EstimationNetworkImpl::energy_just_before(){
    return this->energy_keep;
}


// -----------------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{NVI_just_before}
// -----------------------------------------------------------------------------
torch::Tensor EstimationNetworkImpl::NVI_just_before(){
    return this->NVI_keep;
}


// -----------------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{precision_just_before}
// -----------------------------------------------------------------------------
torch::Tensor EstimationNetworkImpl::precision_just_before(){
    return this->precision_keep;
}


// ----------------------------------------------------------------------
// struct{EstimationNetworkImpl}(nn::Module) -> function{anomaly_score}
// ----------------------------------------------------------------------
torch::Tensor EstimationNetworkImpl::anomaly_score(torch::Tensor z, torch::Tensor mu, torch::Tensor sigma, torch::Tensor phi){

    size_t nz = mu.size(1);  // Z = the number of latent variables
    size_t nk = mu.size(0);  // K = the number of attribution probability

    torch::Tensor z_dev = z.unsqueeze(1) - mu.unsqueeze(0);  // z{N,1,Z}, mu{1,K,Z} ===> z_dev{N,K,Z}

    torch::Tensor sigma_k = sigma[0] + (torch::eye(nz) * this->eps).detach().to(z.device());  // sigma[i]{Z,Z} ===> sigma_k{Z,Z}
    torch::Tensor sigma_inv = torch::inverse(sigma_k).unsqueeze(0);  // sigma_k{Z,Z} ===> sigma_inv{1,Z,Z}
    torch::Tensor det_sigma = torch::det(2.0 * PI * sigma_k).unsqueeze(0) + this->eps;  // sigma_k{Z,Z} ===> det_sigma{1}
    for (size_t i = 1; i < nk; i++){
        sigma_k = sigma[i] + (torch::eye(nz) * this->eps).detach().to(z.device());  // sigma[i]{Z,Z} ===> sigma_k{Z,Z}
        sigma_inv = torch::cat({sigma_inv, torch::inverse(sigma_k).unsqueeze(0)}, /*dim=*/0);  // sigma_inv{i,Z,Z} + {1,Z,Z} ===> sigma_inv{i+1,Z,Z}
        det_sigma = torch::cat({det_sigma, torch::det(2.0 * PI * sigma_k).unsqueeze(0) + this->eps}, /*dim=*/0);  // det_sigma{i} + {1} ===> det_sigma{i+1}
    }

    torch::Tensor exp_term_left = torch::sum(z_dev.unsqueeze(3) * sigma_inv.unsqueeze(0), /*dim=*/2);  // z_dev{N,K,Z,1}, sigma_inv{1,K,Z,Z} ===> exp_term_left{N,K,Z}
    torch::Tensor exp_term_base = -0.5 * torch::sum(exp_term_left * z_dev, /*dim=*/2);  // exp_term_left{N,K,Z}, z_dev{N,K,Z} ===> exp_term_base{N,K}
    torch::Tensor max_value = std::get<0>(torch::max(exp_term_base.clamp(/*min=*/0.0), /*dim=*/1, /*keepdim=*/true));  // exp_term_base{N,K} ===> max_value{N,1}  (Note: To prevent overflow in "logsumexp" calculation.)
    torch::Tensor exp_term = torch::exp(exp_term_base - max_value);  // exp_term_base{N,K}, max_value{N,1} ===> exp_term{N,K}
    torch::Tensor energy = - max_value.squeeze(1) - torch::log(torch::sum(phi.unsqueeze(0) * exp_term / torch::sqrt(det_sigma).unsqueeze(0), /*dim=*/1) + this->eps);  // max_value{N}, phi{1,K}, exp_term{N,K}, det_sigma{1,1} ===> energy{N}

    torch::Tensor out = torch::mean(energy);  // energy{N} ===> energy{}
    return out;

}


// ----------------------------------------------------------------------
// struct{LinearResBlockImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
LinearResBlockImpl::LinearResBlockImpl(const size_t nr, bool use_dropout){
    this->model->push_back(nn::Linear(nr, nr));
    this->model->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    if (use_dropout){
        this->model->push_back(nn::Dropout(0.5));
    }
    this->model->push_back(nn::Linear(nr, nr));
    register_module("LinearResBlock", this->model);
}


// ----------------------------------------------------------------------
// struct{LinearResBlockImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor LinearResBlockImpl::forward(torch::Tensor x){
    torch::Tensor res = this->model->forward(x);  // {R} ===> {R}
    torch::Tensor out = x + res;
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


// ----------------------------
// function{DownSampling}
// ----------------------------
void DownSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias){
    sq->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 4).stride(2).padding(1).bias(bias)));
    if (BN){
        sq->push_back(nn::BatchNorm2d(out_nc));
    }
    if (ReLU){
        sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    return;
}


// ----------------------------
// function{UpSampling}
// ----------------------------
void UpSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias){
    sq->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(in_nc, out_nc, 4).stride(2).padding(1).bias(bias)));
    if (BN){
        sq->push_back(nn::BatchNorm2d(out_nc));
    }
    if (ReLU){
        sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    return;
}


// ------------------------------------
// function{AbsoluteEuclideanDistance}
// ------------------------------------
torch::Tensor AbsoluteEuclideanDistance(torch::Tensor x1, torch::Tensor x2, const bool rec_detach){
    x1 = x1.contiguous().view({x1.size(0), x1.size(1) * x1.size(2) * x1.size(3)});  // {C,H,W} ===> {C*H*W}
    x2 = x2.contiguous().view({x2.size(0), x2.size(1) * x2.size(2) * x2.size(3)});  // {C,H,W} ===> {C*H*W}
    torch::Tensor out = torch::sqrt(torch::sum(torch::pow(x1 - x2, 2.0), /*dim=*/1, /*keepdim=*/true));  // ||x1 - x2||_2
    if (rec_detach){
        return out.detach();
    }
    return out;
}


// ------------------------------------
// function{RelativeEuclideanDistance}
// ------------------------------------
torch::Tensor RelativeEuclideanDistance(torch::Tensor x1, torch::Tensor x2, const bool rec_detach){
    x1 = x1.contiguous().view({x1.size(0), x1.size(1) * x1.size(2) * x1.size(3)});  // {C,H,W} ===> {C*H*W}
    x2 = x2.contiguous().view({x2.size(0), x2.size(1) * x2.size(2) * x2.size(3)});  // {C,H,W} ===> {C*H*W}
    torch::Tensor ed_12 = torch::sqrt(torch::sum(torch::pow(x1 - x2, 2.0), /*dim=*/1, /*keepdim=*/true));  // ||x1 - x2||_2
    torch::Tensor ed_1 = torch::sqrt(torch::sum(torch::pow(x1, 2.0), /*dim=*/1, /*keepdim=*/true));        // ||x1||_2
    torch::Tensor out = ed_12 / ed_1;  // ||x1 - x2||_2 / ||x1||_2
    if (rec_detach){
        return out.detach();
    }
    return out;
}


// ------------------------------------
// function{CosineSimilarity}
// ------------------------------------
torch::Tensor CosineSimilarity(torch::Tensor x1, torch::Tensor x2, const bool rec_detach){
    x1 = x1.contiguous().view({x1.size(0), x1.size(1) * x1.size(2) * x1.size(3)});  // {C,H,W} ===> {C*H*W}
    x2 = x2.contiguous().view({x2.size(0), x2.size(1) * x2.size(2) * x2.size(3)});  // {C,H,W} ===> {C*H*W}
    torch::Tensor ed_1 = torch::sqrt(torch::sum(torch::pow(x1, 2.0), /*dim=*/1, /*keepdim=*/true));  // ||x1||_2
    torch::Tensor ed_2 = torch::sqrt(torch::sum(torch::pow(x2, 2.0), /*dim=*/1, /*keepdim=*/true));  // ||x2||_2
    torch::Tensor ip_12 = torch::sum(x1 * x2, /*dim=*/1, /*keepdim=*/true);  // <x1,x2>
    torch::Tensor out = ip_12 / (ed_1 * ed_2);  // <x1,x2> / (||x1||_2 * ||x2||_2)
    if (rec_detach){
        return out.detach();
    }
    return out;
}


// ------------------------------------
// function{save_params}
// ------------------------------------
void save_params(const std::string path, torch::Tensor mu, torch::Tensor sigma, torch::Tensor phi){

    long int i, nk, nz;
    float *mu_array, *sigma_array, *phi_array;

    nk = mu.size(0);
    nz = mu.size(1);
    mu_array = mu.to(torch::kCPU).data_ptr<float>();
    sigma_array = sigma.to(torch::kCPU).data_ptr<float>();
    phi_array = phi.to(torch::kCPU).data_ptr<float>();

    std::ofstream ofs(path, std::ios::binary|std::ios::out);
    ofs.write((char*)&nk, sizeof(nk));
    ofs.write((char*)&nz, sizeof(nz));
    for (i = 0; i < nk*nz; i++){
        ofs.write((char*)&mu_array[i], sizeof(mu_array[i]));
    }
    for (i = 0; i < nk*nz*nz; i++){
        ofs.write((char*)&sigma_array[i], sizeof(sigma_array[i]));
    }
    for (i = 0; i < nk; i++){
        ofs.write((char*)&phi_array[i], sizeof(phi_array[i]));
    }
    ofs.close();

    return;
}


// ------------------------------------
// function{load_params}
// ------------------------------------
void load_params(const std::string path, torch::Tensor &mu, torch::Tensor &sigma, torch::Tensor &phi){

    long int i, nk, nz;
    float *mu_array, *sigma_array, *phi_array;

    std::ifstream ifs(path, std::ios::binary|std::ios::in);
    ifs.read((char*)&nk, sizeof(nk));
    ifs.read((char*)&nz, sizeof(nz));
    mu_array = new float[nk*nz];
    sigma_array = new float[nk*nz*nz];
    phi_array = new float[nk];
    for (i = 0; i < nk*nz; i++){
        ifs.read((char*)&mu_array[i], sizeof(mu_array[i]));
    }
    for (i = 0; i < nk*nz*nz; i++){
        ifs.read((char*)&sigma_array[i], sizeof(sigma_array[i]));
    }
    for (i = 0; i < nk; i++){
        ifs.read((char*)&phi_array[i], sizeof(phi_array[i]));
    }
    ifs.close();

    mu = torch::from_blob(mu_array, {nk, nz}, torch::kFloat).clone();  // mu{K,Z}
    sigma = torch::from_blob(sigma_array, {nk, nz, nz}, torch::kFloat).clone();  // sigma{K,Z,Z}
    phi = torch::from_blob(phi_array, {nk}, torch::kFloat).clone();  // phi{K}

    delete[] mu_array;
    delete[] sigma_array;
    delete[] phi_array;

    return;
}
