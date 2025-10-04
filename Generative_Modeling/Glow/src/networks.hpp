#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <vector>
#include <tuple>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);


// -------------------------------------------------
// struct{ActNorm2dImpl}(nn::Module)
// -------------------------------------------------
struct ActNorm2dImpl : nn::Module{
private:
    bool initialized;
    torch::Tensor scale, bias;
public:
    ActNorm2dImpl(){}
    ActNorm2dImpl(long int dim);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    torch::Tensor inverse(torch::Tensor y);
};
TORCH_MODULE(ActNorm2d);


// -------------------------------------------------
// struct{InvConv1x1Impl}(nn::Module)
// -------------------------------------------------
struct InvConv1x1Impl : nn::Module{
private:
    long int C;
    torch::Tensor w_p, u_mask, l_mask, s_sign, l_eye, w_l, w_s, w_u;
    torch::Tensor build_weight();
public:
    InvConv1x1Impl(){}
    InvConv1x1Impl(long int dim);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    torch::Tensor inverse(torch::Tensor y);
};
TORCH_MODULE(InvConv1x1);


// -------------------------------------------------
// struct{ZeroConv2dImpl}(nn::Module)
// -------------------------------------------------
struct ZeroConv2dImpl : nn::Module{
private:
    nn::Conv2d conv = nullptr;
    torch::Tensor scale;
public:
    ZeroConv2dImpl(){}
    ZeroConv2dImpl(long int in_nc, long int out_nc);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ZeroConv2d);


// -------------------------------------------------
// struct{CouplingLayerImpl}(nn::Module)
// -------------------------------------------------
struct CouplingLayerImpl : nn::Module{
private:
    long int dim, h_dim;
    nn::Sequential net;
public:
    CouplingLayerImpl(){}
    CouplingLayerImpl(long int dim, long int h_dim);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor inverse(torch::Tensor y);
};
TORCH_MODULE(CouplingLayer);


// -------------------------------------------------
// struct{FlowImpl}(nn::Module)
// -------------------------------------------------
struct FlowImpl : nn::Module{
private:
    ActNorm2d actnorm;
    InvConv1x1 invconv;
    CouplingLayer coupling;
public:
    FlowImpl(){}
    FlowImpl(long int in_nc, long int h_dim);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor z);
    torch::Tensor inverse(torch::Tensor z);
};
TORCH_MODULE(Flow);


// -------------------------------------------------
// struct{BlockImpl}(nn::Module)
// -------------------------------------------------
struct BlockImpl : nn::Module{
private:
    bool split;
    nn::ModuleList flows;
    ZeroConv2d prior;
public:
    BlockImpl(){}
    BlockImpl(long int dim, long int h_dim, long int n_flow, bool split_);
    torch::Tensor squeeze2x(torch::Tensor x);
    torch::Tensor unsqueeze2x(torch::Tensor y);
    torch::Tensor gaussian_log_p(torch::Tensor x, torch::Tensor mean, torch::Tensor log_sd);
    torch::Tensor gaussian_sample(torch::Tensor eps, torch::Tensor mean, torch::Tensor log_sd);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    torch::Tensor inverse(torch::Tensor y, torch::Tensor eps);
};
TORCH_MODULE(Block);


// -------------------------------------------------
// struct{NormalizingFlowImpl}(nn::Module)
// -------------------------------------------------
struct NormalizingFlowImpl : nn::Module{
private:
    nn::ModuleList blocks;
public:
    NormalizingFlowImpl(){}
    NormalizingFlowImpl(po::variables_map &vm); 
    std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> forward(torch::Tensor z);
    torch::Tensor inverse(std::vector<torch::Tensor> z_list);
};
TORCH_MODULE(NormalizingFlow);


#endif