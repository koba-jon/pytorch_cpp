#ifndef NETWORKS_HPP
#define NETWORKS_HPP

// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);


// -------------------------------------------------
// struct{EfficientNetConfig}
// -------------------------------------------------
struct EfficientNetConfig{
    double width_mul;
    double depth_mul;
    size_t image_size;
    double dropout;
};

// -------------------------------------------------
// struct{BlockConfig}
// -------------------------------------------------
struct BlockConfig{
    size_t k, exp, c, r, s;
};

// -------------------------------------------------
// struct{StochasticDepthImpl}(nn::Module)
// -------------------------------------------------
struct StochasticDepthImpl : nn::Module{
private:
    float p;
public:
    StochasticDepthImpl(){}
    StochasticDepthImpl(const float p_);
    torch::Tensor forward(torch::Tensor x);
    void pretty_print(std::ostream& stream) const override;
};
TORCH_MODULE(StochasticDepth);

// -------------------------------------------------
// struct{Conv2dNormActivationImpl}(nn::Module)
// -------------------------------------------------
struct Conv2dNormActivationImpl : nn::Module{
private:
    nn::Sequential model;
public:
    Conv2dNormActivationImpl(){}
    Conv2dNormActivationImpl(const size_t in_nc, const size_t out_nc, const size_t kernel_size, const size_t stride, const size_t padding, const size_t groups, const bool SiLU);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Conv2dNormActivation);

// -------------------------------------------------
// struct{SqueezeExcitationImpl}(nn::Module)
// -------------------------------------------------
struct SqueezeExcitationImpl : nn::Module{
private:
    nn::AdaptiveAvgPool2d avgpool = nullptr;
    nn::Conv2d fc1 = nullptr;
    nn::Conv2d fc2 = nullptr;
    nn::SiLU act = nullptr;
    nn::Sigmoid scale_act = nullptr;
public:
    SqueezeExcitationImpl(){}
    SqueezeExcitationImpl(const size_t in_nc, const size_t mid);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SqueezeExcitation);

// -------------------------------------------------
// struct{MBConvImpl}(nn::Module)
// -------------------------------------------------
struct MBConvImpl : nn::Module{
private:
    bool residual;
    nn::Sequential block;
    StochasticDepth sd;
public:
    MBConvImpl(){}
    MBConvImpl(const size_t in_nc, const size_t out_nc, const size_t kernel_size, const size_t stride, const size_t exp, const float dropconnect);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MBConv);

// -------------------------------------------------
// struct{MC_EfficientNetImpl}(nn::Module)
// -------------------------------------------------
struct MC_EfficientNetImpl : nn::Module{
private:
    nn::Sequential features, classifier;
    nn::AdaptiveAvgPool2d avgpool = nullptr;
    size_t make_divisible(size_t v, size_t divisor=8);
    size_t round_filters(size_t c, double width_mul);
    size_t round_repeats(size_t r, double depth_mul);
public:
    EfficientNetConfig cfg;
    MC_EfficientNetImpl(){}
    MC_EfficientNetImpl(po::variables_map &vm);
    void init();
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MC_EfficientNet);


#endif
