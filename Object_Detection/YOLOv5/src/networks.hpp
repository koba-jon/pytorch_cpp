#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <vector>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);
torch::Tensor UpSampling(torch::Tensor x, const std::vector<long int> shape);


// -------------------------------------------------
// struct{ConvBlockImpl}(nn::Module)
// -------------------------------------------------
struct ConvBlockImpl : nn::Module{
private:
    nn::Sequential sq;
public:
    ConvBlockImpl(){}
    ConvBlockImpl(const size_t in_nc, const size_t out_nc, const size_t kernel, const size_t stride, const size_t padding, const bool BN=true, const bool SiLU=true, const bool bias=false);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ConvBlock);


// -------------------------------------------------
// struct{BottleneckImpl}(nn::Module)
// -------------------------------------------------
struct BottleneckImpl : nn::Module{
private:
    ConvBlock cv1, cv2;
    bool residual;
public:
    BottleneckImpl(){}
    BottleneckImpl(const size_t in_nc, const size_t out_nc, const bool shortcut, const double expansion);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Bottleneck);


// -------------------------------------------------
// struct{C3Impl}(nn::Module)
// -------------------------------------------------
struct C3Impl : nn::Module{
private:
    ConvBlock cv1, cv2, cv3;
    nn::Sequential m;
public:
    C3Impl(){}
    C3Impl(const size_t in_nc, const size_t out_nc, const size_t n, const bool shortcut=true);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C3);


// -------------------------------------------------
// struct{SPPFImpl}(nn::Module)
// -------------------------------------------------
struct SPPFImpl : nn::Module{
private:
    ConvBlock cv1, cv2;
    nn::MaxPool2d maxpool = nullptr;
public:
    SPPFImpl(){}
    SPPFImpl(const size_t in_nc, const size_t out_nc, const size_t kernel);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SPPF);


// -------------------------------------------------
// struct{YOLOv5Impl}(nn::Module)
// -------------------------------------------------
struct YOLOv5Impl : nn::Module{
private:
    ConvBlock conv_0, conv_1, conv_3, conv_5, conv_7;
    ConvBlock head_conv_10, head_conv_14, head_conv_18, head_conv_21;
    C3 c3_2, c3_4, c3_6, c3_8, head_c3_13, head_c3_17, head_c3_20, head_c3_23;
    SPPF sppf_9;
    nn::Conv2d detect_small = nullptr;
    nn::Conv2d detect_medium = nullptr;
    nn::Conv2d detect_large = nullptr;
    size_t mul(const double base, const double scale);
public:
    YOLOv5Impl(){}
    YOLOv5Impl(po::variables_map &vm);
    std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(YOLOv5);


#endif