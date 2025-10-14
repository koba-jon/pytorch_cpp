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
// struct{C2fImpl}(nn::Module)
// -------------------------------------------------
struct C2fImpl : nn::Module{
private:
    size_t hidden_nc;
    ConvBlock cv1, cv2;
    std::vector<Bottleneck> m;
public:
    C2fImpl(){}
    C2fImpl(const size_t in_nc, const size_t out_nc, const size_t n, const bool shortcut=true);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C2f);


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
// struct{YOLOv8Impl}(nn::Module)
// -------------------------------------------------
struct YOLOv8Impl : nn::Module{
private:
    ConvBlock conv_0, conv_1, conv_3, conv_5, conv_7;
    ConvBlock head_conv_16, head_conv_19;
    C2f c2f_2, c2f_4, c2f_6, c2f_8, head_c2f_12, head_c2f_15, head_c2f_18, head_c2f_21;
    SPPF sppf_9;
    nn::Conv2d detect_small_coord{nullptr}, detect_small_obj{nullptr}, detect_small_class{nullptr};
    nn::Conv2d detect_medium_coord{nullptr}, detect_medium_obj{nullptr}, detect_medium_class{nullptr};
    nn::Conv2d detect_large_coord{nullptr}, detect_large_obj{nullptr}, detect_large_class{nullptr};
    size_t mul(const double base, const double scale);
public:
    YOLOv8Impl(){}
    YOLOv8Impl(po::variables_map &vm);
    std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(YOLOv8);


#endif