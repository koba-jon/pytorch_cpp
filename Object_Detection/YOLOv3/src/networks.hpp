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
// struct{YOLOv3Impl}(nn::Module)
// -------------------------------------------------
struct YOLOv3Impl : nn::Module{
private:
    nn::Sequential stage1, stage2, stage3, stage4, stage5, stage6;
    nn::Sequential stageA, stageB, stageC;
public:
    YOLOv3Impl(){}
    YOLOv3Impl(po::variables_map &vm);
    std::vector<torch::Tensor> forward(torch::Tensor x);
};

// -------------------------------------------------
// struct{ResBlockImpl}(nn::Module)
// -------------------------------------------------
struct ResBlockImpl : nn::Module{
private:
    nn::Sequential sq;
public:
    ResBlockImpl(){}
    ResBlockImpl(const size_t outside_nc, const size_t inside_nc);
    torch::Tensor forward(torch::Tensor x);
};

// -------------------------------------------------
// struct{ConvBlockImpl}(nn::Module)
// -------------------------------------------------
struct ConvBlockImpl : nn::Module{
private:
    nn::Sequential sq;
public:
    ConvBlockImpl(){}
    ConvBlockImpl(const size_t in_nc, const size_t out_nc, const size_t ksize, const size_t stride, const size_t pad, const bool BN, const bool LReLU, const bool bias=false);
    torch::Tensor forward(torch::Tensor x);
};


TORCH_MODULE(YOLOv3);
TORCH_MODULE(ResBlock);
TORCH_MODULE(ConvBlock);


#endif