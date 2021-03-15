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
void Convolution(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const size_t ksize, const size_t stride, const size_t pad, const bool BN, const bool LReLU, const bool bias=false);


// -------------------------------------------------
// struct{YOLOv2Impl}(nn::Module)
// -------------------------------------------------
struct YOLOv2Impl : nn::Module{
private:
    nn::Sequential stage1, stage2a, stage2b, stage3;
public:
    YOLOv2Impl(){}
    YOLOv2Impl(po::variables_map &vm);
    torch::Tensor forward(torch::Tensor x);
};

// -------------------------------------------------
// struct{FloorAvgPool2dImpl}(nn::Module)
// -------------------------------------------------
struct FloorAvgPool2dImpl : nn::Module{
private:
    std::vector<long int> multiple;
public:
    FloorAvgPool2dImpl(){}
    FloorAvgPool2dImpl(std::vector<long int> multiple_);
    torch::Tensor forward(torch::Tensor x);
};

// -------------------------------------------------
// struct{ReorganizeImpl}(nn::Module)
// -------------------------------------------------
struct ReorganizeImpl : nn::Module{
private:
    long int stride;
public:
    ReorganizeImpl(){}
    ReorganizeImpl(long int stride_);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(YOLOv2);
TORCH_MODULE(FloorAvgPool2d);
TORCH_MODULE(Reorganize);


#endif