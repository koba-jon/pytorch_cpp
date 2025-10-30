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


// ------------------------------
// struct{GS3DImpl}(nn::Module)
// ------------------------------
struct GS3DImpl : nn::Module{
private:
    size_t size, num_gaussians;
    float focal_length, init_radius;
    torch::Tensor positions, sigma, rho, colors, log_opacity, background_logit;
public:
    GS3DImpl(){}
    GS3DImpl(po::variables_map &vm);
    torch::Tensor render_image(torch::Tensor pose);
    torch::Tensor forward(torch::Tensor pose);
    void init_gaussians();
};
TORCH_MODULE(GS3D);


#endif