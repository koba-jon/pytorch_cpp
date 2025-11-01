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
    float base_scale, init_opacity;
    size_t density_control_interval, density_control_max_new;
    float density_control_prune_opacity, density_control_grow_opacity;
    float density_control_position_noise, density_control_scale_noise;
    torch::Tensor mu_world, log_scale, quat, colors, log_opacity, background_logit, mask;
public:
    GS3DImpl(){}
    GS3DImpl(po::variables_map &vm);
    torch::Tensor render_image(torch::Tensor pose);
    torch::Tensor quat_to_rotmat(torch::Tensor q);
    torch::Tensor forward(torch::Tensor pose);
    void init_gaussians();
    void adaptive_density_control();
    size_t active_gaussians() const;
    size_t total_gaussians() const;
    size_t density_interval() const;
};
TORCH_MODULE(GS3D);


#endif