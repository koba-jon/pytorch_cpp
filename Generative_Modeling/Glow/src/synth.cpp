#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <vector>                      // std::vector
#include <utility>                     // std::pair
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // NormalizingFlow
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
std::vector<torch::Tensor> rand_z_list_(long int size, long int nc, long int n_block, torch::Device device);
std::vector<torch::Tensor> blend(std::vector<torch::Tensor> z1_list, std::vector<torch::Tensor> z2_list, float alpha);


// -------------------
// Synthesis Function
// -------------------
void synth(po::variables_map &vm, torch::Device &device, NormalizingFlow &model){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    size_t count;
    std::string path, result_dir;
    std::stringstream ss;
    torch::Tensor output, outputs;
    std::vector<torch::Tensor> z_list, z1_list, z2_list;

    // (1) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["synth_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (2) Image Generation
    torch::NoGradGuard no_grad;
    model->eval();
    count = vm["synth_count"].as<size_t>();
    z1_list = rand_z_list_(vm["size"].as<size_t>(), vm["nc"].as<size_t>(), vm["n_block"].as<size_t>(), device);
    z2_list = rand_z_list_(vm["size"].as<size_t>(), vm["nc"].as<size_t>(), vm["n_block"].as<size_t>(), device);
    outputs = model->inverse(z1_list);
    for (size_t i = 1; i < count; i++){
        z_list = blend(z1_list, z2_list, float(count - i - 1) / float(count - 1));
        output = model->inverse(z_list);
        outputs = torch::cat({outputs, output}, /*dim=*/0);
    }
    result_dir = vm["synth_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ss.str(""); ss.clear(std::stringstream::goodbit);
    ss << result_dir << "/Generated_Image."  << extension;
    visualizer::save_image(outputs.detach(), ss.str(), /*range=*/output_range, /*cols=*/count);

    // End Processing
    return;

}


// --------------------------------
// Function to Make Random z list
// --------------------------------
std::vector<torch::Tensor> rand_z_list_(long int size, long int nc, long int n_block, torch::Device device){

    std::vector<torch::Tensor> z_list;

    z_list = std::vector<torch::Tensor>(n_block);
    for (long int i = 0; i < n_block - 1; i++){
        size = size / 2;
        nc = nc * 2;
        z_list[i] = torch::randn({1, nc, size, size}).to(device);
    }
    size = size / 2;
    z_list[n_block - 1] = torch::randn({1, nc * 4, size, size}).to(device);

    return z_list;

}


// --------------------------------
// Blending Function
// --------------------------------
std::vector<torch::Tensor> blend(std::vector<torch::Tensor> z1_list, std::vector<torch::Tensor> z2_list, float alpha){

    long int size;
    std::vector<torch::Tensor> z_list;

    size = z1_list.size();
    z_list = std::vector<torch::Tensor>(size);
    for (long int i = 0; i < size; i++){
        z_list[i] = z1_list[i] * alpha + z2_list[i] * (1.0 - alpha);
    }

    return z_list;

}