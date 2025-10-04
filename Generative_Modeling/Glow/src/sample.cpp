#include <iostream>                    // std::cout
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <vector>                      // std::vector
#include <utility>                     // std::pair
#include <ios>                         // std::right
#include <iomanip>                     // std::setw, std::setfill
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
std::vector<torch::Tensor> rand_z_list(long int size, long int nc, long int n_block, torch::Device device);


// -------------------
// Sampling Function
// -------------------
void sample(po::variables_map &vm, torch::Device &device, NormalizingFlow &model){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    size_t total, digit;
    std::string path, result_dir, fname;
    std::stringstream ss;
    torch::Tensor z, output;
    std::vector<torch::Tensor> z_list;

    // (1) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["sample_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (2) Image Generation
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = vm["sample_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    total = vm["sample_total"].as<size_t>();
    digit = std::to_string(total - 1).length();
    std::cout << "total sampling images : " << total << std::endl << std::endl;
    for (size_t i = 0; i < total; i++){

        z_list = rand_z_list(vm["size"].as<size_t>(), vm["nc"].as<size_t>(), vm["n_block"].as<size_t>(), device);
        output = model->inverse(z_list);

        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << std::setfill('0') << std::right << std::setw(digit) << i;
        fname = result_dir + '/' + ss.str() + '.' + std::string(extension);
        visualizer::save_image(output.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

        std::cout << '<' << fname << "> Generated!" << std::endl;

    }

    // End Processing
    return;

}


// --------------------------------
// Function to Make Random z list
// --------------------------------
std::vector<torch::Tensor> rand_z_list(long int size, long int nc, long int n_block, torch::Device device){

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
