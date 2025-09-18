#include <iostream>                    // std::cout
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std:;tuple
#include <vector>                      // std::vector
#include <utility>                     // std::pair
#include <ios>                         // std::right
#include <iomanip>                     // std::setw, std::setfill
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // VQVAE2, PixelSnail
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// -------------------
// Sampling Function
// -------------------
void sample(po::variables_map &vm, torch::Device &device, VQVAE2 &model, PixelSnail &pixelsnail_t, PixelSnail &pixelsnail_b){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    size_t total, digit;
    std::string path, result_dir, fname;
    std::stringstream ss;
    std::tuple<std::vector<long int>, std::vector<long int>> idx_shape;
    torch::Tensor z, output;

    // (1) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["sample_vqvae2_load_epoch"].as<std::string>() + "_vqvae2.pth";  torch::load(model, path, device);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["sample_pixelsnail_load_epoch"].as<std::string>() + "_pixelsnail_t.pth";  torch::load(pixelsnail_t, path, device);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["sample_pixelsnail_load_epoch"].as<std::string>() + "_pixelsnail_b.pth";  torch::load(pixelsnail_b, path, device);

    // (2) Image Generation
    torch::NoGradGuard no_grad;
    model->eval();
    pixelsnail_t->eval();
    pixelsnail_b->eval();
    result_dir = vm["sample_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    total = vm["sample_total"].as<size_t>();
    digit = std::to_string(total - 1).length();
    std::cout << "total sampling images : " << total << std::endl << std::endl;
    idx_shape = model->get_idx_shape({1, (long int)vm["nc"].as<size_t>(), (long int)vm["size"].as<size_t>(), (long int)vm["size"].as<size_t>()}, device);
    for (size_t i = 0; i < total; i++){

        output = model->sampling(idx_shape, pixelsnail_t, pixelsnail_b, device);

        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << std::setfill('0') << std::right << std::setw(digit) << i;
        fname = result_dir + '/' + ss.str() + '.' + std::string(extension);
        visualizer::save_image(output.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

        std::cout << '<' << fname << "> Generated!" << std::endl;

    }

    // End Processing
    return;

}