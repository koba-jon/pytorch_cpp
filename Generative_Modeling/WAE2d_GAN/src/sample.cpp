#include <iostream>                    // std::cout
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <utility>                     // std::pair
#include <ios>                         // std::right
#include <iomanip>                     // std::setw, std::setfill
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // WAE_Decoder
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// -------------------
// Sampling Function
// -------------------
void sample(po::variables_map &vm, torch::Device &device, WAE_Decoder &dec){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    size_t total, digit;
    std::string path, result_dir, fname;
    std::stringstream ss;
    torch::Tensor z, output;

    // (1) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["sample_load_epoch"].as<std::string>() + "_dec.pth";
    torch::load(dec, path, device);

    // (2) Image Generation
    dec->eval();
    result_dir = vm["sample_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    total = vm["sample_total"].as<size_t>();
    digit = std::to_string(total - 1).length();
    std::cout << "total sampling images : " << total << std::endl << std::endl;
    for (size_t i = 0; i < total; i++){

        z = torch::randn({1, (long int)vm["nz"].as<size_t>()}).to(device);
        output = dec->forward(z);

        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << std::setfill('0') << std::right << std::setw(digit) << i;
        fname = result_dir + '/' + ss.str() + '.' + std::string(extension);
        visualizer::save_image(output.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

        std::cout << '<' << fname << "> Generated!" << std::endl;

    }

    // End Processing
    return;

}