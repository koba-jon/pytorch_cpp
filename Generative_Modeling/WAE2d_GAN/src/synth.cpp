#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <utility>                     // std::pair
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
// Synthesis Function
// -------------------
void synth(po::variables_map &vm, torch::Device &device, WAE_Decoder &dec){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    size_t max_counter;
    float value;
    std::string path, result_dir;
    std::stringstream ss;
    torch::Tensor z, output, outputs;

    // (1) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["synth_load_epoch"].as<std::string>() + "_dec.pth";
    torch::load(dec, path);

    // (2) Image Generation
    dec->eval();
    max_counter = (int)(vm["synth_sigma_max"].as<float>() / vm["synth_sigma_inter"].as<float>() * 2) + 1;
    z = torch::full({1, (long int)vm["nz"].as<size_t>()}, /*value=*/-vm["synth_sigma_max"].as<float>(), torch::TensorOptions().dtype(torch::kFloat)).to(device);
    outputs = dec->forward(z);
    for (size_t i = 1; i < max_counter; i++){
        value = -vm["synth_sigma_max"].as<float>() + (float)i * vm["synth_sigma_inter"].as<float>();
        z = torch::full({1, (long int)vm["nz"].as<size_t>()}, /*value=*/value, torch::TensorOptions().dtype(torch::kFloat)).to(device);
        output = dec->forward(z);
        outputs = torch::cat({outputs, output}, /*dim=*/0);
    }
    result_dir = vm["synth_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ss.str(""); ss.clear(std::stringstream::goodbit);
    ss << result_dir << "/Generated_Image."  << extension;
    visualizer::save_image(outputs.detach(), ss.str(), /*range=*/output_range, /*cols=*/max_counter);

    // End Processing
    return;

}
