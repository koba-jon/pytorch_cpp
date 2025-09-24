#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std:;tuple
#include <vector>                      // std::vector
#include <utility>                     // std::pair
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
// Synthesis Function
// -------------------
void synth(po::variables_map &vm, torch::Device &device, VQVAE2 &model, PixelSnail &pixelsnail_t, PixelSnail &pixelsnail_b){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    size_t count;
    std::string path, result_dir;
    std::stringstream ss;
    std::tuple<std::vector<long int>, std::vector<long int>> idx_shape;
    torch::Tensor x, y, output, outputs;

    // (1) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["synth_vqvae2_load_epoch"].as<std::string>() + "_vqvae2.pth"; torch::load(model, path, device);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["synth_pixelsnail_load_epoch"].as<std::string>() + "_pixelsnail_t.pth"; torch::load(pixelsnail_t, path, device);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["synth_pixelsnail_load_epoch"].as<std::string>() + "_pixelsnail_b.pth"; torch::load(pixelsnail_b, path, device);

    // (2) Image Generation
    torch::NoGradGuard no_grad;
    model->eval();
    pixelsnail_t->eval();
    pixelsnail_b->eval();
    count = vm["synth_count"].as<size_t>();
    idx_shape = model->get_idx_shape({1, (long int)vm["nc"].as<size_t>(), (long int)vm["size"].as<size_t>(), (long int)vm["size"].as<size_t>()}, device);
    x = model->sampling(idx_shape, pixelsnail_t, pixelsnail_b, device);
    y = model->sampling(idx_shape, pixelsnail_t, pixelsnail_b, device);
    outputs = model->synthesis(x, y, 1.0);
    for (size_t i = 1; i < count; i++){
        output = model->synthesis(x, y, float(count - i - 1) / float(count - 1));
        outputs = torch::cat({outputs, output}, /*dim=*/0);
    }
    result_dir = vm["synth_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ss.str(""); ss.clear(std::stringstream::goodbit);
    ss << result_dir << "/Generated_Image."  << extension;
    visualizer::save_image(outputs.detach(), ss.str(), /*range=*/output_range, /*cols=*/count);

    // End Processing
    return;

}
