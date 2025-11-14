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
#include "networks.hpp"                // GS3D
#include "visualizer.hpp"              // visualizer

// Define
#define PI 3.14159265358979

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;
using torch::indexing::Slice;


// -------------------
// Sampling Function
// -------------------
void sample(po::variables_map &vm, torch::Device &device, GS3D &model){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    size_t total, digit;
    std::string path, result_dir, fname;
    std::stringstream ss;
    torch::Tensor pose, rendered;
    torch::Tensor camera_origin, forward, up, right, world_up;
    float radius, theta, phi, x, y, z;

    // (1) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["sample_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (2) Image Generation
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = vm["sample_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    total = vm["sample_total"].as<size_t>();
    digit = std::to_string(total - 1).length();
    radius = vm["sample_radius"].as<float>();
    world_up = torch::tensor({0.0f, 0.0f, 1.0f}, torch::kFloat).to(device);
    std::cout << "total sampling images : " << total << std::endl << std::endl;
    for (size_t i = 0; i < total; i++){

        theta = vm["sample_theta"].as<float>() * PI / 180.0;
        phi = float(i) / (total) * 2.0 * PI;
        x = radius * std::sin(theta) * std::cos(phi);
        y = radius * std::sin(theta) * std::sin(phi);
        z = radius * std::cos(theta);
        camera_origin = torch::tensor({x, y, z}, torch::kFloat).to(device);
        
        forward = (-camera_origin).clone();
        forward = forward / forward.norm();
        right = torch::cross(world_up, forward, 0);
        right = right / right.norm();
        up = torch::cross(forward, right, 0);

        pose = torch::eye(4, torch::kFloat).unsqueeze(0).to(device);
        pose.index_put_({0, Slice(0, 3), Slice(0, 3)}, torch::stack({right, up, forward}, 1));
        pose.index_put_({0, Slice(0, 3), 3}, camera_origin);
        rendered = model->render_image(pose);
        rendered = rendered.flip({2}).flip({3});

        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << std::setfill('0') << std::right << std::setw(digit) << i;
        fname = result_dir + '/' + ss.str() + '.' + std::string(extension);
        visualizer::save_image(rendered.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

        std::cout << '<' << fname << "> Generated!" << std::endl;

    }

    // End Processing
    return;

}
