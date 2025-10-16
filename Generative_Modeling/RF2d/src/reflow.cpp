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
#include "networks.hpp"                // RF
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// -------------------
// Sampling Function
// -------------------
void reflow(po::variables_map &vm, torch::Device &device, RF &model){

    // (0) Initialization and Declaration
    size_t total, digit;
    std::string path, reflowI_dir, reflowO_dir, fnameI, fnameO;
    std::stringstream ss;
    std::vector<long int> z_shape;
    torch::Tensor z, x_hat;

    // (1) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["reflow_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (2) Set Directories
    reflowI_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["reflow_in_dir"].as<std::string>();  fs::create_directories(reflowI_dir);
    reflowO_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["reflow_out_dir"].as<std::string>();  fs::create_directories(reflowO_dir);

    // (3) Image Generation
    torch::NoGradGuard no_grad;
    model->eval();
    total = vm["reflow_total"].as<size_t>();
    digit = std::to_string(total - 1).length();
    std::cout << "total reflow images : " << total << std::endl << std::endl;
    for (size_t i = 0; i < total; i++){

        z = torch::randn({1, (long int)vm["nc"].as<size_t>(), (long int)vm["size"].as<size_t>(), (long int)vm["size"].as<size_t>()}).to(device);
        x_hat = model->forward_z(z);

        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << std::setfill('0') << std::right << std::setw(digit) << i;
        fnameI = reflowI_dir + '/' + ss.str() + ".pth";
        fnameO = reflowO_dir + '/' + ss.str() + ".pth";
        torch::save(z.squeeze(0).detach().cpu(), fnameI);
        torch::save(x_hat.squeeze(0).detach().cpu(), fnameO);

        std::cout << '<' << fnameI << "> <" << fnameO << "> Generated!" << std::endl;

    }

    // End Processing
    return;

}