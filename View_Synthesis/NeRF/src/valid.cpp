#include <iostream>                    // std::flush
#include <fstream>                     // std::ofstream
#include <string>                      // std::string
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // NeRF
#include "dataloader.hpp"              // DataLoader::ImageFolderCameraPoseWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderCameraPoseWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, NeRF &model, const size_t epoch, visualizer::graph &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_loss, total_loss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor t, x_t, noise, loss, image, output;
    std::tuple<torch::Tensor, torch::Tensor> x_t_with_noise;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_loss = 0.0;
    while (valid_dataloader(mini_batch)){
        image = std::get<0>(mini_batch).to(device);
        t = torch::randint(1, vm["timesteps"].as<size_t>() + 1, {image.size(0)}).to(device);
        x_t_with_noise = model->add_noise(image, t);
        x_t = std::get<0>(x_t_with_noise);
        noise = std::get<1>(x_t_with_noise);
        output = model->forward(x_t, t);
        loss = criterion(output, noise);
        total_loss += loss.item<float>();
        iteration++;
    }

    // (2) Calculate Average Loss
    ave_loss = total_loss / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "loss:" << ave_loss << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_loss});

    // End Processing
    return;

}