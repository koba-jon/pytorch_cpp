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
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor image, pose, rays_o, rays_d, target_rgb, rgb_fine, rgb_coarse;
    torch::Tensor loss, loss_fine, loss_coarse;
    std::tuple<torch::Tensor, torch::Tensor> x_t_with_noise;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_loss = 0.0;
    while (valid_dataloader(mini_batch)){
        image = std::get<0>(mini_batch).to(device);
        pose = std::get<1>(mini_batch).to(device);
        std::tie(rays_o, rays_d) = model->build_rays(pose);
        target_rgb = image.permute({0, 2, 3, 1}).view({image.size(0), -1, 3}).contiguous();
        std::tie(rgb_fine, rgb_coarse) = model->forward(rays_o, rays_d);
        loss_fine = criterion(rgb_fine, target_rgb);
        loss_coarse = criterion(rgb_coarse, target_rgb);
        loss = loss_fine + loss_coarse;
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