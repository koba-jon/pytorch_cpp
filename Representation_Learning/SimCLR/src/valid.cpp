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
#include "networks.hpp"                // SimCLR
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, SimCLR &model, const size_t epoch, visualizer::graph &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_loss, total_loss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, aug1, aug2, z1, z2;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> output_set;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_loss = 0.0;
    while (valid_dataloader(mini_batch)){
        image = std::get<0>(mini_batch).to(device);
        aug1 = model->augmentation(image);
        aug2 = model->augmentation(image);
        z1 = model->forward(aug1);
        z2 = model->forward(aug2);
        loss = criterion(z1, z2);
        total_loss += loss.item<float>();
        iteration++;
    }

    // (2) Calculate Average Loss
    ave_loss = total_loss / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "rec:" << ave_loss << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_loss});

    // End Processing
    return;

}