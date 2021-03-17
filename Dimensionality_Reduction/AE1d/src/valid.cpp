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
#include "networks.hpp"                // AutoEncoder1d
#include "dataloader.hpp"              // DataLoader::Data1dFolderWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::Data1dFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, AutoEncoder1d &model, const size_t epoch, visualizer::graph &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_loss, total_loss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, data, output;

    // (1) Tensor Forward per Mini Batch
    model->eval();
    iteration = 0;
    total_loss = 0.0;
    while (valid_dataloader(mini_batch)){
        data = std::get<0>(mini_batch).to(device);
        output = model->forward(data);
        loss = criterion(output, data);
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