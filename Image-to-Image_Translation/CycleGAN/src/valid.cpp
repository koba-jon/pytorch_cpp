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
#include "networks.hpp"                // ResNet_Generator
#include "dataloader.hpp"              // DataLoader::ImageFolderPairWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderPairWithPaths &valid_dataloader, torch::Device &device, torch::nn::L1Loss &criterion, ResNet_Generator &genAB, ResNet_Generator &genBA, const size_t epoch, visualizer::graph &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    float total_AtoB_loss, total_BtoA_loss;
    float ave_AtoB_loss, ave_BtoA_loss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor realA, realB, fakeA, fakeB;
    torch::Tensor AtoB_loss, BtoA_loss;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    genAB->eval();
    genBA->eval();
    iteration = 0;
    total_AtoB_loss = 0.0; total_BtoA_loss = 0.0;
    while (valid_dataloader(mini_batch)){
        
        realA = std::get<0>(mini_batch).to(device);
        realB = std::get<1>(mini_batch).to(device);

        // (1.1) Generator Forward
        fakeB = genAB->forward(realA);
        fakeA = genBA->forward(realB);

        // (1.2) Generation Loss
        AtoB_loss = criterion(fakeB, realB);
        BtoA_loss = criterion(fakeA, realA);

        // (1.3) Update Loss
        total_AtoB_loss += AtoB_loss.item<float>();
        total_BtoA_loss += BtoA_loss.item<float>();

        iteration++;
    }

    // (2) Calculate Average Loss
    ave_AtoB_loss = total_AtoB_loss / (float)iteration;
    ave_BtoA_loss = total_BtoA_loss / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "AtoB:" << ave_AtoB_loss << ' ' << std::flush;
    ofs << "BtoA:" << ave_BtoA_loss << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_AtoB_loss, ave_BtoA_loss});

    // End Processing
    return;

}