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
#include "networks.hpp"                // VariationalAutoEncoder
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, VariationalAutoEncoder &model, const size_t epoch, visualizer::graph &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_rec_loss, total_rec_loss;
    float ave_kld_loss, total_kld_loss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor rec, kld, loss, image, output;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_rec_loss = 0.0; total_kld_loss = 0.0;
    while (valid_dataloader(mini_batch)){
        image = std::get<0>(mini_batch).to(device);
        output = model->forward(image);
        rec = criterion(output, image);
        kld = vm["Lambda"].as<float>() * model->kld_just_before();
        loss = rec + kld;
        total_rec_loss += rec.item<float>();
        total_kld_loss += kld.item<float>();
        iteration++;
    }

    // (2) Calculate Average Loss
    ave_rec_loss = total_rec_loss / (float)iteration;
    ave_kld_loss = total_kld_loss / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "rec:" << ave_rec_loss << ' ' << std::flush;
    ofs << "kld:" << ave_kld_loss << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_rec_loss + ave_kld_loss, ave_rec_loss, ave_kld_loss});

    // End Processing
    return;

}