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
#include "networks.hpp"                // VQVAE
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid1(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, VQVAE &model, const size_t epoch, visualizer::graph &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_rec_loss, total_rec_loss;
    float ave_latent_loss, total_latent_loss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor rec, latent, loss, image;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_rec_loss = 0.0; total_latent_loss = 0.0;
    while (valid_dataloader(mini_batch)){
        image = std::get<0>(mini_batch).to(device);
        auto [output, z_e, z_q] = model->forward(image);
        rec = criterion(output, image);
        latent = torch::mean((z_e.detach() - z_q).pow(2.0)) + vm["Lambda"].as<float>() * torch::mean((z_e - z_q.detach()).pow(2.0));
        loss = rec + latent;
        total_rec_loss += rec.item<float>();
        total_latent_loss += latent.item<float>();
        iteration++;
    }

    // (2) Calculate Average Loss
    ave_rec_loss = total_rec_loss / (float)iteration;
    ave_latent_loss = total_latent_loss / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid1.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["train1_epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "rec:" << ave_rec_loss << ' ' << std::flush;
    ofs << "latent:" << ave_latent_loss << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_rec_loss + ave_latent_loss, ave_rec_loss, ave_latent_loss});

    // End Processing
    return;

}