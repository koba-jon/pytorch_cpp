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
#include "networks.hpp"                // DiT
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, DiT &model, const size_t epoch, visualizer::graph &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_loss, total_loss;
    float ave_loss_diff, total_loss_diff;
    float ave_loss_rec, total_loss_rec;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, t, pred_noise, target_noise, rec, latent_noisy, loss_diff, loss_rec, loss;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_loss = 0.0;
    total_loss_diff = 0.0;
    total_loss_rec = 0.0;
    while (valid_dataloader(mini_batch)){

        image = std::get<0>(mini_batch).to(device);
        t = torch::randint(1, vm["timesteps"].as<size_t>() + 1, {image.size(0)}).to(device);
        std::tie(pred_noise, target_noise, rec, latent_noisy) = model->forward(image, t);
        loss_diff = criterion(pred_noise, target_noise);
        loss_rec = vm["Lambda"].as<float>() * criterion(rec, image);
        loss = loss_diff + loss_rec;

        total_loss += loss.item<float>();
        total_loss_diff += loss_diff.item<float>();
        total_loss_rec += loss_rec.item<float>();

        iteration++;

    }

    // (2) Calculate Average Loss
    ave_loss = total_loss / (float)iteration;
    ave_loss_diff = total_loss_diff / (float)iteration;
    ave_loss_rec = total_loss_rec / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "loss:" << ave_loss << ' ' << std::flush;
    ofs << "diff:" << ave_loss_diff << ' ' << std::flush;
    ofs << "rec:" << ave_loss_rec << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_loss, ave_loss_diff, ave_loss_rec});

    // End Processing
    return;

}