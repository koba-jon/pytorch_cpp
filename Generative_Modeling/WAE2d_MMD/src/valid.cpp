#include <iostream>                    // std::flush
#include <fstream>                     // std::ofstream
#include <string>                      // std::string
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss, MMDLoss
#include "networks.hpp"                // WAE_Encoder, WAE_Decoder
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, MMDLoss &criterion_MMD, WAE_Encoder &enc, WAE_Decoder &dec, const size_t epoch, visualizer::graph &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    size_t mini_batch_size;
    float ave_rec_loss, total_rec_loss;
    float ave_mmd_loss, total_mmd_loss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, output, z_real, z_fake;
    torch::Tensor rec_loss, mmd_loss;

    // (1) Tensor Forward per Mini Batch
    enc->eval();
    dec->eval();
    iteration = 0;
    total_rec_loss = 0.0; total_mmd_loss = 0.0;
    while (valid_dataloader(mini_batch)){

        image = std::get<0>(mini_batch).to(device);
        mini_batch_size = image.size(0);

        // (1.1) Calculation of Reconstruction Loss
        z_real = torch::randn({(long int)mini_batch_size, (long int)vm["nz"].as<size_t>()}).to(device);
        z_fake = enc->forward(image);
        output = dec->forward(z_fake);
        rec_loss = criterion(output, image);

        // (1.2) Calculation of Maximum Mean Discrepancy (MMD) Loss
        mmd_loss = criterion_MMD(z_fake, z_real) * vm["Lambda"].as<float>();
        
        // (1.3) Update Loss
        total_rec_loss += rec_loss.item<float>();
        total_mmd_loss += mmd_loss.item<float>();
        
        iteration++;
    }

    // (2) Calculate Average Loss
    ave_rec_loss = total_rec_loss / (float)iteration;
    ave_mmd_loss = total_mmd_loss / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "rec:" << ave_rec_loss << ' ' << std::flush;
    ofs << "mmd:" << ave_mmd_loss << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_rec_loss, ave_mmd_loss});

    // End Processing
    return;

}