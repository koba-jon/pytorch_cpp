#include <iostream>                    // std::flush
#include <fstream>                     // std::ofstream
#include <string>                      // std::string
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <cmath>                       // std::isinf
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // Encoder, Decoder, EstimationNetwork, RelativeEuclideanDistance, CosineSimilarity
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, Encoder &enc, Decoder &dec, EstimationNetwork &est, torch::Tensor mu, torch::Tensor sigma, torch::Tensor phi, const size_t epoch, visualizer::graph &writer, visualizer::graph &writer_rec){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_rec, total_rec;
    float ave_anomaly_score, total_anomaly_score, score;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, output;
    torch::Tensor z, z_c, z_r, z_r1, z_r2;
    torch::Tensor rec, anomaly_score;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    enc->eval();
    dec->eval();
    est->eval();
    iteration = 0;
    total_rec = 0.0; total_anomaly_score = 0.0;
    while (valid_dataloader(mini_batch)){

        image = std::get<0>(mini_batch).to(device);

        // (1.1) Encoder-Decoder Forward
        z_c = enc->forward(image);   // {C,H,W} ===> {ZC,1,1}
        output = dec->forward(z_c);  // {ZC,1,1} ===> {C,H,W}

        // (1.2) Setting Latent Space
        z_c = z_c.view({z_c.size(0), z_c.size(1)});  // {ZC,1,1} ===> {ZC}
        if (vm["RED"].as<bool>()){
            z_r1 = RelativeEuclideanDistance(image, output);
        }
        else{
            z_r1 = AbsoluteEuclideanDistance(image, output);
        }
        z_r2 = CosineSimilarity(image, output);
        z_r = torch::cat({z_r1, z_r2}, /*dim=*/1);  // {1} + {1} ===> {ZR} = {2}
        z = torch::cat({z_c, z_r}, /*dim=*/1);  // {ZC} + {ZR} ===> {Z} = {ZC+ZR}

        // (1.3) Calculation of Loss
        rec = criterion(output, image);
        anomaly_score = est->anomaly_score(z, mu, sigma, phi);

        // (1.4) Update Loss
        total_rec += rec.item<float>();
        score = anomaly_score.item<float>();
        score = (std::isinf(score) ? (float)(long)score : score);
        total_anomaly_score += score;

        iteration++;
    }

    // (2) Calculate Average Loss
    ave_rec = total_rec / (float)iteration;
    ave_anomaly_score = total_anomaly_score / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "rec:" << ave_rec << ' ' << std::flush;
    ofs << "anomaly_score:" << ave_anomaly_score << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_rec + ave_anomaly_score, ave_rec, ave_anomaly_score});
    writer_rec.plot(/*base=*/epoch, /*value=*/{ave_rec});

    // End Processing
    return;

}