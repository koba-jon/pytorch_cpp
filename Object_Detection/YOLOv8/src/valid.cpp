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
#include "networks.hpp"                // YOLOv8
#include "dataloader.hpp"              // DataLoader::ImageFolderBBWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderBBWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, YOLOv8 &model, const std::vector<std::string> class_names, const size_t epoch, std::vector<visualizer::graph> &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_loss_box, total_loss_box;
    float ave_loss_obj, total_loss_obj;
    float ave_loss_class, total_loss_class;
    float ave_loss_all;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor image;
    torch::Tensor loss_box, loss_obj, loss_class;
    std::vector<torch::Tensor> output;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> losses;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_loss_box = 0.0; total_loss_obj = 0.0; total_loss_class = 0.0;
    while (valid_dataloader(mini_batch)){
        
        image = std::get<0>(mini_batch).to(device);  // {N,C,H,W} (images)
        label = std::get<1>(mini_batch);  // {N, ({BB_n}, {BB_n,4}) } (annotations)
        
        output = model->forward(image);  // {N,C,H,W} ===> {S,{N,G,G,FF}}
        losses = criterion(output, label);

        loss_box = std::get<0>(losses) * vm["Lambda_box"].as<float>();
        loss_obj = std::get<1>(losses) * vm["Lambda_obj"].as<float>();
        loss_class = std::get<2>(losses) * vm["Lambda_class"].as<float>();

        total_loss_box += loss_box.item<float>();
        total_loss_obj += loss_obj.item<float>();
        total_loss_class += loss_class.item<float>();

        iteration++;

    }

    // (2) Calculate Average Loss
    ave_loss_box = total_loss_box / (float)iteration;
    ave_loss_obj = total_loss_obj / (float)iteration;
    ave_loss_class = total_loss_class / (float)iteration;
    ave_loss_all = ave_loss_box + ave_loss_obj + ave_loss_class;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "box:" << ave_loss_box << ' ' << std::flush;
    ofs << "obj:" << ave_loss_obj << ' ' << std::flush;
    ofs << "class:" << ave_loss_class << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.at(0).plot(/*base=*/epoch, /*value=*/{ave_loss_all});
    writer.at(1).plot(/*base=*/epoch, /*value=*/{ave_loss_box});
    writer.at(2).plot(/*base=*/epoch, /*value=*/{ave_loss_obj});
    writer.at(3).plot(/*base=*/epoch, /*value=*/{ave_loss_class});

    // End Processing
    return;

}