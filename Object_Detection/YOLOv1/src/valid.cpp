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
#include "networks.hpp"                // YOLOv1
#include "dataloader.hpp"              // DataLoader::ImageFolderBBWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderBBWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, YOLOv1 &model, const std::vector<std::string> class_names, const size_t epoch, std::vector<visualizer::graph> &writer){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_loss_coord_xy, total_loss_coord_xy;
    float ave_loss_coord_wh, total_loss_coord_wh;
    float ave_loss_obj, total_loss_obj;
    float ave_loss_noobj, total_loss_noobj;
    float ave_loss_class, total_loss_class;
    float ave_loss_all;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor image, output;
    torch::Tensor loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> losses;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_loss_coord_xy = 0.0; total_loss_coord_wh = 0.0; total_loss_obj = 0.0; total_loss_noobj = 0.0; total_loss_class = 0.0;
    while (valid_dataloader(mini_batch)){
        
        image = std::get<0>(mini_batch).to(device);  // {N,C,H,W} (images)
        label = std::get<1>(mini_batch);  // {N, ({BB_n}, {BB_n,4}) } (annotations)
        
        output = model->forward(image);  // {N,C,H,W} ===> {N,G,G,FF}
        losses = criterion(output, label);

        loss_coord_xy = std::get<0>(losses) * vm["Lambda_coord"].as<float>();
        loss_coord_wh = std::get<1>(losses) * vm["Lambda_coord"].as<float>();
        loss_obj = std::get<2>(losses) * vm["Lambda_object"].as<float>();
        loss_noobj = std::get<3>(losses) * vm["Lambda_noobject"].as<float>();
        loss_class = std::get<4>(losses) * vm["Lambda_class"].as<float>();

        total_loss_coord_xy += loss_coord_xy.item<float>();
        total_loss_coord_wh += loss_coord_wh.item<float>();
        total_loss_obj += loss_obj.item<float>();
        total_loss_noobj += loss_noobj.item<float>();
        total_loss_class += loss_class.item<float>();

        iteration++;

    }

    // (2) Calculate Average Loss
    ave_loss_coord_xy = total_loss_coord_xy / (float)iteration;
    ave_loss_coord_wh = total_loss_coord_wh / (float)iteration;
    ave_loss_obj = total_loss_obj / (float)iteration;
    ave_loss_noobj = total_loss_noobj / (float)iteration;
    ave_loss_class = total_loss_class / (float)iteration;
    ave_loss_all = ave_loss_coord_xy + ave_loss_coord_wh + ave_loss_obj + ave_loss_noobj + ave_loss_class;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "coord_xy:" << ave_loss_coord_xy << ' ' << std::flush;
    ofs << "coord_wh:" << ave_loss_coord_wh << ' ' << std::flush;
    ofs << "conf_o:" << ave_loss_obj << ' ' << std::flush;
    ofs << "conf_x:" << ave_loss_noobj << ' ' << std::flush;
    ofs << "class:" << ave_loss_class << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.at(0).plot(/*base=*/epoch, /*value=*/{ave_loss_all});
    writer.at(1).plot(/*base=*/epoch, /*value=*/{ave_loss_coord_xy});
    writer.at(2).plot(/*base=*/epoch, /*value=*/{ave_loss_coord_wh});
    writer.at(3).plot(/*base=*/epoch, /*value=*/{ave_loss_obj});
    writer.at(4).plot(/*base=*/epoch, /*value=*/{ave_loss_noobj});
    writer.at(5).plot(/*base=*/epoch, /*value=*/{ave_loss_class});

    // End Processing
    return;

}