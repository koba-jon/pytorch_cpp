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
#include "networks.hpp"                // UNet
#include "dataloader.hpp"              // DataLoader::ImageFolderSegmentWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderSegmentWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, UNet &model, const size_t epoch, visualizer::graph &writer){

    // (0) Initialization and Declaration
    size_t correct, correct_per_class, total_class_pixel, class_count;
    size_t iteration;
    float ave_loss, total_loss;
    double pixel_wise_accuracy, ave_pixel_wise_accuracy, total_pixel_wise_accuracy;
    double mean_accuracy, ave_mean_accuracy, total_mean_accuracy;
    std::ofstream ofs;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> mini_batch;
    torch::Tensor image, label, output, output_argmax, answer_mask, response_mask;
    torch::Tensor loss;

    // (1) Tensor Forward per Mini Batch
    model->eval();
    iteration = 0;
    total_loss = 0.0;
    total_pixel_wise_accuracy = 0.0;
    total_mean_accuracy = 0.0;
    while (valid_dataloader(mini_batch)){

        image = std::get<0>(mini_batch).to(device);
        label = std::get<1>(mini_batch).to(device);

        output = model->forward(image);
        loss = criterion(output, label);

        output_argmax = output.exp().argmax(/*dim=*/1, /*keepdim=*/true);
        correct = (label == output_argmax).sum().item<long int>();
        pixel_wise_accuracy = (double)correct / (double)(label.size(0) * label.size(1) * label.size(2));

        class_count = 0;
        mean_accuracy = 0.0;
        for (size_t i = 0; i < std::get<4>(mini_batch).size(); i++){
            answer_mask = torch::full({label.size(0), label.size(1), label.size(2)}, /*value=*/(long int)i, torch::TensorOptions().dtype(torch::kLong)).to(device);
            total_class_pixel = (label == answer_mask).sum().item<long int>();
            if (total_class_pixel != 0){
                response_mask = torch::full({label.size(0), label.size(1), label.size(2)}, /*value=*/2, torch::TensorOptions().dtype(torch::kLong)).to(device);
                correct_per_class = (((label == output_argmax).to(torch::kLong) + (label == answer_mask).to(torch::kLong)) == response_mask).sum().item<long int>();
                mean_accuracy += (double)correct_per_class / (double)total_class_pixel;
                class_count++;
            }
        }
        mean_accuracy = mean_accuracy / (double)class_count;

        total_loss += loss.item<float>();
        total_pixel_wise_accuracy += pixel_wise_accuracy;
        total_mean_accuracy += mean_accuracy;
        iteration++;
    }

    // (2) Calculate Average Loss
    ave_loss = total_loss / (float)iteration;
    ave_pixel_wise_accuracy = total_pixel_wise_accuracy / (double)iteration;
    ave_mean_accuracy = total_mean_accuracy / (double)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "classify:" << ave_loss << ' ' << std::flush;
    ofs << "pixel-wise-accuracy:" << ave_pixel_wise_accuracy << ' ' << std::flush;
    ofs << "mean-accuracy:" << ave_mean_accuracy << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_loss});

    // End Processing
    return;

}