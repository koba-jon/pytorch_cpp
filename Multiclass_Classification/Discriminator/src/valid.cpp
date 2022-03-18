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
#include "networks.hpp"                // MC_Discriminator
#include "dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderClassesWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, MC_Discriminator &model, const std::vector<std::string> class_names, const size_t epoch, visualizer::graph &writer, visualizer::graph &writer_accuracy, visualizer::graph &writer_each_accuracy){

    constexpr size_t class_num_thresh = 10;  // threshold for the number of classes for determining whether to display accuracy graph for each class

    // (0) Initialization and Declaration
    size_t iteration;
    size_t mini_batch_size;
    size_t class_num;
    size_t total_match, total_counter;
    long int response, answer;
    float total_accuracy;
    float ave_loss, total_loss;
    std::ofstream ofs;
    std::vector<size_t> class_match, class_counter;
    std::vector<float> class_accuracy;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output, responses;

    // (1) Memory Allocation
    class_num = class_names.size();
    class_match = std::vector<size_t>(class_num, 0);
    class_counter = std::vector<size_t>(class_num, 0);
    class_accuracy = std::vector<float>(class_num, 0.0);

    // (2) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_loss = 0.0;
    total_match = 0; total_counter = 0;
    while (valid_dataloader(mini_batch)){
        
        image = std::get<0>(mini_batch).to(device);
        label = std::get<1>(mini_batch).to(device);
        mini_batch_size = image.size(0);

        output = model->forward(image);
        loss = criterion(output, label);

        responses = output.exp().argmax(/*dim=*/1);
        for (size_t i = 0; i < mini_batch_size; i++){
            response = responses[i].item<long int>();
            answer = label[i].item<long int>();
            class_counter[answer]++;
            total_counter++;
            if (response == answer){
                class_match[answer]++;
                total_match++;
            }
        }

        total_loss += loss.item<float>();
        iteration++;
    }

    // (3) Calculate Average Loss
    ave_loss = total_loss / (float)iteration;

    // (4) Calculate Accuracy
    for (size_t i = 0; i < class_num; i++){
        class_accuracy[i] = (float)class_match[i] / (float)class_counter[i];
    }
    total_accuracy = (float)total_match / (float)total_counter;

    // (5.1) Record Loss (Log/Loss)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "classify:" << ave_loss << ' ' << std::flush;
    ofs << "accuracy:" << total_accuracy << std::endl;
    ofs.close();

    // (5.2) Record Loss (Log/Accuracy)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.csv", std::ios::app);
    if (epoch == 1){
        ofs << "epoch," << std::flush;
        ofs << "accuracy," << std::flush;
        for (size_t i = 0; i < class_num; i++){
            ofs << i << "(" << class_names.at(i) << ")," << std::flush;
        }
        ofs << std::endl;
    }
    ofs << epoch << '/' << vm["epochs"].as<size_t>() << ',' << std::flush;
    ofs << total_accuracy << ',' << std::flush;
    for (size_t i = 0; i < class_num; i++){
        ofs << class_accuracy[i] << ',' << std::flush;
    }
    ofs << std::endl;
    ofs.close();

    // (5.3) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_loss});
    writer_accuracy.plot(/*base=*/epoch, /*value=*/{total_accuracy});
    if (class_num <= class_num_thresh){
        writer_each_accuracy.plot(/*base=*/epoch, /*value=*/class_accuracy);
    }

    // End Processing
    return;

}
