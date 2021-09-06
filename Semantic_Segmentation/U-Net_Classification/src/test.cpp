#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <chrono>                      // std::chrono
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // UNet
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderSegmentWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderSegmentWithPaths
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, UNet &model, std::vector<transforms_Compose> &transformI, std::vector<transforms_Compose> &transformO){

    // (0) Initialization and Declaration
    size_t correct, correct_per_class, total_class_pixel, class_count;
    float ave_loss;
    double seconds, ave_time;
    double pixel_wise_accuracy, ave_pixel_wise_accuracy;
    double mean_accuracy, ave_mean_accuracy;
    std::string path, result_dir, fname;
    std::string input_dir, output_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> data;
    torch::Tensor image, label, output, output_argmax, answer_mask, response_mask;
    torch::Tensor loss;
    datasets::ImageFolderSegmentWithPaths dataset;
    DataLoader::ImageFolderSegmentWithPaths dataloader;

    // (1) Get Test Dataset
    input_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_in_dir"].as<std::string>();
    output_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_out_dir"].as<std::string>();
    dataset = datasets::ImageFolderSegmentWithPaths(input_dir, output_dir, transformI, transformO);
    dataloader = DataLoader::ImageFolderSegmentWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path);

    // (3) Set Loss Function
    auto criterion = Loss();

    // (4) Initialization of Value
    ave_loss = 0.0;
    ave_pixel_wise_accuracy = 0.0;
    ave_mean_accuracy = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    model->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        label = std::get<1>(data).to(device);
        
        torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        output = model->forward(image);

        torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;

        loss = criterion(output, label);
        
        output_argmax = output.exp().argmax(/*dim=*/1, /*keepdim=*/true);
        correct = (label == output_argmax).sum().item<long int>();
        pixel_wise_accuracy = (double)correct / (double)(label.size(0) * label.size(1) * label.size(2));

        class_count = 0;
        mean_accuracy = 0.0;
        for (size_t i = 0; i < std::get<4>(data).size(); i++){
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

        ave_loss += loss.item<float>();
        ave_pixel_wise_accuracy += pixel_wise_accuracy;
        ave_mean_accuracy += mean_accuracy;
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << " pixel-wise-accuracy:" << pixel_wise_accuracy << " mean-accuracy:" << mean_accuracy << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << " pixel-wise-accuracy:" << pixel_wise_accuracy << " mean-accuracy:" << mean_accuracy << std::endl;

        fname = result_dir + '/' + std::get<3>(data).at(0);
        visualizer::save_label(output_argmax.detach(), fname, std::get<4>(data), /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_pixel_wise_accuracy = ave_pixel_wise_accuracy / (double)dataset.size();
    ave_mean_accuracy = ave_mean_accuracy / (double)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> cross-entropy:" << ave_loss << " pixel-wise-accuracy:" << ave_pixel_wise_accuracy << " mean-accuracy:" << ave_mean_accuracy << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> cross-entropy:" << ave_loss << " pixel-wise-accuracy:" << ave_pixel_wise_accuracy << " mean-accuracy:" << ave_mean_accuracy << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}