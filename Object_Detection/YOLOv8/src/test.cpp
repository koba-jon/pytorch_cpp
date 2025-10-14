#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <chrono>                      // std::chrono
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // YOLOv8
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderBBWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderBBWithPaths

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, YOLOv8 &model, std::vector<transforms_Compose> &transform){

    // (0) Initialization and Declaration
    float ave_loss_box, ave_loss_obj, ave_loss_class;
    double seconds, ave_time;
    std::string path, result_dir;
    std::string input_dir, output_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor image;
    torch::Tensor loss_box, loss_obj, loss_class;
    std::vector<torch::Tensor> output;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> losses;
    datasets::ImageFolderBBWithPaths dataset;
    DataLoader::ImageFolderBBWithPaths dataloader;
    std::vector<transforms_Compose> null;

    // (1) Get Test Dataset
    input_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_in_dir"].as<std::string>();
    output_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_out_dir"].as<std::string>();
    dataset = datasets::ImageFolderBBWithPaths(input_dir, output_dir, null, transform);
    dataloader = DataLoader::ImageFolderBBWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (3) Set Loss Function
    auto criterion = Loss((long int)vm["class_num"].as<size_t>(), (long int)vm["reg_max"].as<size_t>());

    // (4) Initialization of Value
    ave_loss_box = 0.0;
    ave_loss_obj = 0.0;
    ave_loss_class = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        label = std::get<1>(data);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        output = model->forward(image);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        losses = criterion(output, label);
        loss_box = std::get<0>(losses) * vm["Lambda_box"].as<float>();
        loss_obj = std::get<1>(losses) * vm["Lambda_obj"].as<float>();
        loss_class = std::get<2>(losses) * vm["Lambda_class"].as<float>();
        
        ave_loss_box += loss_box.item<float>();
        ave_loss_obj += loss_obj.item<float>();
        ave_loss_class += loss_class.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> box:" << loss_box.item<float>() << " obj:" << loss_obj.item<float>() << " class:" << loss_class.item<float>() << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> box:" << loss_box.item<float>() << " obj:" << loss_obj.item<float>() << " class:" << loss_class.item<float>() << std::endl;

    }

    // (6) Calculate Average
    ave_loss_box = ave_loss_box / (float)dataset.size();
    ave_loss_obj = ave_loss_obj / (float)dataset.size();
    ave_loss_class = ave_loss_class / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> box:" << ave_loss_box << " obj:" << ave_loss_obj <<  " class:" << ave_loss_class << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> box:" << ave_loss_box << " obj:" << ave_loss_obj << " class:" << ave_loss_class << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
