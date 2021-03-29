#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <chrono>                      // std::chrono
#include <utility>                     // std::pair
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // AutoEncoder1d
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::Data1dFolderPairWithPaths
#include "dataloader.hpp"              // DataLoader::Data1dFolderPairWithPaths

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, AutoEncoder1d &model, std::vector<transforms_Compose> &transform){

    // (0) Initialization and Declaration
    float ave_loss, ave_GT_loss;
    double seconds, ave_time;
    std::string path, result_dir, fname;
    std::string input_dir, output_dir;
    std::ofstream ofs, ofs_data;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor dataI, dataO, output;
    torch::Tensor loss, GT_loss;
    datasets::Data1dFolderPairWithPaths dataset;
    DataLoader::Data1dFolderPairWithPaths dataloader;

    // (1) Get Test Dataset
    input_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_in_dir"].as<std::string>();
    output_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_out_dir"].as<std::string>();
    dataset = datasets::Data1dFolderPairWithPaths(input_dir, output_dir, transform, transform);
    dataloader = DataLoader::Data1dFolderPairWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test data : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path);

    // (3) Set Loss Function
    auto criterion = Loss(vm["loss"].as<std::string>());

    // (4) Initialization of Value
    ave_loss = 0.0;
    ave_GT_loss = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    model->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        dataI = std::get<0>(data).to(device);
        dataO = std::get<1>(data).to(device);
        
        start = std::chrono::system_clock::now();
        
        output = model->forward(dataI);  // {1,D} ===> {1,D}

        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss = criterion(output, dataI);
        GT_loss = criterion(output, dataO);
        
        ave_loss += loss.item<float>();
        ave_GT_loss += GT_loss.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> " << vm["loss"].as<std::string>() << ':' << loss.item<float>() << " GT_" << vm["loss"].as<std::string>() << ':' << GT_loss.item<float>() << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> " << vm["loss"].as<std::string>() << ':' << loss.item<float>() << " GT_" << vm["loss"].as<std::string>() << ':' << GT_loss.item<float>() << std::endl;

        fname = result_dir + '/' + std::get<3>(data).at(0);
        ofs_data.open(fname, std::ios::out);
        for (long int i = 0; i < output.size(1); i++){
            ofs_data << output[0][i].item<float>() << std::endl;
        }
        ofs_data.close();

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_GT_loss = ave_GT_loss / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> " << vm["loss"].as<std::string>() << ':' << ave_loss << " GT_" << vm["loss"].as<std::string>() << ':' << ave_GT_loss << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> " << vm["loss"].as<std::string>() << ':' << ave_loss << " GT_" << vm["loss"].as<std::string>() << ':' << ave_GT_loss << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
