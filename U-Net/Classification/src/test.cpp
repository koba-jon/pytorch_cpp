#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <string>                      // std::string
#include <chrono>                      // std::chrono
#include <sys/stat.h>                  // mkdir
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // UNet
#include "transforms.hpp"              // transforms::Compose
#include "datasets.hpp"                // datasets::ImageFolderSegmentWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderSegmentWithPaths
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace po = boost::program_options;


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, UNet &model, std::vector<transforms::Compose*> &transformI, std::vector<transforms::Compose*> &transformO){

    // (0) Initialization and Declaration
    float ave_loss;
    double seconds, ave_time;
    std::string path, result_dir, fname;
    std::string input_dir, output_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> data;
    torch::Tensor image, label, output, output_argmax;
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
    ave_time = 0.0;

    // (5) Tensor Forward
    model->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  mkdir(result_dir.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        label = std::get<1>(data).to(device);
        
        start = std::chrono::system_clock::now();
        
        output = model->forward(image);

        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss = criterion(output, label);
        
        ave_loss += loss.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << std::endl;

        fname = result_dir + '/' + std::get<3>(data).at(0);
        output_argmax = output.exp().argmax(/*dim=*/1, /*keepdim=*/true);
        visualizer::save_label(output_argmax.detach(), fname, std::get<4>(data), /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> cross-entropy:" << ave_loss << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> cross-entropy:" << ave_loss << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}