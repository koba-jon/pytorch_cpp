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
#include "loss.hpp"                    // Loss_PixelCNN
#include "networks.hpp"                // VQVAE
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// ---------------
// Test Function
// ---------------
void test2(po::variables_map &vm, torch::Device &device, VQVAE &vqvae, GatedPixelCNN &model, std::vector<transforms_Compose> &transform){

    // (0) Initialization and Declaration
    float ave_loss;
    double seconds, ave_time;
    std::string path, result_dir;
    std::string dataroot;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, std::vector<std::string>> data;
    torch::Tensor image, idx, output;
    torch::Tensor loss;
    datasets::ImageFolderWithPaths dataset;
    DataLoader::ImageFolderWithPaths dataloader;

    // (1) Get Test Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test2_dir"].as<std::string>();
    dataset = datasets::ImageFolderWithPaths(dataroot, transform);
    dataloader = DataLoader::ImageFolderWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test2_vqvae_load_epoch"].as<std::string>() + "_vqvae.pth"; torch::load(vqvae, path, device);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test2_pixelcnn_load_epoch"].as<std::string>() + "_pixelcnn.pth"; torch::load(model, path, device);

    // (3) Set Loss Function
    auto criterion = Loss_PixelCNN(vm["K"].as<size_t>());

    // (4) Initialization of Value
    ave_loss = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    vqvae->eval();
    model->eval();
    result_dir = vm["test2_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        idx = vqvae->forward_idx(image);
        output = model->forward(idx);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss = criterion(output, idx);
        
        ave_loss += loss.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<1>(data).at(0) << "> " << "loss:" << loss.item<float>() << std::endl;
        ofs << '<' << std::get<1>(data).at(0) << "> " << "loss:" << loss.item<float>() << std::endl;

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> " << "loss:" << ave_loss << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> " << "loss:" << ave_loss << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
