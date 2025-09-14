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
#include "networks.hpp"                // ResNet_Generator
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderPairWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderPairWithPaths
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, ResNet_Generator &genAB, ResNet_Generator &genBA, std::vector<transforms_Compose> &transformA, std::vector<transforms_Compose> &transformB){

    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    float ave_AtoB_loss, ave_BtoA_loss;
    double seconds, ave_time;
    std::string path, result_dir, fnameAB, fnameBA;
    std::string A_dir, B_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor realA, realB, fakeA, fakeB;
    torch::Tensor AtoB_loss, BtoA_loss;
    datasets::ImageFolderPairWithPaths dataset;
    DataLoader::ImageFolderPairWithPaths dataloader;

    // (1) Get Test Dataset
    A_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_A_dir"].as<std::string>();
    B_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_B_dir"].as<std::string>();
    dataset = datasets::ImageFolderPairWithPaths(A_dir, B_dir, transformA, transformB);
    dataloader = DataLoader::ImageFolderPairWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_genAB.pth";  torch::load(genAB, path, device);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_genBA.pth";  torch::load(genBA, path, device);

    // (3) Set Loss Function
    auto criterion = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kMean));

    // (4) Initialization of Value
    ave_AtoB_loss = 0.0;
    ave_BtoA_loss = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    genAB->eval();
    genBA->train();
    result_dir = vm["test_result_dir"].as<std::string>();
    fs::create_directories(result_dir);
    fs::create_directories(result_dir + "/AtoB");
    fs::create_directories(result_dir + "/BtoA");
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        realA = std::get<0>(data).to(device);
        realB = std::get<1>(data).to(device);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        fakeB = genAB->forward(realA);
        fakeA = genBA->forward(realB);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        AtoB_loss = criterion(fakeB, realB);
        BtoA_loss = criterion(fakeA, realA);
        
        ave_AtoB_loss += AtoB_loss.item<float>();
        ave_BtoA_loss += BtoA_loss.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> AtoB:" << AtoB_loss.item<float>() << " BtoA:" << BtoA_loss.item<float>() << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> AtoB:" << AtoB_loss.item<float>() << " BtoA:" << BtoA_loss.item<float>() << std::endl;

        fnameAB = result_dir + "/AtoB/" + std::get<3>(data).at(0);
        fnameBA = result_dir + "/BtoA/" + std::get<3>(data).at(0);
        visualizer::save_image(fakeB.detach(), fnameAB, /*range=*/output_range, /*cols=*/1, /*padding=*/0);
        visualizer::save_image(fakeA.detach(), fnameBA, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_AtoB_loss = ave_AtoB_loss / (float)dataset.size();
    ave_BtoA_loss = ave_BtoA_loss / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> AtoB:" << ave_AtoB_loss << " BtoA:" << ave_BtoA_loss << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> AtoB:" << ave_AtoB_loss << " BtoA:" << ave_BtoA_loss << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
