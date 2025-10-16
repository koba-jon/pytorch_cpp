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
#include "networks.hpp"                // UNet_Generator
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
void test(po::variables_map &vm, torch::Device &device, UNet_Generator &gen, std::vector<transforms_Compose> &transformI, std::vector<transforms_Compose> &transformO){

    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    float ave_loss_l1, ave_loss_l2;
    double seconds, ave_time;
    std::string path, result_dir, fname;
    std::string input_dir, output_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor realI, realO, fakeO;
    torch::Tensor loss_l1, loss_l2;
    datasets::ImageFolderPairWithPaths dataset;
    DataLoader::ImageFolderPairWithPaths dataloader;

    // (1) Get Test Dataset
    input_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_in_dir"].as<std::string>();
    output_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_out_dir"].as<std::string>();
    dataset = datasets::ImageFolderPairWithPaths(input_dir, output_dir, transformI, transformO);
    dataloader = DataLoader::ImageFolderPairWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_gen.pth";
    torch::load(gen, path, device);

    // (3) Set Loss Function
    auto criterion_L1 = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kMean));
    auto criterion_L2 = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));

    // (4) Initialization of Value
    ave_loss_l1 = 0.0;
    ave_loss_l2 = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    gen->train();  // Dropout is required to make the generated images diverse
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        realI = std::get<0>(data).to(device);
        realO = std::get<1>(data).to(device);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        fakeO = gen->forward(realI);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss_l1 = criterion_L1(fakeO, realO);
        loss_l2 = criterion_L2(fakeO, realO);
        
        ave_loss_l1 += loss_l1.item<float>();
        ave_loss_l2 += loss_l2.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> L1:" << loss_l1.item<float>() << " L2:" << loss_l2.item<float>() << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> L1:" << loss_l1.item<float>() << " L2:" << loss_l2.item<float>() << std::endl;

        fname = result_dir + '/' + std::get<3>(data).at(0);
        visualizer::save_image(fakeO.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_loss_l1 = ave_loss_l1 / (float)dataset.size();
    ave_loss_l2 = ave_loss_l2 / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> L1:" << ave_loss_l1 << " L2:" << ave_loss_l2 << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> L1:" << ave_loss_l1 << " L2:" << ave_loss_l2 << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
