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
#include "networks.hpp"                // GS3D
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderCameraPoseWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderCameraPoseWithPaths
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, GS3D &model, std::vector<transforms_Compose> &transform){

    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    float ave_loss;
    double seconds, ave_time;
    std::string path, result_dir, fname;
    std::string image_dir, pose_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor image, pose, output, loss;
    std::tuple<torch::Tensor, torch::Tensor> x_t_with_noise;
    datasets::ImageFolderCameraPoseWithPaths dataset;
    DataLoader::ImageFolderCameraPoseWithPaths dataloader;

    // (1) Get Test Dataset
    image_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_image_dir"].as<std::string>();
    pose_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_pose_dir"].as<std::string>();
    dataset = datasets::ImageFolderCameraPoseWithPaths(image_dir, pose_dir, transform);
    dataloader = DataLoader::ImageFolderCameraPoseWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (3) Set Loss Function
    auto criterion = Loss(vm["Lambda"].as<float>(), device);

    // (4) Initialization of Value
    ave_loss = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        pose = std::get<1>(data).to(device);

        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        output = model->forward(pose);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss = criterion(output, image);
        
        ave_loss += loss.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> loss:" << loss.item<float>() << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> loss:" << loss.item<float>() << std::endl;

        fname = result_dir + '/' + std::get<2>(data).at(0);
        visualizer::save_image(output.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> loss:" << ave_loss << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> loss:" << ave_loss << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
