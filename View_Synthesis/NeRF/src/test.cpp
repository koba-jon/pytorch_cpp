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
#include "networks.hpp"                // NeRF
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
void test(po::variables_map &vm, torch::Device &device, NeRF &model, std::vector<transforms_Compose> &transform){

    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    float ave_loss;
    double seconds, ave_time;
    std::string path, result_dir, fname;
    std::string image_dir, pose_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor image, pose, rays_o, rays_d, target_rgb, rgb_fine, rgb_coarse, rendered;
    torch::Tensor loss, loss_fine, loss_coarse;
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
    auto criterion = Loss(vm["loss"].as<std::string>());

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

        std::tie(rays_o, rays_d) = model->build_rays(pose);
        target_rgb = image.permute({0, 2, 3, 1}).view({image.size(0), -1, 3}).contiguous();
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        std::tie(rgb_fine, rgb_coarse) = model->forward(rays_o, rays_d);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss_fine = criterion(rgb_fine, target_rgb);
        loss_coarse = criterion(rgb_coarse, target_rgb);
        loss = loss_fine + loss_coarse;
        
        ave_loss += loss.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> " << vm["loss"].as<std::string>() << ':' << loss.item<float>() << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> " << vm["loss"].as<std::string>() << ':' << loss.item<float>() << std::endl;

        rendered = model->render_image(pose);
        fname = result_dir + '/' + std::get<3>(data).at(0);
        visualizer::save_image(rendered.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> " << vm["loss"].as<std::string>() << ':' << ave_loss << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> " << vm["loss"].as<std::string>() << ':' << ave_loss << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
