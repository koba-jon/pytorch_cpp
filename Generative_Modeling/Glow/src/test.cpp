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
#include "networks.hpp"                // NormalizingFlow
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
void test(po::variables_map &vm, torch::Device &device, NormalizingFlow &model, std::vector<transforms_Compose> &transform){

    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    float ave_loss, ave_rec_loss;
    double seconds, ave_time;
    std::string path, result_dir, fname;
    std::string dataroot;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, std::vector<std::string>> data;
    torch::Tensor sum_logdet, sum_log_p, rec, loss, image, output;
    std::vector<torch::Tensor> z_list;
    datasets::ImageFolderWithPaths dataset;
    DataLoader::ImageFolderWithPaths dataloader;

    // (1) Get Test Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_dir"].as<std::string>();
    dataset = datasets::ImageFolderWithPaths(dataroot, transform);
    dataloader = DataLoader::ImageFolderWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (3) Set Loss Function
    auto criterion = Loss();
    auto criterionMSE = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));

    // (4) Initialization of Value
    ave_loss = 0.0;
    ave_rec_loss = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        image = (image * 255.0 + torch::rand_like(image)) / 256.0;
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        std::tie(z_list, sum_logdet, sum_log_p) = model->forward(image);
        output = model->inverse(z_list);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss = criterion(sum_logdet, sum_log_p, vm["size"].as<size_t>() * vm["size"].as<size_t>() * vm["nc"].as<size_t>());
        rec = criterionMSE(output, image);
        
        ave_loss += loss.item<float>();
        ave_rec_loss += rec.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<1>(data).at(0) << "> loss:" << loss.item<float>() << " rec:" << rec.item<float>() << std::endl;
        ofs << '<' << std::get<1>(data).at(0) << "> loss:" << loss.item<float>() << " rec:" << rec.item<float>() << std::endl;

        fname = result_dir + '/' + std::get<1>(data).at(0);
        visualizer::save_image(output.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_rec_loss = ave_rec_loss / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> loss:" << ave_loss << " rec:" << ave_rec_loss << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> loss:" << ave_loss << " rec:" << ave_rec_loss << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
