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
#include "networks.hpp"                // Encoder, Decoder
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
void test(po::variables_map &vm, torch::Device &device, Encoder &enc1, Encoder &enc2, Decoder &dec, std::vector<transforms_Compose> &transform){

    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    float ave_con_loss, ave_enc_loss, ave_anomaly_score;
    double seconds, ave_time;
    std::string path, result_dir, fname;
    std::string dataroot;
    std::ofstream ofs, ofs_score;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, std::vector<std::string>> data;
    torch::Tensor image, z, output, z_rec;
    torch::Tensor con_loss, enc_loss, anomaly_score;
    datasets::ImageFolderWithPaths dataset;
    DataLoader::ImageFolderWithPaths dataloader;

    // (1) Get Test Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_dir"].as<std::string>();
    dataset = datasets::ImageFolderWithPaths(dataroot, transform);
    dataloader = DataLoader::ImageFolderWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_enc1.pth"; torch::load(enc1, path);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_enc2.pth"; torch::load(enc2, path);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_dec.pth"; torch::load(dec, path);

    // (3) Set Loss Function
    auto criterion_con = Loss(vm["loss_con"].as<std::string>());
    auto criterion_enc = Loss(vm["loss_enc"].as<std::string>());

    // (4) Initialization of Value
    ave_con_loss = 0.0;
    ave_enc_loss = 0.0;
    ave_anomaly_score = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    enc1->eval();
    enc2->eval();
    dec->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    ofs_score.open(result_dir + "/anomaly_score.txt", std::ios::out);
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        
        start = std::chrono::system_clock::now();
        
        z = enc1->forward(image);
        output = dec->forward(z);
        z_rec = enc2->forward(output);
        con_loss = criterion_con(output, image) * vm["Lambda_con"].as<float>();
        enc_loss = criterion_enc(z_rec, z) * vm["Lambda_enc"].as<float>();
        anomaly_score = torch::abs(z - z_rec).sum();

        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        ave_con_loss += con_loss.item<float>();
        ave_enc_loss += enc_loss.item<float>();
        ave_anomaly_score += anomaly_score.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<1>(data).at(0) << "> con_" << vm["loss_con"].as<std::string>() << ':' << con_loss.item<float>() << " enc_" << vm["loss_enc"].as<std::string>() << ':' << enc_loss.item<float>() << " anomaly_score:" << anomaly_score.item<float>() << std::endl;
        ofs << '<' << std::get<1>(data).at(0) << "> con_" << vm["loss_con"].as<std::string>() << ':' << con_loss.item<float>() << " enc_" << vm["loss_enc"].as<std::string>() << ':' << enc_loss.item<float>() << " anomaly_score:" << anomaly_score.item<float>() << std::endl;
        ofs_score << anomaly_score.item<float>() << std::endl;

        fname = result_dir + '/' + std::get<1>(data).at(0);
        visualizer::save_image(output.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_con_loss = ave_con_loss / (float)dataset.size();
    ave_enc_loss = ave_enc_loss / (float)dataset.size();
    ave_anomaly_score = ave_anomaly_score / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> con_" << vm["loss_con"].as<std::string>() << ':' << ave_con_loss << " enc_" << vm["loss_enc"].as<std::string>() << ':' << ave_enc_loss << " anomaly_score:" << ave_anomaly_score << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> con_" << vm["loss_con"].as<std::string>() << ':' << ave_con_loss << " enc_" << vm["loss_enc"].as<std::string>() << ':' << ave_enc_loss << " anomaly_score:" << ave_anomaly_score << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();
    ofs_score.close();

    // End Processing
    return;

}
