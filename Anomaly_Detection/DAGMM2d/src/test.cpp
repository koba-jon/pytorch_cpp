#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <chrono>                      // std::chrono
#include <cmath>                       // std::isinf
#include <utility>                     // std::pair
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // Encoder, Decoder, EstimationNetwork, RelativeEuclideanDistance, CosineSimilarity, load_params
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
void test(po::variables_map &vm, torch::Device &device, Encoder &enc, Decoder &dec, EstimationNetwork &est, std::vector<transforms_Compose> &transform){

    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    float ave_loss, ave_anomaly_score, score;
    double seconds, ave_time;
    std::string path, result_dir, output_dir, heatmap_dir, fname;
    std::string dataroot;
    std::ofstream ofs, ofs_loss, ofs_score;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, std::vector<std::string>> data;
    torch::Tensor image, output, heatmap;
    torch::Tensor z, z_c, z_r, z_r1, z_r2;
    torch::Tensor mu, sigma, phi;
    torch::Tensor loss, anomaly_score;
    datasets::ImageFolderWithPaths dataset;
    DataLoader::ImageFolderWithPaths dataloader;

    // (1) Get Test Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_dir"].as<std::string>();
    dataset = datasets::ImageFolderWithPaths(dataroot, transform);
    dataloader = DataLoader::ImageFolderWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_gmp.dat"; load_params(path, mu, sigma, phi);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_enc.pth"; torch::load(enc, path, device);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_dec.pth"; torch::load(dec, path, device);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + "_est.pth"; torch::load(est, path, device);
    mu = mu.to(device);
    sigma = sigma.to(device);
    phi = phi.to(device);

    // (3) Set Loss Function
    auto criterion = Loss(vm["loss"].as<std::string>());

    // (4) Initialization of Value
    ave_loss = 0.0;
    ave_anomaly_score = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    enc->eval();
    dec->eval();
    est->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    output_dir = result_dir + "/output";  fs::create_directories(output_dir);
    heatmap_dir = result_dir + "/heatmap";  fs::create_directories(heatmap_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    ofs_loss.open(result_dir + "/reconstruction_error.txt", std::ios::out);
    ofs_score.open(result_dir + "/anomaly_score.txt", std::ios::out);
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        // (5.1) Encoder-Decoder Forward
        z_c = enc->forward(image);   // {C,H,W} ===> {ZC,1,1}
        output = dec->forward(z_c);  // {ZC,1,1} ===> {C,H,W}

        // (5.2) Setting Latent Space
        z_c = z_c.view({z_c.size(0), z_c.size(1)});  // {ZC,1,1} ===> {ZC}
        if (vm["RED"].as<bool>()){
            z_r1 = RelativeEuclideanDistance(image, output);
        }
        else{
            z_r1 = AbsoluteEuclideanDistance(image, output);
        }
        z_r2 = CosineSimilarity(image, output);
        z_r = torch::cat({z_r1, z_r2}, /*dim=*/1);  // {1} + {1} ===> {ZR} = {2}
        z = torch::cat({z_c, z_r}, /*dim=*/1);  // {ZC} + {ZR} ===> {Z} = {ZC+ZR}

        // (5.3) Calculation of Anomaly Score
        anomaly_score = est->anomaly_score(z, mu, sigma, phi);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss = criterion(output, image);
        
        ave_loss += loss.item<float>();
        score = anomaly_score.item<float>();
        score = (std::isinf(score) ? (float)(long)score : score);
        ave_anomaly_score += score;
        ave_time += seconds;

        std::cout << '<' << std::get<1>(data).at(0) << "> " << vm["loss"].as<std::string>() << ':' << loss.item<float>() << " anomaly_score:" << score << std::endl;
        ofs << '<' << std::get<1>(data).at(0) << "> " << vm["loss"].as<std::string>() << ':' << loss.item<float>() << " anomaly_score:" << score << std::endl;
        ofs_loss << loss.item<float>() << std::endl;
        ofs_score << score << std::endl;

        fname = output_dir + '/' + std::get<1>(data).at(0);
        visualizer::save_image(output.detach(), fname, /*range=*/output_range, /*cols=*/1, /*padding=*/0);

        fname = heatmap_dir + '/' + std::get<1>(data).at(0);
        heatmap = visualizer::create_heatmap(torch::abs(image - output).mean(/*dim=*/1, /*keepdim=*/true), /*range=*/{0, (output_range.second - output_range.first) * vm["heatmap_max"].as<float>()});
        visualizer::save_image(heatmap.detach(), fname, /*range=*/{0.0, 1.0}, /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_anomaly_score = ave_anomaly_score / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> " << vm["loss"].as<std::string>() << ':' << ave_loss << " anomaly_score:" << ave_anomaly_score << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> " << vm["loss"].as<std::string>() << ':' << ave_loss << " anomaly_score:" << ave_anomaly_score << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();
    ofs_loss.close();
    ofs_score.close();

    // End Processing
    return;

}
