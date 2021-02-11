#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <tuple>                       // std::tuple
#include <chrono>                      // std::chrono
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // YOLOv1
#include "transforms.hpp"              // transforms::Compose
#include "datasets.hpp"                // datasets::ImageFolderBBWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderBBWithPaths

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, YOLOv1 &model, std::vector<transforms::Compose*> &transform, const std::vector<std::string> class_names){

    // (0) Initialization and Declaration
    float ave_loss_coord_xy, ave_loss_coord_wh, ave_loss_obj, ave_loss_noobj, ave_loss_class;
    double seconds, ave_time;
    std::string path, result_dir;
    std::string input_dir, output_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor image, output;
    torch::Tensor loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> losses;
    datasets::ImageFolderBBWithPaths dataset;
    DataLoader::ImageFolderBBWithPaths dataloader;
    std::vector<transforms::Compose*> null;

    // (1) Get Test Dataset
    input_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_in_dir"].as<std::string>();
    output_dir = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_out_dir"].as<std::string>();
    dataset = datasets::ImageFolderBBWithPaths(input_dir, output_dir, null, transform);
    dataloader = DataLoader::ImageFolderBBWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path);

    // (3) Set Loss Function
    auto criterion = Loss((long int)vm["class_num"].as<size_t>(), (long int)vm["ng"].as<size_t>(), (long int)vm["nb"].as<size_t>());

    // (4) Initialization of Value
    ave_loss_coord_xy = 0.0;
    ave_loss_coord_wh = 0.0;
    ave_loss_obj = 0.0;
    ave_loss_noobj = 0.0;
    ave_loss_class = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    model->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        label = std::get<1>(data);
        
        start = std::chrono::system_clock::now();
        
        output = model->forward(image);

        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        losses = criterion(output, label);
        loss_coord_xy = std::get<0>(losses) * vm["Lambda_coord"].as<float>();
        loss_coord_wh = std::get<1>(losses) * vm["Lambda_coord"].as<float>();
        loss_obj = std::get<2>(losses) * vm["Lambda_object"].as<float>();
        loss_noobj = std::get<3>(losses) * vm["Lambda_noobject"].as<float>();
        loss_class = std::get<4>(losses) * vm["Lambda_class"].as<float>();
        
        ave_loss_coord_xy += loss_coord_xy.item<float>();
        ave_loss_coord_wh += loss_coord_wh.item<float>();
        ave_loss_obj += loss_obj.item<float>();
        ave_loss_noobj += loss_noobj.item<float>();
        ave_loss_class += loss_class.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> coord_xy:" << loss_coord_xy.item<float>() << " coord_wh:" << loss_coord_wh.item<float>() << " conf_o:" << loss_obj.item<float>() << " conf_x:" << loss_noobj.item<float>() << " class:" << loss_class.item<float>() << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> coord_xy:" << loss_coord_xy.item<float>() << " coord_wh:" << loss_coord_wh.item<float>() << " conf_o:" << loss_obj.item<float>() << " conf_x:" << loss_noobj.item<float>() << " class:" << loss_class.item<float>() << std::endl;

    }
    ofs << "coord_xy:" << ave_loss_coord_xy << ' ' << std::flush;
    ofs << "coord_wh:" << ave_loss_coord_wh << ' ' << std::flush;
    ofs << "conf_o:" << ave_loss_obj << ' ' << std::flush;
    ofs << "conf_x:" << ave_loss_noobj << ' ' << std::flush;
    ofs << "class:" << ave_loss_class << std::endl;

    // (6) Calculate Average
    ave_loss_coord_xy = ave_loss_coord_xy / (float)dataset.size();
    ave_loss_coord_wh = ave_loss_coord_wh / (float)dataset.size();
    ave_loss_obj = ave_loss_obj / (float)dataset.size();
    ave_loss_noobj = ave_loss_noobj / (float)dataset.size();
    ave_loss_class = ave_loss_class / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> coord_xy:" << ave_loss_coord_xy << " coord_wh:" << ave_loss_coord_wh << " conf_o:" << ave_loss_obj << " conf_x:" << ave_loss_noobj << " class:" << ave_loss_class << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> coord_xy:" << ave_loss_coord_xy << " coord_wh:" << ave_loss_coord_wh << " conf_o:" << ave_loss_obj << " conf_x:" << ave_loss_noobj << " class:" << ave_loss_class << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
