#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <chrono>                      // std::chrono
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // MC_VGGNet
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, MC_VGGNet &model, std::vector<transforms_Compose> &transform, const std::vector<std::string> class_names){

    // (0) Initialization and Declaration
    size_t class_num;
    size_t match, counter;
    long int response, answer;
    char judge;
    float accuracy;
    float ave_loss;
    double seconds, ave_time;
    std::string path, result_dir;
    std::string dataroot;
    std::ofstream ofs, ofs2;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> data;
    torch::Tensor image, label, output;
    torch::Tensor loss;
    datasets::ImageFolderClassesWithPaths dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader;

    // (1) Get Test Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_dir"].as<std::string>();
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["test_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (3) Set Loss Function
    auto criterion = Loss();

    // (4) Initialization of Value
    ave_loss = 0.0;
    ave_time = 0.0;
    match = 0;
    counter = 0;
    class_num = class_names.size();

    // (5) File Pre-processing
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    ofs2.open(result_dir + "/likelihood.csv", std::ios::out);
    ofs2 << "file name," << std::flush;
    ofs2 << "judge," << std::flush;
    for (size_t i = 0; i < class_num; i++){
        ofs2 << i << "(" << class_names.at(i) << ")," << std::flush;
    }
    ofs2 << std::endl;

    // (6) Tensor Forward
    model->eval();
    while (dataloader(data)){
        
        image = std::get<0>(data).to(device);
        label = std::get<1>(data).to(device);
        
        torch::cuda::synchronize();
        start = std::chrono::system_clock::now();
        
        output = model->forward(image);

        torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        loss = criterion(output, label);
        
        ave_loss += loss.item<float>();
        ave_time += seconds;

        output = output.exp();
        response = output.argmax(/*dim=*/1).item<long int>();
        answer = label[0].item<long int>();
        counter += 1;
        judge = 'F';
        if (response == answer){
            match += 1;
            judge = 'T';
        }

        std::cout << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << " judge:" << judge << " response:" << response << '(' << class_names.at(response) << ") answer:" << answer << '(' << class_names.at(answer) << ')' << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << " judge:" << judge << " response:" << response << '(' << class_names.at(response) << ") answer:" << answer << '(' << class_names.at(answer) << ')' << std::endl;
        ofs2 << std::get<2>(data).at(0) << ',' << std::flush;
        ofs2 << judge << ',' << std::flush;
        output = output[0];  // {1, CN} ===> {CN}
        for (size_t i = 0; i < class_num; i++){
            ofs2 << output[i].item<float>() << ',' << std::flush;
        }
        ofs2 << std::endl;

    }

    // (7.1) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7.2) Calculate Accuracy
    accuracy = (float)match / float(counter);

    // (8) Average Output
    std::cout << "<All> cross-entropy:" << ave_loss << " accuracy:" << accuracy << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> cross-entropy:" << ave_loss << " accuracy:" << accuracy << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();
    ofs2.close();

    // End Processing
    return;

}
