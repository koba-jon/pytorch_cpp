#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <utility>                     // std::pair
#include <iomanip>                     // std::setprecision
// For External Library
#include <torch/torch.h>               // torch
#include <opencv2/opencv.hpp>          // cv::Mat
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // YOLOv2
#include "detector.hpp"                // YOLODetector
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderPairWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderPairWithPaths
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// --------------------
// Detection Function
// --------------------
void detect(po::variables_map &vm, torch::Device &device, YOLOv2 &model, std::vector<transforms_Compose> &transformI, std::vector<transforms_Compose> &transformD, const std::vector<std::string> class_names, const std::vector<std::tuple<float, float>> anchors){

    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    size_t BB_n;
    float prob;
    std::string path, result_dir, fname;
    std::string dataroot;
    std::string class_name;
    std::stringstream ss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor imageI, imageD, output;
    torch::Tensor ids, coords, probs;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_result;
    cv::Mat imageO;
    datasets::ImageFolderPairWithPaths dataset;
    DataLoader::ImageFolderPairWithPaths dataloader;

    // (1) Get Detection Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["detect_dir"].as<std::string>();
    dataset = datasets::ImageFolderPairWithPaths(dataroot, dataroot, transformI, transformD);
    dataloader = DataLoader::ImageFolderPairWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total detect images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["detect_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path);

    // (3) Set Detector
    auto detector = YOLODetector(anchors, (long int)vm["class_num"].as<size_t>(), vm["prob_thresh"].as<float>(), vm["nms_thresh"].as<float>());
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette = detector.get_label_palette();

    // (4) Tensor Forward
    model->eval();
    result_dir = vm["detect_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(result_dir + "/detect.txt", std::ios::out);
    while (dataloader(data)){
        
        // (4.1) Get data
        imageI = std::get<0>(data).to(device);  // {1,C,H,W} (image for input)
        imageD = std::get<1>(data);  // {1,3,H_D,W_D} (image for detection)

        // (4.2) Inference and Detection
        output = model->forward(imageI);  // {1,C,H,W} ===> {1,G,G,FF}
        detect_result = detector(output[0]);  // output[0]{G,G,FF} ===> detect_result{ (ids{BB_n}, coords{BB_n,4}, probs{BB_n}) }
        ids = std::get<0>(detect_result);  // ids{BB_n}
        coords = std::get<1>(detect_result);  // coords{BB_n,4}
        probs = std::get<2>(detect_result);  // probs{BB_n}
        imageO = visualizer::draw_detections_des(imageD[0].detach(), {ids, coords}, probs, class_names, label_palette, /*range=*/output_range);

        // (4.3) Save image
        fname = result_dir + '/' + std::get<2>(data).at(0);  // {1,C,H,W} ===> {1,G,G,FF}
        cv::imwrite(fname, imageO);

        // (4.4) Write detection result
        BB_n = ids.size(0);
        std::cout << '<' << std::get<2>(data).at(0) << "> " << BB_n << " { " << std::flush;
        ofs << '<' << std::get<2>(data).at(0) << "> " << BB_n << " { " << std::flush;
        for (size_t i = 0; i < BB_n; i++){
            class_name = class_names.at(ids[i].item<long int>());
            prob = probs[i].item<float>() * 100.0;
            ss.str(""); ss.clear(std::stringstream::goodbit);
            ss << std::fixed << std::setprecision(1) << prob;
            std::cout << class_name << ":" << ss.str() << "% " << std::flush;
            ofs << class_name << ":" << ss.str() << "% " << std::flush;
        }
        std::cout << "}" << std::endl;
        ofs << "}" << std::endl;

    }

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
