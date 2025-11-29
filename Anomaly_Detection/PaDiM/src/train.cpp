#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <utility>                     // std::pair
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // MC_ResNet, SelectIndex
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// -------------------
// Training Function
// -------------------
void train(po::variables_map &vm, torch::Device &device, MC_ResNet &model, SelectIndex &select, std::vector<transforms_Compose> &transform){

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t total_iter;
    size_t N, dim, feature_size;
    std::string date;
    std::string checkpoint_dir, path;
    std::string dataroot;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, feature, features;
    torch::Tensor mean, cov, xm, xm_perm, xm_t, I;
    datasets::ImageFolderWithPaths dataset;
    DataLoader::ImageFolderWithPaths dataloader;
    progress::display *show_progress;

    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_dir"].as<std::string>();
    dataset = datasets::ImageFolderWithPaths(dataroot, transform);
    dataloader = DataLoader::ImageFolderWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // (2) Make Directories
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/models";  fs::create_directories(path);

    // (3) Get Weights
    torch::load(model, vm["resnet_path"].as<std::string>(), device);

    // (4) Display Date
    date = progress::current_date();
    date = progress::separator_center("Train (" + date + ")");
    std::cout << std::endl << std::endl << date << std::endl;

    // -----------------------------------
    // a2. Training Setting
    // -----------------------------------
    
    // (1) Set Parameters
    total_iter = dataloader.get_count_max();

    // (2) Training
    torch::NoGradGuard no_grad;
    model->eval();
    select->eval();
    show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{1, 1}, /*loss_=*/{});

    // -----------------------------------
    // a3. Feature Extraction
    // -----------------------------------
    image = torch::rand({1, 3, 224, 224}).to(device);
    feature = select->forward(model->forward(image));
    features = torch::empty({0, feature.size(1), feature.size(2), feature.size(3)}).to(device);
    while (dataloader(mini_batch)){
        image = std::get<0>(mini_batch).to(device);
        feature = select->forward(model->forward(image));
        features = torch::cat({features, feature}, 0);
        show_progress->increment(/*loss_value=*/{});
    }

    // ------------------------------------------
    // a4. Calculate Mean and Covariance Matrix
    // ------------------------------------------

    // (1) Set parameters
    N = features.size(0);
    dim = features.size(1);
    feature_size = features.size(2) * features.size(3);
    features = features.view({(long int)N, (long int)dim, (long int)feature_size});  // {N,D,H*W}

    // (2) Calculate mean
    mean = torch::mean(features, /*dim=*/0);  // {D,H*W}

    // (3) Calculate covariance matrix
    xm = features - mean.unsqueeze(0);  // {N,D,H*W}
    xm_perm = xm.permute({2, 1, 0});  // {H*W,D,N}
    xm_t = xm_perm.transpose(1, 2);  // {H*W,N,D}
    cov = torch::bmm(xm_perm, xm_t) / (N - 1);  // {H*W,D,D}
    I = torch::eye(dim).unsqueeze(0).to(device) * 0.01;  // {1,D,D}
    cov = cov + I;  // {H*W,D,D}
    cov = cov.permute({1, 2, 0}).contiguous();  // {D,D,H*W}
    
    // -----------------------------------
    // a5. Save Model Weights
    // -----------------------------------
    path = checkpoint_dir + "/models/select.pth";  torch::save(select, path);
    path = checkpoint_dir + "/models/mean.pth";  torch::save(mean, path);
    path = checkpoint_dir + "/models/cov.pth";  torch::save(cov, path);

    // End Processing
    return;

}
