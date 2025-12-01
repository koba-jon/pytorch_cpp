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
#include "networks.hpp"                // MC_ResNet
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
void train(po::variables_map &vm, torch::Device &device, MC_ResNet &model, std::vector<transforms_Compose> &transform){

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t total_iter;
    long int N, Nc, dim;
    std::string date;
    std::string checkpoint_dir, path;
    std::string dataroot;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, feature, features, memory_bank;
    torch::Tensor id, coreset_idx, NN_dist, NN_dist_new, coreset;
    torch::nn::Linear psi{nullptr};
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
    // a2. Feature Extraction
    // -----------------------------------
    
    // (1) Set Parameters
    total_iter = dataloader.get_count_max();

    // (2) Training
    torch::NoGradGuard no_grad;
    model->eval();
    show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{1, 1}, /*loss_=*/{});

    // (3) Tensor Forward
    image = torch::rand({1, 3, 224, 224}).to(device);
    feature = model->forward(image);
    features = torch::empty({0, feature.size(1), feature.size(2), feature.size(3)}).to(device);
    while (dataloader(mini_batch)){
        image = std::get<0>(mini_batch).to(device);
        feature = model->forward(image);
        features = torch::cat({features, feature}, 0);
        show_progress->increment(/*loss_value=*/{});
    }
    std::cout << std::endl;

    // -----------------------
    // a3. Coreset Sampling
    // -----------------------

    // (1) Set parameters
    N = features.size(0) * features.size(2) * features.size(3);
    Nc = N * vm["coreset_rate"].as<float>();
    dim = features.size(1);
    features = features.permute({0, 2, 3, 1}).contiguous().view({N, dim}).contiguous();  // {N,D,H,W} ===> {N,H,W,D} ===> {N*H*W,D}
    show_progress = new progress::display(/*count_max_=*/Nc, /*epoch=*/{1, 1}, /*loss_=*/{});

    // (2) Reduce dimensions
    psi = torch::nn::Linear(torch::nn::LinearOptions(dim, vm["d_star"].as<size_t>()).bias(false));
    psi->to(device);
    memory_bank = psi->forward(features);  // {N*H*W,D} ===> {N*H*W,DD}

    // (2) Select coreset
    id = torch::randint(0, N, {1}).to(device);  // {1}
    coreset_idx = id;  // {1}
    NN_dist = torch::sqrt((memory_bank - memory_bank.index_select(0, id)).pow(2.0).sum(1));  // {N*H*W,DD} ===> {N*H*W}
    show_progress->increment(/*loss_value=*/{});
    for (long int i = 1; i < Nc; i++){
        id = torch::argmax(NN_dist, /*dim=*/0, /*keepdim=*/true);  // {1}
        coreset_idx = torch::cat({coreset_idx, id}, 0);  // {c} + {1} ===> {c+1}
        NN_dist_new = torch::sqrt((memory_bank - memory_bank.index_select(0, id)).pow(2.0).sum(1));  // {N*H*W,DD} ===> {N*H*W}
        NN_dist = torch::minimum(NN_dist, NN_dist_new);  // {N*H*W}
        show_progress->increment(/*loss_value=*/{});
    }
    coreset = features.index_select(0, coreset_idx);  // {Nc,D}
    std::cout << std::endl;

    // -----------------------------------
    // a4. Save Model Weights
    // -----------------------------------
    path = checkpoint_dir + "/models/coreset.pth";  torch::save(coreset, path);

    // End Processing
    return;

}
