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
#include "loss.hpp"                    // Loss_PixelSnail
#include "networks.hpp"                // VQVAE2, PixelSnail
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void valid3(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss_PixelSnail &criterion, VQVAE2 &vqvae2, PixelSnail &model, const size_t epoch, visualizer::graph &writer);


// -------------------
// Training Function
// -------------------
void train3(po::variables_map &vm, torch::Device &device, VQVAE2 &vqvae2, PixelSnail &model, std::vector<transforms_Compose> &transform){

    constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
    constexpr size_t train_workers = 4;  // the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 4;  // the number of workers to retrieve data from the validation dataset

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t epoch;
    size_t total_iter;
    size_t start_epoch, total_epoch;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, path;
    std::string dataroot, valid_dataroot;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, idx_t, idx_b, output;
    std::tuple<torch::Tensor, torch::Tensor> idx;
    datasets::ImageFolderWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderWithPaths dataloader, valid_dataloader;
    visualizer::graph train3_loss, valid3_loss;
    progress::display *show_progress;
    progress::irregular irreg_progress;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train3_dir"].as<std::string>();
    dataset = datasets::ImageFolderWithPaths(dataroot, transform);
    dataloader = DataLoader::ImageFolderWithPaths(dataset, vm["train3_batch_size"].as<size_t>(), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // (2) Get Validation Dataset
    if (vm["valid3"].as<bool>()){
        valid_dataroot = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["valid3_dir"].as<std::string>();
        valid_dataset = datasets::ImageFolderWithPaths(valid_dataroot, transform);
        valid_dataloader = DataLoader::ImageFolderWithPaths(valid_dataset, vm["valid3_batch_size"].as<size_t>(), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    // (3) Set Optimizer Method
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(vm["lr"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

    // (4) Set Loss Function
    auto criterion = Loss_PixelSnail(vm["K"].as<size_t>());

    // (5) Make Directories
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);

    // (6) Set Training Loss for Graph
    path = checkpoint_dir + "/graph";
    train3_loss = visualizer::graph(path, /*gname_=*/"train3_loss", /*label_=*/{"Index"});
    if (vm["valid3"].as<bool>()){
        valid3_loss = visualizer::graph(path, /*gname_=*/"valid3_loss", /*label_=*/{"Index"});
    }
    
    // (7) Get Weights and File Processing
    if (vm["train3_load_epoch"].as<std::string>() == ""){
        model->apply(weights_init);
        ofs.open(checkpoint_dir + "/log/train3.txt", std::ios::out);
        if (vm["valid3"].as<bool>()){
            init.open(checkpoint_dir + "/log/valid3.txt", std::ios::trunc);
            init.close();
        }
        start_epoch = 0;
    }
    else{
        path = checkpoint_dir + "/models/epoch_" + vm["train3_load_epoch"].as<std::string>() + "_pixelsnail_b.pth";  torch::load(model, path, device);
        path = checkpoint_dir + "/optims/epoch_" + vm["train3_load_epoch"].as<std::string>() + "_pixelsnail_b.pth";  torch::load(optimizer, path, device);
        ofs.open(checkpoint_dir + "/log/train3.txt", std::ios::app);
        ofs << std::endl << std::endl;
        if (vm["train3_load_epoch"].as<std::string>() == "latest"){
            infoi.open(checkpoint_dir + "/models/info.txt", std::ios::in);
            std::getline(infoi, buff);
            infoi.close();
            latest = "";
            for (auto &c : buff){
                if (('0' <= c) && (c <= '9')){
                    latest += c;
                }
            }
            start_epoch = std::stoi(latest);
        }
        else{
            start_epoch = std::stoi(vm["train3_load_epoch"].as<std::string>());
        }
    }

    // (8) Get model for training 1
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["train3_vqvae2_load_epoch"].as<std::string>() + "_vqvae2.pth";
    torch::load(vqvae2, path, device);
    vqvae2->eval();

    // (9) Display Date
    date = progress::current_date();
    date = progress::separator_center("Train Loss (" + date + ")");
    std::cout << std::endl << std::endl << date << std::endl;
    ofs << date << std::endl;


    // -----------------------------------
    // a2. Training Model
    // -----------------------------------
    
    // (1) Set Parameters
    start_epoch++;
    total_iter = dataloader.get_count_max();
    total_epoch = vm["train3_epochs"].as<size_t>();

    // (2) Training per Epoch
    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++){

        model->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{epoch, total_epoch}, /*loss_=*/{"index"});

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){

            image = std::get<0>(mini_batch).to(device);

            // -----------------------------------
            // c1. VQVAE2 Training Phase
            // -----------------------------------
            {
                torch::NoGradGuard no_grad;
                idx = vqvae2->forward_idx(image);
            }
            idx_t = std::get<0>(idx);
            idx_b = std::get<1>(idx);
            output = model->forward(idx_b, idx_t);
            loss = criterion(output, idx_b);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            show_progress->increment(/*loss_value=*/{loss.item<float>()});
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "index:" << loss.item<float>() << "(ave:" <<  show_progress->get_ave(0) << ')' << std::endl;

        }

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        train3_loss.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(0)});
        delete show_progress;
        
        // -----------------------------------
        // b4. Validation Mode
        // -----------------------------------
        if (vm["valid3"].as<bool>() && ((epoch - 1) % vm["valid3_freq"].as<size_t>() == 0)){
            valid3(vm, valid_dataloader, device, criterion, vqvae2, model, epoch, valid3_loss);
        }

        // -----------------------------------
        // b5. Save Model Weights and Optimizer Parameters
        // -----------------------------------
        if (epoch % vm["train3_save_epoch"].as<size_t>() == 0){
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_pixelsnail_b.pth";  torch::save(model, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_pixelsnail_b.pth";  torch::save(optimizer, path);
        }
        path = checkpoint_dir + "/models/epoch_latest_pixelsnail_b.pth";  torch::save(model, path);
        path = checkpoint_dir + "/optims/epoch_latest_pixelsnail_b.pth";  torch::save(optimizer, path);
        infoo.open(checkpoint_dir + "/models/info.txt", std::ios::out);
        infoo << "latest = " << epoch << std::endl;
        infoo.close();

        // -----------------------------------
        // b6. Show Elapsed Time
        // -----------------------------------
        if (epoch % 10 == 0){

            // -----------------------------------
            // c1. Get Output String
            // -----------------------------------
            ss.str(""); ss.clear(std::stringstream::goodbit);
            irreg_progress.nab(epoch);
            ss << "elapsed = " << irreg_progress.get_elap() << '(' << irreg_progress.get_sec_per() << "sec/epoch)   ";
            ss << "remaining = " << irreg_progress.get_rem() << "   ";
            ss << "now = " << irreg_progress.get_date() << "   ";
            ss << "finish = " << irreg_progress.get_date_fin();
            date_out = ss.str();

            // -----------------------------------
            // c2. Terminal Output
            // -----------------------------------
            std::cout << date_out << std::endl << progress::separator() << std::endl;
            ofs << date_out << std::endl << progress::separator() << std::endl;

        }

    }

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
