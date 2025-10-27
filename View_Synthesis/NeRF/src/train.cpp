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
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // NeRF
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderCameraPoseWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderCameraPoseWithPaths
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void valid(po::variables_map &vm, DataLoader::ImageFolderCameraPoseWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, NeRF &model, const size_t epoch, visualizer::graph &writer);


// -------------------
// Training Function
// -------------------
void train(po::variables_map &vm, torch::Device &device, NeRF &model, std::vector<transforms_Compose> &transform){

    constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
    constexpr size_t train_workers = 4;  // the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 4;  // the number of workers to retrieve data from the validation dataset
    constexpr size_t save_sample_iter = 50;  // the frequency of iteration to save sample images
    constexpr std::string_view extension = "jpg";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t epoch, iter;
    size_t total_iter;
    size_t start_epoch, total_epoch;
    long int mini_batch_size;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, save_images_dir, path;
    std::string image_dir, pose_dir, valid_image_dir, valid_pose_dir;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor image, pose, rays_o, rays_d, target_rgb, rgb_fine, rgb_coarse, sample_idx, sample_idx_expand;
    torch::Tensor loss, loss_coarse, loss_fine;
    torch::Tensor pose_example, recon_image, gt_image, pair;
    datasets::ImageFolderCameraPoseWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderCameraPoseWithPaths dataloader, valid_dataloader;
    visualizer::graph train_loss, valid_loss;
    progress::display *show_progress;
    progress::irregular irreg_progress;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    image_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_image_dir"].as<std::string>();
    pose_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_pose_dir"].as<std::string>();
    dataset = datasets::ImageFolderCameraPoseWithPaths(image_dir, pose_dir, transform);
    dataloader = DataLoader::ImageFolderCameraPoseWithPaths(dataset, vm["batch_size"].as<size_t>(), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // (2) Get Validation Dataset
    if (vm["valid"].as<bool>()){
        valid_image_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["valid_image_dir"].as<std::string>();
        valid_pose_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["valid_pose_dir"].as<std::string>();
        valid_dataset = datasets::ImageFolderCameraPoseWithPaths(valid_image_dir, valid_pose_dir, transform);
        valid_dataloader = DataLoader::ImageFolderCameraPoseWithPaths(valid_dataset, vm["valid_batch_size"].as<size_t>(), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    // (3) Set Optimizer Method
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(vm["lr"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

    // (4) Set Loss Function
    auto criterion = Loss(vm["loss"].as<std::string>());

    // (5) Make Directories
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);
    save_images_dir = checkpoint_dir + "/samples";  fs::create_directories(save_images_dir);

    // (6) Set Training Loss for Graph
    path = checkpoint_dir + "/graph";
    train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{"Reconstruct"});
    if (vm["valid"].as<bool>()){
        valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{"Reconstruct"});
    }
    
    // (7) Get Weights and File Processing
    if (vm["train_load_epoch"].as<std::string>() == ""){
        model->apply(weights_init);
        ofs.open(checkpoint_dir + "/log/train.txt", std::ios::out);
        if (vm["valid"].as<bool>()){
            init.open(checkpoint_dir + "/log/valid.txt", std::ios::trunc);
            init.close();
        }
        start_epoch = 0;
    }
    else{
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + ".pth";  torch::load(model, path, device);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + ".pth";  torch::load(optimizer, path, device);
        ofs.open(checkpoint_dir + "/log/train.txt", std::ios::app);
        ofs << std::endl << std::endl;
        if (vm["train_load_epoch"].as<std::string>() == "latest"){
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
            start_epoch = std::stoi(vm["train_load_epoch"].as<std::string>());
        }
    }

    // (8) Display Date
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
    total_epoch = vm["epochs"].as<size_t>();

    // (2) Training per Epoch
    mini_batch_size = 0;
    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++){

        model->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{epoch, total_epoch}, /*loss_=*/{"loss"});

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){

            image = std::get<0>(mini_batch).to(device);
            pose = std::get<1>(mini_batch).to(device);
            mini_batch_size = image.size(0);

            std::tie(rays_o, rays_d) = model->build_rays(pose);
            target_rgb = image.permute({0, 2, 3, 1}).view({mini_batch_size, -1, 3}).contiguous();
            if ((size_t)target_rgb.size(1) > vm["rays_per_image"].as<size_t>()){
                sample_idx = torch::randint(0, target_rgb.size(1), {mini_batch_size, (long int)vm["rays_per_image"].as<size_t>()}, torch::kLong).to(device);
                sample_idx_expand = sample_idx.unsqueeze(-1).expand({mini_batch_size, (long int)vm["rays_per_image"].as<size_t>(), 3});
                rays_o = torch::gather(rays_o, 1, sample_idx_expand);
                rays_d = torch::gather(rays_d, 1, sample_idx_expand);
                target_rgb = torch::gather(target_rgb, 1, sample_idx_expand);
            }

            // -------------------------
            // c1. NeRF Training Phase
            // -------------------------
            std::tie(rgb_fine, rgb_coarse) = model->forward(rays_o, rays_d);
            loss_fine = criterion(rgb_fine, target_rgb);
            loss_coarse = criterion(rgb_coarse, target_rgb);
            loss = loss_fine + loss_coarse;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            show_progress->increment(/*loss_value=*/{loss.item<float>()});
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "loss:" << loss.item<float>() << "(ave:" <<  show_progress->get_ave(0) << ')' << std::endl;

            // -----------------------------------
            // c3. Save Sample Images
            // -----------------------------------
            iter = show_progress->get_iters();
            if (iter % save_sample_iter == 1){
                ss.str(""); ss.clear(std::stringstream::goodbit);
                ss << save_images_dir << "/epoch_" << epoch << "-iter_" << iter << '.' << extension;
                pose_example = pose.index({0}).unsqueeze(0);
                recon_image = model->render_image(pose_example);
                gt_image = image.index({0}).unsqueeze(0);
                pair = torch::cat({gt_image, recon_image}, /*dim=*/0);
                visualizer::save_image(pair.detach(), ss.str(), /*range=*/output_range, /*cols=*/mini_batch_size);
            }

        }

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        train_loss.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(0)});

        // -----------------------------------
        // b3. Save Sample Images
        // -----------------------------------
        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        pose_example = pose.index({0}).unsqueeze(0);
        recon_image = model->render_image(pose_example);
        gt_image = image.index({0}).unsqueeze(0);
        pair = torch::cat({gt_image, recon_image}, /*dim=*/0);
        visualizer::save_image(pair.detach(), ss.str(), /*range=*/output_range, /*cols=*/mini_batch_size);
        delete show_progress;
        
        // -----------------------------------
        // b4. Validation Mode
        // -----------------------------------
        if (vm["valid"].as<bool>() && ((epoch - 1) % vm["valid_freq"].as<size_t>() == 0)){
            valid(vm, valid_dataloader, device, criterion, model, epoch, valid_loss);
        }

        // -----------------------------------
        // b5. Save Model Weights and Optimizer Parameters
        // -----------------------------------
        if (epoch % vm["save_epoch"].as<size_t>() == 0){
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + ".pth";  torch::save(model, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + ".pth";  torch::save(optimizer, path);
        }
        path = checkpoint_dir + "/models/epoch_latest.pth";  torch::save(model, path);
        path = checkpoint_dir + "/optims/epoch_latest.pth";  torch::save(optimizer, path);
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

