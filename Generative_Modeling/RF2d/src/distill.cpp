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
#include "networks.hpp"                // RF
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;


// -------------------
// Distillation Function
// -------------------
void distill(po::variables_map &vm, torch::Device &device, RF &model){

    constexpr bool distill_shuffle = true;  // whether to shuffle the distillation dataset
    constexpr size_t distill_workers = 4;  // the number of workers to retrieve data from the distillation dataset
    constexpr size_t save_sample_iter = 1000;  // the frequency of iteration to save sample images
    constexpr std::string_view extension = "jpg";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

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
    std::string reflowI_dir, reflowO_dir;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor t, z, x_hat, v, loss, output, recon, pair;
    datasets::ImageTensorFolderPairWithPaths dataset;
    DataLoader::ImageTensorFolderPairWithPaths dataloader;
    visualizer::graph distill_loss;
    progress::display *show_progress;
    progress::irregular irreg_progress;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Distillation Dataset
    reflowI_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["reflow_in_dir"].as<std::string>();  fs::create_directories(reflowI_dir);
    reflowO_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["reflow_out_dir"].as<std::string>();  fs::create_directories(reflowO_dir);
    dataset = datasets::ImageTensorFolderPairWithPaths(reflowI_dir, reflowO_dir);
    dataloader = DataLoader::ImageTensorFolderPairWithPaths(dataset, vm["distill_batch_size"].as<size_t>(), /*shuffle_=*/distill_shuffle, /*num_workers_=*/distill_workers);
    std::cout << "total distillation images : " << dataset.size() << std::endl;

    // (2) Set Optimizer Method
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(vm["lr"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

    // (3) Set Loss Function
    auto criterion = Loss(vm["loss"].as<std::string>());

    // (4) Make Directories
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);
    save_images_dir = checkpoint_dir + "/distill_samples";  fs::create_directories(save_images_dir);

    // (5) Set Distillation Loss for Graph
    path = checkpoint_dir + "/graph";
    distill_loss = visualizer::graph(path, /*gname_=*/"distill_loss", /*label_=*/{"Reconstruct"});
    
    // (6) Get Weights and File Processing
    if (vm["distill_load_epoch"].as<std::string>() == ""){
        model->apply(weights_init);
        ofs.open(checkpoint_dir + "/log/distill.txt", std::ios::out);
        start_epoch = 0;
    }
    else{
        path = checkpoint_dir + "/models/epoch_" + vm["distill_load_epoch"].as<std::string>() + "_distill.pth";  torch::load(model, path, device);
        path = checkpoint_dir + "/optims/epoch_" + vm["distill_load_epoch"].as<std::string>() + "_distill.pth";  torch::load(optimizer, path, device);
        ofs.open(checkpoint_dir + "/log/distill.txt", std::ios::app);
        ofs << std::endl << std::endl;
        if (vm["distill_load_epoch"].as<std::string>() == "latest"){
            infoi.open(checkpoint_dir + "/models/distill_info.txt", std::ios::in);
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
            start_epoch = std::stoi(vm["distill_load_epoch"].as<std::string>());
        }
    }

    // (7) Display Date
    date = progress::current_date();
    date = progress::separator_center("Distillation Loss (" + date + ")");
    std::cout << std::endl << std::endl << date << std::endl;
    ofs << date << std::endl;


    // -----------------------------------
    // a2. Distillation Model
    // -----------------------------------
    
    // (1) Set Parameters
    start_epoch++;
    total_iter = dataloader.get_count_max();
    total_epoch = vm["distill_epochs"].as<size_t>();

    // (2) Distillation per Epoch
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

            z = std::get<0>(mini_batch).to(device);
            x_hat = std::get<1>(mini_batch).to(device);
            v = x_hat - z;
            mini_batch_size = z.size(0);

            // -------------------------
            // c1. RF Distillation Phase
            // -------------------------
            t = torch::zeros({mini_batch_size}).to(device);
            output = model->forward(z, t);
            loss = criterion(output, v);
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
                recon = z + output;
                pair = torch::cat({z, recon, x_hat}, /*dim=*/0);
                visualizer::save_image(pair.detach(), ss.str(), /*range=*/output_range, /*cols=*/mini_batch_size);
            }

        }

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        distill_loss.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(0)});

        // -----------------------------------
        // b3. Save Sample Images
        // -----------------------------------
        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        recon = z + output;
        pair = torch::cat({z, recon, x_hat}, /*dim=*/0);
        visualizer::save_image(pair.detach(), ss.str(), /*range=*/output_range, /*cols=*/mini_batch_size);
        delete show_progress;
        
        // -----------------------------------
        // b4. Save Model Weights and Optimizer Parameters
        // -----------------------------------
        if (epoch % vm["distill_save_epoch"].as<size_t>() == 0){
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_distill.pth";  torch::save(model, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_distill.pth";  torch::save(optimizer, path);
        }
        path = checkpoint_dir + "/models/epoch_latest_distill.pth";  torch::save(model, path);
        path = checkpoint_dir + "/optims/epoch_latest_distill.pth";  torch::save(optimizer, path);
        infoo.open(checkpoint_dir + "/models/distill_info.txt", std::ios::out);
        infoo << "latest = " << epoch << std::endl;
        infoo.close();

        // -----------------------------------
        // b5. Show Elapsed Time
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
