#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <random>                      // std::mt19937, std::uniform_int_distribution
#include <utility>                     // std::pair
#include <cstdlib>                     // std::rand
#include <cmath>                       // std::pow
// For External Library
#include <torch/torch.h>               // torch
#include <opencv2/opencv.hpp>          // cv::Mat
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // YOLOv8
#include "detector.hpp"                // YOLODetector
#include "augmentation.hpp"            // YOLOBatchAugmentation
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderBBWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderBBWithPaths
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace F = torch::nn::functional;
namespace po = boost::program_options;

// Function Prototype
void valid(po::variables_map &vm, DataLoader::ImageFolderBBWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, YOLOv8 &model, const std::vector<std::string> class_names, const size_t epoch, std::vector<visualizer::graph> &writer);


// -------------------
// Training Function
// -------------------
void train(po::variables_map &vm, torch::Device &device, YOLOv8 &model, std::vector<transforms_Compose> &transformBB, std::vector<transforms_Compose> &transformI, const std::vector<std::string> class_names){

    constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
    constexpr size_t train_workers = 4;  // the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 4;  // the number of workers to retrieve data from the validation dataset
    constexpr size_t save_sample_iter = 1000;  // the frequency of iteration to save sample images
    constexpr std::string_view extension = "jpg";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t epoch, iter;
    size_t total_iter;
    size_t start_epoch, total_epoch;
    float loss_f, loss_box_f, loss_obj_f, loss_class_f;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, save_images_dir, path;
    std::string input_dir, output_dir;
    std::string valid_input_dir, valid_output_dir;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> urand;
    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image;
    torch::Tensor loss_box, loss_obj, loss_class;
    std::vector<torch::Tensor> output, output_one;
    cv::Mat sample;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> losses;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_result;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;
    std::vector<transforms_Compose> null;
    datasets::ImageFolderBBWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderBBWithPaths dataloader, valid_dataloader;
    std::vector<visualizer::graph> train_loss;
    std::vector<visualizer::graph> valid_loss;
    progress::display *show_progress;
    progress::irregular irreg_progress;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    input_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_in_dir"].as<std::string>();
    output_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_out_dir"].as<std::string>();
    dataset = datasets::ImageFolderBBWithPaths(input_dir, output_dir, transformBB, transformI);
    dataloader = DataLoader::ImageFolderBBWithPaths(dataset, vm["batch_size"].as<size_t>(), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // (2) Get Validation Dataset
    if (vm["valid"].as<bool>()){
        valid_input_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["valid_in_dir"].as<std::string>();
        valid_output_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["valid_out_dir"].as<std::string>();
        valid_dataset = datasets::ImageFolderBBWithPaths(valid_input_dir, valid_output_dir, null, transformI);
        valid_dataloader = DataLoader::ImageFolderBBWithPaths(valid_dataset, vm["valid_batch_size"].as<size_t>(), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    // (3) Set Optimizer Method
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(vm["lr"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

    // (4) Set Loss Function
    auto criterion = Loss((long int)vm["class_num"].as<size_t>(), (long int)vm["reg_max"].as<size_t>());

    // (5) Set Augmentation and Detector
    auto augment = YOLOBatchAugmentation(vm["mosaic_rate"].as<double>(), vm["mixup_rate"].as<double>());
    auto detector = YOLODetector((long int)vm["class_num"].as<size_t>(), vm["prob_thresh"].as<float>(), vm["nms_thresh"].as<float>());
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette = detector.get_label_palette();

    // (6) Make Directories
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);
    save_images_dir = checkpoint_dir + "/samples";  fs::create_directories(save_images_dir);

    // (7) Set Training Loss for Graph
    path = checkpoint_dir + "/graph";
    train_loss = std::vector<visualizer::graph>(6);
    train_loss.at(0) = visualizer::graph(path, /*gname_=*/"train_loss_all", /*label_=*/{"Total"});
    train_loss.at(1) = visualizer::graph(path, /*gname_=*/"train_loss_box-IoU", /*label_=*/{"box-IoU"});
    train_loss.at(2) = visualizer::graph(path, /*gname_=*/"train_loss_object", /*label_=*/{"Object"});
    train_loss.at(3) = visualizer::graph(path, /*gname_=*/"train_loss_class", /*label_=*/{"Class"});
    if (vm["valid"].as<bool>()){
        valid_loss = std::vector<visualizer::graph>(6);
        valid_loss.at(0) = visualizer::graph(path, /*gname_=*/"valid_loss_all", /*label_=*/{"Total"});
        valid_loss.at(1) = visualizer::graph(path, /*gname_=*/"valid_loss_box-IoU", /*label_=*/{"box-IoU"});
        valid_loss.at(2) = visualizer::graph(path, /*gname_=*/"valid_loss_object", /*label_=*/{"Object"});
        valid_loss.at(3) = visualizer::graph(path, /*gname_=*/"valid_loss_class", /*label_=*/{"Class"});
    }
    
    // (8) Get Weights and File Processing
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
    mt.seed(std::rand());
    total_epoch = vm["epochs"].as<size_t>();

    // (2) Training per Epoch
    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++){

        model->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{epoch, total_epoch}, /*loss_=*/{"box", "obj", "class"});

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){

            image = std::get<0>(mini_batch).to(device);  // {N,C,H,W} (images)
            label = std::get<1>(mini_batch);  // {N, ({BB_n}, {BB_n,4}) } (annotations)
            if (vm["augmentation"].as<bool>()){
                std::tie(image, label) = augment.forward(image, label);
            }

            // -----------------------------------
            // c1. YOLOv8 Training Phase
            // -----------------------------------
            output = model->forward(image);  // {N,C,H,W} ===> {S,{N,G,G,A*(CN+5)}}
            losses = criterion(output, label);
            loss_box = std::get<0>(losses) * vm["Lambda_box"].as<float>();
            loss_obj = std::get<1>(losses) * vm["Lambda_obj"].as<float>();
            loss_class = std::get<2>(losses) * vm["Lambda_class"].as<float>();
            loss = loss_box + loss_obj + loss_class;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            show_progress->increment(/*loss_value=*/{loss_box.item<float>(), loss_obj.item<float>(), loss_class.item<float>()});
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "box:" << loss_box.item<float>() << "(ave:" <<  show_progress->get_ave(0) << ") " << std::flush;
            ofs << "obj:" << loss_obj.item<float>() << "(ave:" <<  show_progress->get_ave(1) << ") " << std::flush;
            ofs << "class:" << loss_class.item<float>() << "(ave:" <<  show_progress->get_ave(2) << ")" << std::endl;

            // -----------------------------------
            // c3. Save Sample Images
            // -----------------------------------
            iter = show_progress->get_iters();
            if (iter % save_sample_iter == 1){
                ss.str(""); ss.clear(std::stringstream::goodbit);
                ss << save_images_dir << "/epoch_" << epoch << "-iter_" << iter << '.' << extension;
                /*************************************************************************/
                output_one = std::vector<torch::Tensor>(output.size());
                for (size_t i = 0; i < output_one.size(); i++){
                    output_one.at(i) = output.at(i)[0];
                }
                detect_result = detector(output_one);
                /*************************************************************************/
                sample = visualizer::draw_detections_des(image[0].detach(), {std::get<0>(detect_result), std::get<1>(detect_result)}, std::get<2>(detect_result), class_names, label_palette, /*range=*/output_range);
                cv::imwrite(ss.str(), sample);
            }

        }

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        loss_f = show_progress->get_ave(0) + show_progress->get_ave(1) + show_progress->get_ave(2);
        loss_box_f = show_progress->get_ave(0);
        loss_obj_f = show_progress->get_ave(1);
        loss_class_f = show_progress->get_ave(2);
        train_loss.at(0).plot(/*base=*/epoch, /*value=*/{loss_f});
        train_loss.at(1).plot(/*base=*/epoch, /*value=*/{loss_box_f});
        train_loss.at(2).plot(/*base=*/epoch, /*value=*/{loss_obj_f});
        train_loss.at(3).plot(/*base=*/epoch, /*value=*/{loss_class_f});

        // -----------------------------------
        // b3. Save Sample Images
        // -----------------------------------
        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        /*************************************************************************/
        output_one = std::vector<torch::Tensor>(output.size());
        for (size_t i = 0; i < output_one.size(); i++){
            output_one.at(i) = output.at(i)[0];
        }
        detect_result = detector(output_one);
        /*************************************************************************/
        sample = visualizer::draw_detections_des(image[0].detach(), {std::get<0>(detect_result), std::get<1>(detect_result)}, std::get<2>(detect_result), class_names, label_palette, /*range=*/output_range);
        cv::imwrite(ss.str(), sample);
        delete show_progress;
        
        // -----------------------------------
        // b4. Validation Mode
        // -----------------------------------
        if (vm["valid"].as<bool>() && ((epoch - 1) % vm["valid_freq"].as<size_t>() == 0)){
            valid(vm, valid_dataloader, device, criterion, model, class_names, epoch, valid_loss);
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

