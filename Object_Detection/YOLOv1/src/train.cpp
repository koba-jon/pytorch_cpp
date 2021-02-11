#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <utility>                     // std::pair
#include <cmath>                       // std::ceil, std::pow
// For External Library
#include <torch/torch.h>               // torch
#include <opencv2/opencv.hpp>          // cv::Mat
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // YOLOv1
#include "detector.hpp"                // YOLODetector
#include "transforms.hpp"              // transforms::Compose
#include "datasets.hpp"                // datasets::ImageFolderBBWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderBBWithPaths
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
template <typename Optimizer, typename OptimizerOptions> void Update_LR(Optimizer &optimizer, const float lr_init, const float lr_base, const float lr_decay1, const float lr_decay2, const size_t epoch, const float burnin_base, const float burnin_exp=4.0);
void valid(po::variables_map &vm, DataLoader::ImageFolderBBWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, YOLOv1 &model, const std::vector<std::string> class_names, const size_t epoch, visualizer::graph &writer);


// -------------------
// Training Function
// -------------------
void train(po::variables_map &vm, torch::Device &device, YOLOv1 &model, std::vector<transforms::Compose*> &transformBB, std::vector<transforms::Compose*> &transformI, const std::vector<std::string> class_names){

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
    float loss_f, loss_coord_xy_f, loss_coord_wh_f, loss_obj_f, loss_noobj_f, loss_class_f;
    float lr_init, lr_base, lr_decay1, lr_decay2;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, save_images_dir, path;
    std::string input_dir, output_dir;
    std::string valid_input_dir, valid_output_dir;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, output;
    torch::Tensor loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class;
    cv::Mat sample;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> losses;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_result;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;
    std::vector<transforms::Compose*> null;
    datasets::ImageFolderBBWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderBBWithPaths dataloader, valid_dataloader;
    visualizer::graph train_loss;
    visualizer::graph valid_loss;
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
    using Optimizer = torch::optim::SGD;
    using OptimizerOptions = torch::optim::SGDOptions;
    auto optimizer = Optimizer(model->parameters(), OptimizerOptions(vm["lr_init"].as<float>()).momentum(vm["momentum"].as<float>()).weight_decay(vm["weight_decay"].as<float>()));

    // (4) Set Loss Function
    auto criterion = Loss((long int)vm["class_num"].as<size_t>(), (long int)vm["ng"].as<size_t>(), (long int)vm["nb"].as<size_t>());

    // (5) Set Detector
    auto detector = YOLODetector((long int)vm["class_num"].as<size_t>(), (long int)vm["ng"].as<size_t>(), (long int)vm["nb"].as<size_t>(), vm["prob_thresh"].as<float>(), vm["nms_thresh"].as<float>());
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette = detector.get_label_palette();

    // (6) Make Directories
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);
    save_images_dir = checkpoint_dir + "/samples";  fs::create_directories(save_images_dir);

    // (7) Set Training Loss for Graph
    path = checkpoint_dir + "/graph";
    train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{"Total", "Coordinate(center)", "Coordinate(range)", "Confidence(object)", "Confidence(no-object)", "Class"});
    if (vm["valid"].as<bool>()){
        valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{"Total", "Coordinate(center)", "Coordinate(range)", "Confidence(object)", "Confidence(no-object)", "Class"});
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
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + ".pth";  torch::load(model, path);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + ".pth";  torch::load(optimizer, path);
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
    total_iter = std::ceil((float)dataset.size() / (float)vm["batch_size"].as<size_t>());
    total_epoch = vm["epochs"].as<size_t>();
    lr_init = vm["lr_init"].as<float>();
    lr_base = vm["lr_base"].as<float>();
    lr_decay1 = vm["lr_decay1"].as<float>();
    lr_decay2 = vm["lr_decay2"].as<float>();

    // (2) Training per Epoch
    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++){

        model->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{epoch, total_epoch}, /*loss_=*/{"coord_xy", "coord_wh", "conf_o", "conf_x", "class"});

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){

            // -----------------------------------
            // c1. Update Learning Rate
            // -----------------------------------
            Update_LR<Optimizer, OptimizerOptions>(optimizer, lr_init, lr_base, lr_decay1, lr_decay2, epoch, (float)show_progress->get_iters() / (float)total_iter);

            // -----------------------------------
            // c2. YOLOv1 Training Phase
            // -----------------------------------
            image = std::get<0>(mini_batch).to(device);  // {N,C,H,W} (images)
            label = std::get<1>(mini_batch);  // {N, ({BB_n}, {BB_n,4}) } (annotations)
            output = model->forward(image);  // {N,C,H,W} ===> {N,G,G,FF}
            losses = criterion(output, label);
            loss_coord_xy = std::get<0>(losses) * vm["Lambda_coord"].as<float>();
            loss_coord_wh = std::get<1>(losses) * vm["Lambda_coord"].as<float>();
            loss_obj = std::get<2>(losses) * vm["Lambda_object"].as<float>();
            loss_noobj = std::get<3>(losses) * vm["Lambda_noobject"].as<float>();
            loss_class = std::get<4>(losses) * vm["Lambda_class"].as<float>();
            loss = loss_coord_xy + loss_coord_wh + loss_obj + loss_noobj + loss_class;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // -----------------------------------
            // c3. Record Loss (iteration)
            // -----------------------------------
            show_progress->increment(/*loss_value=*/{loss_coord_xy.item<float>(), loss_coord_wh.item<float>(), loss_obj.item<float>(), loss_noobj.item<float>(), loss_class.item<float>()});
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "coord_xy:" << loss_coord_xy.item<float>() << "(ave:" <<  show_progress->get_ave(0) << ") " << std::flush;
            ofs << "coord_wh:" << loss_coord_wh.item<float>() << "(ave:" <<  show_progress->get_ave(1) << ") " << std::flush;
            ofs << "conf_o:" << loss_obj.item<float>() << "(ave:" <<  show_progress->get_ave(2) << ") " << std::flush;
            ofs << "conf_x:" << loss_noobj.item<float>() << "(ave:" <<  show_progress->get_ave(3) << ") " << std::flush;
            ofs << "class:" << loss_class.item<float>() << "(ave:" <<  show_progress->get_ave(4) << ')' << std::endl;

            // -----------------------------------
            // c4. Save Sample Images
            // -----------------------------------
            iter = show_progress->get_iters();
            if (iter % save_sample_iter == 1){
                ss.str(""); ss.clear(std::stringstream::goodbit);
                ss << save_images_dir << "/epoch_" << epoch << "-iter_" << iter << '.' << extension;
                detect_result = detector(output[0]);
                sample = visualizer::draw_detections(image[0].detach(), {std::get<0>(detect_result), std::get<1>(detect_result)}, class_names, label_palette, /*range=*/output_range);
                cv::imwrite(ss.str(), sample);
            }

        }

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        loss_f = show_progress->get_ave(0) + show_progress->get_ave(1) + show_progress->get_ave(2) + show_progress->get_ave(3) + show_progress->get_ave(4);
        loss_coord_xy_f = show_progress->get_ave(0);
        loss_coord_wh_f = show_progress->get_ave(1);
        loss_obj_f = show_progress->get_ave(2);
        loss_noobj_f = show_progress->get_ave(3);
        loss_class_f = show_progress->get_ave(4);
        train_loss.plot(/*base=*/epoch, /*value=*/{loss_f, loss_coord_xy_f, loss_coord_wh_f, loss_obj_f, loss_noobj_f, loss_class_f});

        // -----------------------------------
        // b3. Save Sample Images
        // -----------------------------------
        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        detect_result = detector(output[0]);
        sample = visualizer::draw_detections(image[0].detach(), {std::get<0>(detect_result), std::get<1>(detect_result)}, class_names, label_palette, /*range=*/output_range);
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


// ------------------------------------
// Function to Update Learning Rate
// ------------------------------------
template <typename Optimizer, typename OptimizerOptions>
void Update_LR(Optimizer &optimizer, const float lr_init, const float lr_base, const float lr_decay1, const float lr_decay2, const size_t epoch, const float burnin_base, const float burnin_exp){

    float lr;

    if (epoch == 1){
        lr = lr_init + (lr_base - lr_init) * std::pow(burnin_base, burnin_exp);
    }
    else if(epoch == 2){
        lr = lr_base;
    }
    else if(epoch == 76){
        lr = lr_decay1;
    }
    else if(epoch == 106){
        lr = lr_decay2;
    }
    else{
        return;
    }

    for (auto &param_group : optimizer.param_groups()){
        if (param_group.has_options()){
            auto &options = (OptimizerOptions&)(param_group.options());
            options.lr(lr);
        }
    }

    return;

}