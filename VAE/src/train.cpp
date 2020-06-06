#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <chrono>                      // std::chrono
#include <algorithm>                   // std::find
#include <utility>                     // std::pair
#include <ios>                         // std::right
#include <iomanip>                     // std::setw, std::setfill
#include <cmath>                       // std::ceil
#include <ctime>                       // std::time_t, std::ctime
#include <sys/stat.h>                  // mkdir
#include <sys/ioctl.h>                 // ioctl, TIOCGWINSZ
#include <unistd.h>                    // STDOUT_FILENO
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // VariationalAutoEncoder
#include "transforms.hpp"              // transforms::Compose
#include "datasets.hpp"                // datasets::ImageFolderWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress_display

// Define Namespace
namespace po = boost::program_options;

// Function Prototype
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, VariationalAutoEncoder &model, const size_t epoch, visualizer::graph &writer);


// -------------------
// Training Function
// -------------------
void train(po::variables_map &vm, torch::Device &device, VariationalAutoEncoder &model, std::vector<transforms::Compose*> &transform){

    constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
    constexpr size_t train_workers = 4;  // number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 4;  // number of workers to retrieve data from the validation dataset
    constexpr size_t save_sample_iter = 50;  // the frequency of iteration to save sample images
    constexpr std::string_view extension = "jpg";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t i;
    size_t epoch, iter;
    size_t total_iter;
    size_t start_epoch, total_epoch;
    size_t length, both_width;
    int elap_hour, elap_min, elap_sec, rem_times, rem_hour, rem_min, rem_sec;
    double sec_per_epoch;
    struct winsize ws;
    std::time_t time_now, time_fin;
    std::string elap_hour_str, elap_min_str, elap_sec_str, sec_per_epoch_str, rem_hour_str, rem_min_str, rem_sec_str;
    std::string date, date_fin, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, save_images_dir, path;
    std::string dataroot, valid_dataroot;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor rec, kld, loss, image, output;
    datasets::ImageFolderWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderWithPaths dataloader, valid_dataloader;
    visualizer::graph train_loss, valid_loss;
    progress_display *show_progress;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    dataroot = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_dir"].as<std::string>();
    dataset = datasets::ImageFolderWithPaths(dataroot, transform);
    dataloader = DataLoader::ImageFolderWithPaths(dataset, vm["batch_size"].as<size_t>(), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // (2) Get Validation Dataset
    if (vm["valid"].as<bool>()){
        valid_dataroot = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["valid_dir"].as<std::string>();
        valid_dataset = datasets::ImageFolderWithPaths(valid_dataroot, transform);
        valid_dataloader = DataLoader::ImageFolderWithPaths(valid_dataset, vm["valid_batch_size"].as<size_t>(), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    // (3) Set Optimizer Method
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(vm["lr"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

    // (4) Set Loss Function
    auto criterion = Loss(vm["loss"].as<std::string>());

    // (5) Make Directories
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/models";  mkdir(path.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
    path = checkpoint_dir + "/optims";  mkdir(path.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
    path = checkpoint_dir + "/log";  mkdir(path.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
    save_images_dir = checkpoint_dir + "/samples";  mkdir(save_images_dir.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);

    // (6) Set Training Loss for Graph
    path = checkpoint_dir + "/graph";
    train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{"Total", "Reconstruct", "KL-divergence"});
    if (vm["valid"].as<bool>()){
        valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{"Total", "Reconstruct", "KL-divergence"});
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

    // (8) Catch Terminal Size
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != -1){
       length = ws.ws_col - 1;
    }
    else{
        std::cerr << "Error : Couldn't get the width of terminal" << std::endl;
        std::exit(1);
    }

    // (9) Display Date
    auto now = std::chrono::system_clock::now();
    time_now = std::chrono::system_clock::to_time_t(now);
    ss.str(""); ss.clear(std::stringstream::goodbit);
    ss << std::ctime(&time_now);
    date = ss.str();
    date.erase(std::find(date.begin(), date.end(), '\n'));
    ss.str(""); ss.clear(std::stringstream::goodbit);
    ss << " Train Loss (" << date << ") ";
    both_width = length - ss.str().length();
    std::cout << std::endl << std::endl << std::string(both_width/2, '-') << ss.str() << std::string(both_width/2, '-') << std::endl;
    ofs << std::string(both_width/2, '-') << ss.str() << std::string(both_width/2, '-') << std::endl;


    // -----------------------------------
    // a2. Training Model
    // -----------------------------------
    
    // (1) Set Parameters
    start_epoch++;
    total_iter = std::ceil((float)dataset.size() / (float)vm["batch_size"].as<size_t>());
    total_epoch = vm["epochs"].as<size_t>();

    // (2) Training per Epoch
    auto time_start = std::chrono::system_clock::now();
    for (epoch = start_epoch; epoch <= total_epoch; epoch++){

        model->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress_display(/*count_max_=*/total_iter, /*epoch=*/{epoch, total_epoch}, /*loss_=*/{"rec", "kld"});

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){

            // -----------------------------------
            // c1. Variational Auto Encoder Training Phase
            // -----------------------------------
            image = std::get<0>(mini_batch).to(device);
            output = model->forward(image);
            rec = criterion(output, image);
            kld = vm["Lambda"].as<float>() * model->kld_just_before();
            loss = rec + kld;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            show_progress->increment(/*loss_value=*/{rec.item<float>(), kld.item<float>()});
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "rec:" << rec.item<float>() << "(ave:" <<  show_progress->get_ave(0) << ") " << std::flush;
            ofs << "kld:" << kld.item<float>() << "(ave:" <<  show_progress->get_ave(1) << ')' << std::endl;

            // -----------------------------------
            // c3. Save Sample Images
            // -----------------------------------
            iter = show_progress->get_iters();
            if (iter % save_sample_iter == 1){
                ss.str(""); ss.clear(std::stringstream::goodbit);
                ss << save_images_dir << "/epoch_" << epoch << "-iter_" << iter << '.' << extension;
                visualizer::save_image(output.detach(), ss.str(), /*range=*/output_range);
            }

        }

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        train_loss.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(0) + show_progress->get_ave(1), show_progress->get_ave(0), show_progress->get_ave(1)});

        // -----------------------------------
        // b3. Save Sample Images
        // -----------------------------------
        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        visualizer::save_image(output.detach(), ss.str(), /*range=*/output_range);
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
            auto time_end = std::chrono::system_clock::now();
            for (i = 0; i < 8; i++){
                ss.str(""); ss.clear(std::stringstream::goodbit);
                switch (i){
                    case 0:
                        elap_hour = (int)std::chrono::duration_cast<std::chrono::hours>(time_end - time_start).count();
                        ss << std::setfill('0') << std::right << std::setw(2) << elap_hour;
                        elap_hour_str = ss.str();
                        break;
                    case 1:
                        elap_min = (int)std::chrono::duration_cast<std::chrono::minutes>(time_end - time_start).count() % 60;
                        ss << std::setfill('0') << std::right << std::setw(2) << elap_min;
                        elap_min_str = ss.str();
                        break;
                    case 2:
                        elap_sec = (int)std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start).count() % 60;
                        ss << std::setfill('0') << std::right << std::setw(2) << elap_sec;
                        elap_sec_str = ss.str();
                        break;
                    case 3:
                        sec_per_epoch = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() * 0.001 / (double)(epoch - start_epoch + 1);
                        ss << std::setprecision(5) << sec_per_epoch;
                        sec_per_epoch_str = ss.str();
                        break;
                    case 4:
                        rem_times = (int)(sec_per_epoch * (double)(total_epoch - epoch));
                        break;
                    case 5:
                        rem_hour = rem_times / 3600;
                        ss << std::setfill('0') << std::right << std::setw(2) << rem_hour;
                        rem_hour_str = ss.str();
                        break;
                    case 6:
                        rem_min = (rem_times / 60) % 60;
                        ss << std::setfill('0') << std::right << std::setw(2) << rem_min;
                        rem_min_str = ss.str();
                        break;
                    case 7:
                        rem_sec = rem_times % 60;
                        ss << std::setfill('0') << std::right << std::setw(2) << rem_sec;
                        rem_sec_str = ss.str();
                        break;
                    default:
                        std::cerr << "Error : There is an unexpected value in argument of 'switch'." << std::endl;
                        std::exit(1);
                }
            }

            // -----------------------------------
            // c1. Get Current Date
            // -----------------------------------
            time_now = std::chrono::system_clock::to_time_t(time_end);
            ss.str(""); ss.clear(std::stringstream::goodbit);
            ss << std::ctime(&time_now);
            date = ss.str();
            date.erase(std::find(date.begin(), date.end(), '\n'));

            // -----------------------------------
            // c2. Get Finish Date
            // -----------------------------------
            time_fin = time_now + (time_t)rem_times;
            ss.str(""); ss.clear(std::stringstream::goodbit);
            ss << std::ctime(&time_fin);
            date_fin = ss.str();
            date_fin.erase(std::find(date_fin.begin(), date_fin.end(), '\n'));

            // -----------------------------------
            // c3. Get Output String
            // -----------------------------------
            ss.str(""); ss.clear(std::stringstream::goodbit);
            ss << "elapsed = " << elap_hour_str << ':' << elap_min_str << ':' << elap_sec_str << '(' << sec_per_epoch_str << "sec/epoch)   ";
            ss << "remaining = " << rem_hour_str << ':' << rem_min_str << ':' << rem_sec_str << "   ";
            ss << "now = " << date << "   ";
            ss << "finish = " << date_fin;
            date_out = ss.str();

            // -----------------------------------
            // c4. Terminal Output
            // -----------------------------------
            // (1) Catch Terminal Size
            if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != -1){
               length = ws.ws_col - 1;
            }
            else{
                std::cerr << "Error : Couldn't get the width of terminal" << std::endl;
                std::exit(1);
            }
            // (2) Times and Dates Output
            std::cout << date_out << std::endl << std::string(length, '-') << std::endl;
            ofs << date_out << std::endl << std::string(length, '-') << std::endl;

        }

    }

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
