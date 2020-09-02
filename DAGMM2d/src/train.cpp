#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <chrono>                      // std::chrono
#include <algorithm>                   // std::find
#include <utility>                     // std::pair
#include <cstdlib>                     // std::exit
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
#include "networks.hpp"                // Encoder, Decoder, EstimationNetwork, RelativeEuclideanDistance, CosineSimilarity, save_params
#include "transforms.hpp"              // transforms::Compose
#include "datasets.hpp"                // datasets::ImageFolderWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress::display, progress::irregular

// Define Namespace
namespace po = boost::program_options;

// Function Prototype
void get_gmp(po::variables_map &vm, DataLoader::ImageFolderWithPaths &dataloader, torch::Device &device, Encoder &enc, Decoder &dec, EstimationNetwork &est, size_t total_iter, torch::Tensor &mu, torch::Tensor &sigma, torch::Tensor &phi);
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, Encoder &enc, Decoder &dec, EstimationNetwork &est, torch::Tensor mu, torch::Tensor sigma, torch::Tensor phi, const size_t epoch, visualizer::graph &writer, visualizer::graph &writer_rec);


// -------------------
// Training Function
// -------------------
void train(po::variables_map &vm, torch::Device &device, Encoder &enc, Decoder &dec, EstimationNetwork &est, std::vector<transforms::Compose*> &transform){

    constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
    constexpr size_t train_workers = 4;  // the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 4;  // the number of workers to retrieve data from the validation dataset
    constexpr size_t save_sample_iter = 50;  // the frequency of iteration to save sample images
    constexpr std::string_view extension = "jpg";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t epoch, iter;
    size_t total_iter;
    size_t start_epoch, total_epoch;
    size_t length, both_width;
    struct winsize ws;
    std::time_t time_now;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, save_images_dir, path;
    std::string dataroot, valid_dataroot;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, output;
    torch::Tensor mu, sigma, phi;
    torch::Tensor z, z_c, z_r, z_r1, z_r2, gamma_ap;
    torch::Tensor rec, energy, penalty, loss;
    datasets::ImageFolderWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderWithPaths dataloader, valid_dataloader;
    visualizer::graph train_loss, train_rec_loss;
    visualizer::graph valid_loss, valid_rec_loss;
    progress::display *show_progress;
    progress::irregular irreg_progress;


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
    auto enc_optimizer = torch::optim::Adam(enc->parameters(), torch::optim::AdamOptions(vm["lr_com"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));
    auto dec_optimizer = torch::optim::Adam(dec->parameters(), torch::optim::AdamOptions(vm["lr_com"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));
    auto est_optimizer = torch::optim::Adam(est->parameters(), torch::optim::AdamOptions(vm["lr_est"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

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
    train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{"Total", "Reconstruct", "Energy", "Penalty"});
    train_rec_loss = visualizer::graph(path, /*gname_=*/"train_rec_loss", /*label_=*/{vm["loss"].as<std::string>()});
    if (vm["valid"].as<bool>()){
        valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{"Total", "Reconstruct", "Anomaly Score"});
        valid_rec_loss = visualizer::graph(path, /*gname_=*/"valid_rec_loss", /*label_=*/{vm["loss"].as<std::string>()});
    }
    
    // (7) Get Weights and File Processing
    if (vm["train_load_epoch"].as<std::string>() == ""){
        enc->apply(weights_init);
        dec->apply(weights_init);
        est->apply(weights_init);
        ofs.open(checkpoint_dir + "/log/train.txt", std::ios::out);
        if (vm["valid"].as<bool>()){
            init.open(checkpoint_dir + "/log/valid.txt", std::ios::trunc);
            init.close();
        }
        start_epoch = 0;
    }
    else{
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_enc.pth";  torch::load(enc, path);
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_dec.pth";  torch::load(dec, path);
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_est.pth";  torch::load(est, path);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_enc.pth";  torch::load(enc_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_dec.pth";  torch::load(dec_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_est.pth";  torch::load(est_optimizer, path);
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
    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++){

        enc->train();
        dec->train();
        est->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{epoch, total_epoch}, /*loss_=*/{"rec", "energy", "penalty"});

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){

            image = std::get<0>(mini_batch).to(device);

            // -----------------------------------
            // c1. DAGMM Training Phase
            // -----------------------------------

            // (1) Encoder-Decoder Forward
            z_c = enc->forward(image);   // {C,H,W} ===> {ZC,1,1}
            output = dec->forward(z_c);  // {ZC,1,1} ===> {C,H,W}

            // (2) Setting Latent Space
            if (vm["com_detach"].as<bool>()){
                z_c = z_c.detach();
            }
            z_c = z_c.view({z_c.size(0), z_c.size(1)});  // {ZC,1,1} ===> {ZC}
            if (vm["RED"].as<bool>()){
                z_r1 = RelativeEuclideanDistance(image, output, vm["rec_detach"].as<bool>());
            }
            else{
                z_r1 = AbsoluteEuclideanDistance(image, output, vm["rec_detach"].as<bool>());
            }
            z_r2 = CosineSimilarity(image, output, vm["rec_detach"].as<bool>());
            z_r = torch::cat({z_r1, z_r2}, /*dim=*/1);  // {1} + {1} ===> {ZR} = {2}
            z = torch::cat({z_c, z_r}, /*dim=*/1);  // {ZC} + {ZR} ===> {Z} = {ZC+ZR}

            // (3) Estimation of GMM-Parameters
            gamma_ap = est->forward(z);  // {Z} ===> {K}
            if (vm["no_NVI"].as<bool>()){
                est->estimation(z, gamma_ap);  // calculation of mean, variance, mixing coefficient, energy, precision
                energy = est->energy_just_before() * vm["Lambda_E"].as<float>();
            }
            else{
                est->estimationNVI(z, gamma_ap);  // calculation of mean, variance, mixing coefficient, NVI, precision
                energy = est->NVI_just_before() * vm["Lambda_E"].as<float>();
            }

            // (4) Calculation of Reconstruction and Penalty Loss
            rec = criterion(output, image);
            penalty = est->precision_just_before() * vm["Lambda_P"].as<float>();
            loss = rec + energy + penalty;

            // (5) DAGMM Training
            enc_optimizer.zero_grad();
            dec_optimizer.zero_grad();
            est_optimizer.zero_grad();
            loss.backward();
            enc_optimizer.step();
            dec_optimizer.step();
            est_optimizer.step();

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            show_progress->increment(/*loss_value=*/{rec.item<float>(), energy.item<float>(), penalty.item<float>()});
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "rec:" << rec.item<float>() << "(ave:" <<  show_progress->get_ave(0) << ") " << std::flush;
            ofs << "energy:" << energy.item<float>() << "(ave:" <<  show_progress->get_ave(1) << ") " << std::flush;
            ofs << "penalty:" << penalty.item<float>() << "(ave:" <<  show_progress->get_ave(2) << ")" << std::endl;

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
        train_loss.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(0) + show_progress->get_ave(1) + show_progress->get_ave(2), show_progress->get_ave(0), show_progress->get_ave(1), show_progress->get_ave(2)});
        train_rec_loss.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(0)});

        // -----------------------------------
        // b3. Save Sample Images
        // -----------------------------------
        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        visualizer::save_image(output.detach(), ss.str(), /*range=*/output_range);
        delete show_progress;

        // ----------------------------------------
        // b4. Update Gaussian Mixture Parameters
        // ----------------------------------------
        get_gmp(vm, dataloader, device, enc, dec, est, total_iter, mu, sigma, phi);
        
        // -----------------------------------
        // b5. Validation Mode
        // -----------------------------------
        if (vm["valid"].as<bool>() && ((epoch - 1) % vm["valid_freq"].as<size_t>() == 0)){
            valid(vm, valid_dataloader, device, criterion, enc, dec, est, mu, sigma, phi, epoch, valid_loss, valid_rec_loss);
        }

        // -----------------------------------
        // b6. Save Model Weights and Optimizer Parameters
        // -----------------------------------
        if (epoch % vm["save_epoch"].as<size_t>() == 0){
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_gmp.dat";  save_params(path, mu, sigma, phi);
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_enc.pth";  torch::save(enc, path);
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_dec.pth";  torch::save(dec, path);
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_est.pth";  torch::save(est, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_enc.pth";  torch::save(enc_optimizer, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_dec.pth";  torch::save(dec_optimizer, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_est.pth";  torch::save(est_optimizer, path);
        }
        path = checkpoint_dir + "/models/epoch_latest_gmp.dat";  save_params(path, mu, sigma, phi);
        path = checkpoint_dir + "/models/epoch_latest_enc.pth";  torch::save(enc, path);
        path = checkpoint_dir + "/models/epoch_latest_dec.pth";  torch::save(dec, path);
        path = checkpoint_dir + "/models/epoch_latest_est.pth";  torch::save(est, path);
        path = checkpoint_dir + "/optims/epoch_latest_enc.pth";  torch::save(enc_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_latest_dec.pth";  torch::save(dec_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_latest_est.pth";  torch::save(est_optimizer, path);
        infoo.open(checkpoint_dir + "/models/info.txt", std::ios::out);
        infoo << "latest = " << epoch << std::endl;
        infoo.close();

        // -----------------------------------
        // b7. Show Elapsed Time
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


// ---------------------------------------------
// Function to Get Gaussian Mixture Parameters
// ---------------------------------------------
void get_gmp(po::variables_map &vm, DataLoader::ImageFolderWithPaths &dataloader, torch::Device &device, Encoder &enc, Decoder &dec, EstimationNetwork &est, size_t total_iter, torch::Tensor &mu, torch::Tensor &sigma, torch::Tensor &phi){

    // (0) Initialization and Declaration
    torch::Tensor image, output;
    torch::Tensor z, z_c, z_r, z_r1, z_r2, gamma_ap;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    progress::display *show_progress;

    // (1) Tensor Forward per Mini Batch
    enc->eval();
    dec->eval();
    est->eval();
    est->resetGMP(device);

    show_progress = new progress::display(/*count_max_=*/total_iter, /*header1=*/"---------------", /*header2=*/"Estimation_GMP", /*loss_=*/{});
    while (dataloader(mini_batch)){

        image = std::get<0>(mini_batch).to(device);

        // (1.1) Encoder-Decoder Forward
        z_c = enc->forward(image);   // {C,H,W} ===> {ZC,1,1}
        output = dec->forward(z_c);  // {ZC,1,1} ===> {C,H,W}

        // (1.2) Setting Latent Space
        z_c = z_c.view({z_c.size(0), z_c.size(1)});  // {ZC,1,1} ===> {ZC}
        if (vm["RED"].as<bool>()){
            z_r1 = RelativeEuclideanDistance(image, output);
        }
        else{
            z_r1 = AbsoluteEuclideanDistance(image, output);
        }
        z_r2 = CosineSimilarity(image, output);
        z_r = torch::cat({z_r1, z_r2}, /*dim=*/1);  // {1} + {1} ===> {ZR} = {2}
        z = torch::cat({z_c, z_r}, /*dim=*/1);  // {ZC} + {ZR} ===> {Z} = {ZC+ZR}

        // (1.3) Estimation of Gaussian Mixture Parameters
        gamma_ap = est->forward(z);  // {Z} ===> {K}
        est->estimationGMP(z, gamma_ap);
        show_progress->increment(/*loss_value=*/{});

    }

    // (2) Set Gaussian Mixture Parameters
    mu = est->estimated_mu().clone();
    sigma = est->estimated_sigma().clone();
    phi = est->estimated_phi().clone();

    // Post Processing
    delete show_progress;

    // End Processing
    return;

}