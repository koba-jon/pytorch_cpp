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
#include "networks.hpp"                // GAN_Encoder, GAN_Generator, GAN_Discriminator
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, GAN_Encoder &enc, GAN_Generator &gen, GAN_Discriminator &dis, const size_t epoch, visualizer::graph &writer, visualizer::graph &writer_dis);


// -------------------
// Training Function
// -------------------
void train(po::variables_map &vm, torch::Device &device, GAN_Encoder &enc, GAN_Generator &gen, GAN_Discriminator &dis, std::vector<transforms_Compose> &transform){

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
    size_t mini_batch_size;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, save_images_dir, path;
    std::string dataroot, valid_dataroot;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, fake_image, z_input, z_output;
    torch::Tensor dis_real_out, dis_fake_out;
    torch::Tensor enc_loss, gen_loss, dis_loss, dis_real_loss, dis_fake_loss;
    torch::Tensor label_real, label_fake;
    datasets::ImageFolderWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderWithPaths dataloader, valid_dataloader;
    visualizer::graph train_loss, train_loss_dis, valid_loss, valid_loss_dis;
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
    auto enc_optimizer = torch::optim::Adam(enc->parameters(), torch::optim::AdamOptions(vm["lr_enc"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));
    auto gen_optimizer = torch::optim::Adam(gen->parameters(), torch::optim::AdamOptions(vm["lr_gen"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));
    auto dis_optimizer = torch::optim::Adam(dis->parameters(), torch::optim::AdamOptions(vm["lr_dis"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

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
    train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{"Encoder", "Generator", "Discriminator"});
    train_loss_dis = visualizer::graph(path, /*gname_=*/"train_loss_dis", /*label_=*/{"Total", "Real", "Fake"});
    if (vm["valid"].as<bool>()){
        valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{"Encoder", "Generator", "Discriminator"});
        valid_loss_dis = visualizer::graph(path, /*gname_=*/"valid_loss_dis", /*label_=*/{"Total", "Real", "Fake"});
    }
    
    // (7) Get Weights and File Processing
    if (vm["train_load_epoch"].as<std::string>() == ""){
        enc->apply(weights_init);
        gen->apply(weights_init);
        dis->apply(weights_init);
        ofs.open(checkpoint_dir + "/log/train.txt", std::ios::out);
        if (vm["valid"].as<bool>()){
            init.open(checkpoint_dir + "/log/valid.txt", std::ios::trunc);
            init.close();
        }
        start_epoch = 0;
    }
    else{
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_enc.pth";  torch::load(enc, path);
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_gen.pth";  torch::load(gen, path);
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_dis.pth";  torch::load(dis, path);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_enc.pth";  torch::load(enc_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_gen.pth";  torch::load(gen_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_dis.pth";  torch::load(dis_optimizer, path);
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

        enc->train();
        gen->train();
        dis->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{epoch, total_epoch}, /*loss_=*/{"E", "G", "D_Real", "D_Fake"});

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){

            image = std::get<0>(mini_batch).to(device);
            mini_batch_size = image.size(0);

            // --------------------------------------------------------
            // c1. Discriminator, Encoder and Generator Training Phase
            // --------------------------------------------------------

            // (1) Set Target Label
            label_real = torch::full({(long int)mini_batch_size}, /*value=*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
            label_fake = torch::full({(long int)mini_batch_size}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

            // (2) Discriminator Training
            z_input = torch::randn({(long int)mini_batch_size, (long int)vm["nz"].as<size_t>()}).to(device);
            fake_image = gen->forward(z_input);
            z_output = enc->forward(image);
            dis_fake_out = dis->forward(fake_image.detach(), z_input).first.view({-1});
            dis_real_out = dis->forward(image, z_output.detach()).first.view({-1});
            dis_fake_loss = criterion(dis_fake_out, label_fake);
            dis_real_loss = criterion(dis_real_out, label_real);
            dis_loss = dis_real_loss + dis_fake_loss;
            if (dis_loss.item<float>() > criterion.ideal_value() * 2.0 * vm["D_ltw"].as<float>()){
                dis_optimizer.zero_grad();
                dis_loss.backward();
                dis_optimizer.step();
            }

            // (3) Encoder Training
            dis_real_out = dis->forward(image, z_output).first.view({-1});
            enc_loss = criterion(dis_real_out, label_fake);
            if (enc_loss.item<float>() > criterion.ideal_value() * vm["E_ltw"].as<float>()){
                enc_optimizer.zero_grad();
                enc_loss.backward();
                enc_optimizer.step();
            }

            // (4) Generator Training
            dis_fake_out = dis->forward(fake_image, z_input).first.view({-1});
            gen_loss = criterion(dis_fake_out, label_real);
            if (gen_loss.item<float>() > criterion.ideal_value() * vm["G_ltw"].as<float>()){
                gen_optimizer.zero_grad();
                gen_loss.backward();
                gen_optimizer.step();
            }

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            show_progress->increment(/*loss_value=*/{enc_loss.item<float>(), gen_loss.item<float>(), dis_real_loss.item<float>(), dis_fake_loss.item<float>()});
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "E:" << enc_loss.item<float>() << "(ave:" <<  show_progress->get_ave(0) << ") " << std::flush;
            ofs << "G:" << gen_loss.item<float>() << "(ave:" <<  show_progress->get_ave(1) << ") " << std::flush;
            ofs << "D_Real:" << dis_real_loss.item<float>() << "(ave:" <<  show_progress->get_ave(2) << ") " << std::flush;
            ofs << "D_Fake:" << dis_fake_loss.item<float>() << "(ave:" <<  show_progress->get_ave(3) << ")" << std::endl;

            // -----------------------------------
            // c3. Save Sample Images
            // -----------------------------------
            iter = show_progress->get_iters();
            if (iter % save_sample_iter == 1){
                ss.str(""); ss.clear(std::stringstream::goodbit);
                ss << save_images_dir << "/epoch_" << epoch << "-iter_" << iter << '.' << extension;
                visualizer::save_image(fake_image.detach(), ss.str(), /*range=*/output_range);
            }

        }

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        train_loss.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(0), show_progress->get_ave(1), show_progress->get_ave(2) + show_progress->get_ave(3)});
        train_loss_dis.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(2) + show_progress->get_ave(3), show_progress->get_ave(2), show_progress->get_ave(3)});

        // -----------------------------------
        // b3. Save Sample Images
        // -----------------------------------
        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        visualizer::save_image(fake_image.detach(), ss.str(), /*range=*/output_range);
        delete show_progress;
        
        // -----------------------------------
        // b4. Validation Mode
        // -----------------------------------
        if (vm["valid"].as<bool>() && ((epoch - 1) % vm["valid_freq"].as<size_t>() == 0)){
            valid(vm, valid_dataloader, device, criterion, enc, gen, dis, epoch, valid_loss, valid_loss_dis);
        }

        // -----------------------------------
        // b5. Save Model Weights and Optimizer Parameters
        // -----------------------------------
        if (epoch % vm["save_epoch"].as<size_t>() == 0){
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_enc.pth";  torch::save(enc, path);
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_gen.pth";  torch::save(gen, path);
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_dis.pth";  torch::save(dis, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_enc.pth";  torch::save(enc_optimizer, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_gen.pth";  torch::save(gen_optimizer, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_dis.pth";  torch::save(dis_optimizer, path);
        }
        path = checkpoint_dir + "/models/epoch_latest_enc.pth";  torch::save(enc, path);
        path = checkpoint_dir + "/models/epoch_latest_gen.pth";  torch::save(gen, path);
        path = checkpoint_dir + "/models/epoch_latest_dis.pth";  torch::save(dis, path);
        path = checkpoint_dir + "/optims/epoch_latest_enc.pth";  torch::save(enc_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_latest_gen.pth";  torch::save(gen_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_latest_dis.pth";  torch::save(dis_optimizer, path);
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
