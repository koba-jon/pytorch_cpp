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
#include "networks.hpp"                // ResNet_Generator, PatchGAN_Discriminator
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderPairWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderPairWithPaths
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void valid(po::variables_map &vm, DataLoader::ImageFolderPairWithPaths &valid_dataloader, torch::Device &device, torch::nn::L1Loss &criterion, ResNet_Generator &genA, ResNet_Generator &genB, const size_t epoch, visualizer::graph &writer);


// -------------------
// Training Function
// -------------------
void train(po::variables_map &vm, torch::Device &device, ResNet_Generator &genAB, ResNet_Generator &genBA, PatchGAN_Discriminator &disA, PatchGAN_Discriminator &disB, std::vector<transforms_Compose> &transformA, std::vector<transforms_Compose> &transformB){

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
    std::string A_dir, B_dir;
    std::string valid_A_dir, valid_B_dir;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor realA, realB, fakeA, fakeB, recA, recB, pair;
    torch::Tensor dis_realA_out, dis_realB_out, dis_fakeA_out, dis_fakeB_out;
    torch::Tensor gen_loss, G_AB_GAN_loss, G_BA_GAN_loss, G_ABA_Cycle_loss, G_BAB_Cycle_loss;
    torch::Tensor disA_loss, disB_loss, dis_realA_loss, dis_realB_loss, dis_fakeA_loss, dis_fakeB_loss;
    torch::Tensor label_real, label_fake;
    datasets::ImageFolderRandomSampling2WithPaths dataset;
    datasets::ImageFolderPairWithPaths valid_dataset;
    DataLoader::ImageFolderRandomSampling2WithPaths dataloader;
    DataLoader::ImageFolderPairWithPaths valid_dataloader;
    visualizer::graph train_loss, train_loss_gen, train_loss_disA, train_loss_disB;
    visualizer::graph valid_loss;
    progress::display *show_progress;
    progress::irregular irreg_progress;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    A_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_A_dir"].as<std::string>();
    B_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_B_dir"].as<std::string>();
    dataset = datasets::ImageFolderRandomSampling2WithPaths(A_dir, B_dir, transformA, transformB);
    dataloader = DataLoader::ImageFolderRandomSampling2WithPaths(dataset, vm["batch_size"].as<size_t>(), /*num_workers_=*/train_workers);
    std::cout << "total training images A : " << dataset.size1() << std::endl;
    std::cout << "total training images B : " << dataset.size2() << std::endl;

    // (2) Get Validation Dataset
    if (vm["valid"].as<bool>()){
        valid_A_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["valid_A_dir"].as<std::string>();
        valid_B_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["valid_B_dir"].as<std::string>();
        valid_dataset = datasets::ImageFolderPairWithPaths(valid_A_dir, valid_B_dir, transformA, transformB);
        valid_dataloader = DataLoader::ImageFolderPairWithPaths(valid_dataset, vm["valid_batch_size"].as<size_t>(), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    // (3) Set Optimizer Method
    auto genAB_optimizer = torch::optim::Adam(genAB->parameters(), torch::optim::AdamOptions(vm["lr_gen"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));
    auto genBA_optimizer = torch::optim::Adam(genBA->parameters(), torch::optim::AdamOptions(vm["lr_gen"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));
    auto disA_optimizer = torch::optim::Adam(disA->parameters(), torch::optim::AdamOptions(vm["lr_dis"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));
    auto disB_optimizer = torch::optim::Adam(disB->parameters(), torch::optim::AdamOptions(vm["lr_dis"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

    // (4) Set Loss Function
    auto criterion_GAN = Loss(vm["loss"].as<std::string>());
    auto criterion_L1 = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kMean));

    // (5) Make Directories
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);
    save_images_dir = checkpoint_dir + "/samples";  fs::create_directories(save_images_dir);

    // (6) Set Training Loss for Graph
    path = checkpoint_dir + "/graph";
    train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{"Generator", "DiscriminatorA", "DiscriminatorB"});
    train_loss_gen = visualizer::graph(path, /*gname_=*/"train_loss_gen", /*label_=*/{"Total", "AB_{GAN}", "BA_{GAN}", "ABA_{Cycle}", "BAB_{Cycle}"});
    train_loss_disA = visualizer::graph(path, /*gname_=*/"train_loss_disA", /*label_=*/{"Total", "Real", "Fake"});
    train_loss_disB = visualizer::graph(path, /*gname_=*/"train_loss_disB", /*label_=*/{"Total", "Real", "Fake"});
    if (vm["valid"].as<bool>()){
        valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{"AtoB", "BtoA"});
    }
    
    // (7) Get Weights and File Processing
    if (vm["train_load_epoch"].as<std::string>() == ""){
        genAB->apply(weights_init);
        genBA->apply(weights_init);
        disA->apply(weights_init);
        disB->apply(weights_init);
        ofs.open(checkpoint_dir + "/log/train.txt", std::ios::out);
        if (vm["valid"].as<bool>()){
            init.open(checkpoint_dir + "/log/valid.txt", std::ios::trunc);
            init.close();
        }
        start_epoch = 0;
    }
    else{
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_genAB.pth";  torch::load(genAB, path, device);
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_genBA.pth";  torch::load(genBA, path, device);
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_disA.pth";  torch::load(disA, path, device);
        path = checkpoint_dir + "/models/epoch_" + vm["train_load_epoch"].as<std::string>() + "_disB.pth";  torch::load(disB, path, device);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_genAB.pth";  torch::load(genAB_optimizer, path, device);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_genBA.pth";  torch::load(genBA_optimizer, path, device);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_disA.pth";  torch::load(disA_optimizer, path, device);
        path = checkpoint_dir + "/optims/epoch_" + vm["train_load_epoch"].as<std::string>() + "_disB.pth";  torch::load(disB_optimizer, path, device);
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
    total_iter = vm["iters"].as<size_t>();
    total_epoch = vm["epochs"].as<size_t>();

    // (2) Training per Epoch
    mini_batch_size = 0;
    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++){

        genAB->train();
        genBA->train();
        disA->train();
        disB->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{epoch, total_epoch}, /*loss_=*/{"G", "G_AB_GAN", "G_BA_GAN", "G_ABA_Cycle", "G_BAB_Cycle", "D_A", "D_A_Real", "D_A_Fake", "D_B", "D_B_Real", "D_B_Fake"});

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){

            realA = std::get<0>(mini_batch).to(device);
            realB = std::get<1>(mini_batch).to(device);
            mini_batch_size = realA.size(0);

            // -----------------------------------
            // c1. Discriminator and Generator Training Phase
            // -----------------------------------

            // (1) Generator and Discriminator Forward
            fakeB = genAB->forward(realA);
            recA = genBA->forward(fakeB);
            fakeA = genBA->forward(realB);
            recB = genAB->forward(fakeA);
            dis_fakeA_out = disA->forward(fakeA.detach());
            dis_fakeB_out = disB->forward(fakeB.detach());

            // (2) Set Target Label
            label_real = torch::full_like(dis_fakeA_out, /*value*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
            label_fake = torch::full_like(dis_fakeA_out, /*value*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

            // (3.A) Discriminator A Training
            dis_realA_out = disA->forward(realA);
            dis_realA_loss = criterion_GAN(dis_realA_out, label_real);
            dis_fakeA_loss = criterion_GAN(dis_fakeA_out, label_fake);
            disA_loss = dis_realA_loss + dis_fakeA_loss;
            disA_optimizer.zero_grad();
            disA_loss.backward();
            disA_optimizer.step();

            // (3.B) Discriminator B Training
            dis_realB_out = disB->forward(realB);
            dis_realB_loss = criterion_GAN(dis_realB_out, label_real);
            dis_fakeB_loss = criterion_GAN(dis_fakeB_out, label_fake);
            disB_loss = dis_realB_loss + dis_fakeB_loss;
            disB_optimizer.zero_grad();
            disB_loss.backward();
            disB_optimizer.step();

            // (4) Generator AB and BA Training
            dis_fakeA_out = disA->forward(fakeA);
            dis_fakeB_out = disB->forward(fakeB);
            G_AB_GAN_loss = criterion_GAN(dis_fakeB_out, label_real);
            G_BA_GAN_loss = criterion_GAN(dis_fakeA_out, label_real);
            G_ABA_Cycle_loss = criterion_L1(recA, realA) * vm["Lambda"].as<float>();
            G_BAB_Cycle_loss = criterion_L1(recB, realB) * vm["Lambda"].as<float>();
            gen_loss = G_AB_GAN_loss + G_BA_GAN_loss + G_ABA_Cycle_loss + G_BAB_Cycle_loss;
            genAB_optimizer.zero_grad();
            genBA_optimizer.zero_grad();
            gen_loss.backward();
            genAB_optimizer.step();
            genBA_optimizer.step();

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            show_progress->increment(/*loss_value=*/{gen_loss.item<float>(), G_AB_GAN_loss.item<float>(), G_BA_GAN_loss.item<float>(), G_ABA_Cycle_loss.item<float>(), G_BAB_Cycle_loss.item<float>(), disA_loss.item<float>(), dis_realA_loss.item<float>(), dis_fakeA_loss.item<float>(), disB_loss.item<float>(), dis_realB_loss.item<float>(), dis_fakeB_loss.item<float>()}, {1, 2, 3, 4, 6, 7, 9, 10});
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "G_AB_GAN:" << G_AB_GAN_loss.item<float>() << "(ave:" <<  show_progress->get_ave(1) << ") " << std::flush;
            ofs << "G_BA_GAN:" << G_BA_GAN_loss.item<float>() << "(ave:" <<  show_progress->get_ave(2) << ") " << std::flush;
            ofs << "G_ABA_Cycle:" << G_ABA_Cycle_loss.item<float>() << "(ave:" <<  show_progress->get_ave(3) << ") " << std::flush;
            ofs << "G_BAB_Cycle:" << G_BAB_Cycle_loss.item<float>() << "(ave:" <<  show_progress->get_ave(4) << ") " << std::flush;
            ofs << "D_A_Real:" << dis_realA_loss.item<float>() << "(ave:" <<  show_progress->get_ave(6) << ") " << std::flush;
            ofs << "D_A_Fake:" << dis_fakeA_loss.item<float>() << "(ave:" <<  show_progress->get_ave(7) << ") " << std::flush;
            ofs << "D_B_Real:" << dis_realB_loss.item<float>() << "(ave:" <<  show_progress->get_ave(9) << ") " << std::flush;
            ofs << "D_B_Fake:" << dis_fakeB_loss.item<float>() << "(ave:" <<  show_progress->get_ave(10) << ")" << std::endl;


            // -----------------------------------
            // c3. Save Sample Images
            // -----------------------------------
            iter = show_progress->get_iters();
            if (iter % save_sample_iter == 1){
                ss.str(""); ss.clear(std::stringstream::goodbit);
                ss << save_images_dir << "/epoch_" << epoch << "-iter_" << iter << '.' << extension;
                if ((realA.size(1) == 3) || (realB.size(1) == 3) || (fakeA.size(1) == 3) || (fakeB.size(1) == 3) || (recA.size(1) == 3) || (recB.size(1) == 3)){
                    if (realA.size(1) == 1) realA = realA.expand({realA.size(0), 3, realA.size(2), realA.size(3)}).contiguous();
                    realB = realB.expand_as(realA).contiguous();
                    fakeA = fakeA.expand_as(realA).contiguous();
                    fakeB = fakeB.expand_as(realA).contiguous();
                    recA = recA.expand_as(realA).contiguous();
                    recB = recB.expand_as(realA).contiguous();
                }
                pair = torch::cat({realA, fakeB, recA, realB, fakeA, recB}, /*dim=*/0);
                visualizer::save_image(pair.detach(), ss.str(), /*range=*/output_range, /*cols=*/mini_batch_size);
            }

            if (iter == total_iter) break;

        }

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        train_loss.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(0), show_progress->get_ave(5), show_progress->get_ave(8)});
        train_loss_gen.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(0), show_progress->get_ave(1), show_progress->get_ave(2), show_progress->get_ave(3), show_progress->get_ave(4)});
        train_loss_disA.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(5), show_progress->get_ave(6), show_progress->get_ave(7)});
        train_loss_disB.plot(/*base=*/epoch, /*value=*/{show_progress->get_ave(8), show_progress->get_ave(9), show_progress->get_ave(10)});

        // -----------------------------------
        // b3. Save Sample Images
        // -----------------------------------
        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        if ((realA.size(1) == 3) || (realB.size(1) == 3) || (fakeA.size(1) == 3) || (fakeB.size(1) == 3) || (recA.size(1) == 3) || (recB.size(1) == 3)){
            if (realA.size(1) == 1) realA = realA.expand({realA.size(0), 3, realA.size(2), realA.size(3)}).contiguous();
            realB = realB.expand_as(realA).contiguous();
            fakeA = fakeA.expand_as(realA).contiguous();
            fakeB = fakeB.expand_as(realA).contiguous();
            recA = recA.expand_as(realA).contiguous();
            recB = recB.expand_as(realA).contiguous();
        }
        pair = torch::cat({realA, fakeB, recA, realB, fakeA, recB}, /*dim=*/0);
        visualizer::save_image(pair.detach(), ss.str(), /*range=*/output_range, /*cols=*/mini_batch_size);
        delete show_progress;
        
        // -----------------------------------
        // b4. Validation Mode
        // -----------------------------------
        if (vm["valid"].as<bool>() && ((epoch - 1) % vm["valid_freq"].as<size_t>() == 0)){
            valid(vm, valid_dataloader, device, criterion_L1, genAB, genBA, epoch, valid_loss);
        }

        // -----------------------------------
        // b5. Save Model Weights and Optimizer Parameters
        // -----------------------------------
        if (epoch % vm["save_epoch"].as<size_t>() == 0){
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_genAB.pth";  torch::save(genAB, path);
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_genBA.pth";  torch::save(genBA, path);
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_disA.pth";  torch::save(disA, path);
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_disB.pth";  torch::save(disB, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_genAB.pth";  torch::save(genAB_optimizer, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_genBA.pth";  torch::save(genBA_optimizer, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_disA.pth";  torch::save(disA_optimizer, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_disB.pth";  torch::save(disB_optimizer, path);
        }
        path = checkpoint_dir + "/models/epoch_latest_genAB.pth";  torch::save(genAB, path);
        path = checkpoint_dir + "/models/epoch_latest_genBA.pth";  torch::save(genBA, path);
        path = checkpoint_dir + "/models/epoch_latest_disA.pth";  torch::save(disA, path);
        path = checkpoint_dir + "/models/epoch_latest_disB.pth";  torch::save(disB, path);
        path = checkpoint_dir + "/optims/epoch_latest_genAB.pth";  torch::save(genAB_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_latest_genBA.pth";  torch::save(genBA_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_latest_disA.pth";  torch::save(disA_optimizer, path);
        path = checkpoint_dir + "/optims/epoch_latest_disB.pth";  torch::save(disB_optimizer, path);
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
