#include <iostream>                    // std::flush
#include <fstream>                     // std::ofstream
#include <string>                      // std::string
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // UNet_Generator, GAN_Discriminator
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, torch::nn::BCEWithLogitsLoss &criterion_adv, Loss &criterion_con, Loss &criterion_lat, UNet_Generator &gen, GAN_Discriminator &dis, const size_t epoch, visualizer::graph &writer_rec, visualizer::graph &writer_lat, visualizer::graph &writer_gan, visualizer::graph &writer_dis){

    // (0) Initialization and Declaration
    size_t iteration;
    size_t mini_batch_size;
    float ave_adv_loss, total_adv_loss;
    float ave_con_loss, total_con_loss;
    float ave_lat_loss, total_lat_loss;
    float ave_dis_real_loss, total_dis_real_loss;
    float ave_dis_fake_loss, total_dis_fake_loss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, fake_image;
    torch::Tensor dis_real_feat, dis_fake_feat;
    torch::Tensor dis_real_out, dis_fake_out;
    torch::Tensor adv_loss, con_loss, lat_loss, dis_real_loss, dis_fake_loss;
    torch::Tensor label_real, label_fake;

    // (1) Tensor Forward per Mini Batch
    gen->eval();
    dis->eval();
    iteration = 0;
    total_adv_loss = 0.0; total_con_loss = 0.0; total_lat_loss = 0.0; total_dis_real_loss = 0.0; total_dis_fake_loss = 0.0;
    while (valid_dataloader(mini_batch)){

        image = std::get<0>(mini_batch).to(device);
        mini_batch_size = image.size(0);

        // (1.1) Set Target Label
        label_real = torch::full({(long int)mini_batch_size}, /*value=*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
        label_fake = torch::full({(long int)mini_batch_size}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

        // (1.2) Discriminator Loss
        fake_image = gen->forward(image);
        dis_fake_out = dis->forward(fake_image).first.view({-1});
        dis_real_out = dis->forward(image).first.view({-1});
        dis_fake_loss = criterion_adv(dis_fake_out, label_fake);
        dis_real_loss = criterion_adv(dis_real_out, label_real);

        // (1.3) Generator and Encoder Loss
        dis_fake_out = dis->forward(fake_image).first.view({-1});
        dis_fake_feat = dis->forward(fake_image).second;
        dis_real_feat = dis->forward(image).second;
        adv_loss = criterion_adv(dis_fake_out, label_real) * vm["Lambda_adv"].as<float>();
        con_loss = criterion_con(fake_image, image) * vm["Lambda_con"].as<float>();
        lat_loss = criterion_lat(dis_fake_feat, dis_real_feat) * vm["Lambda_lat"].as<float>();

        // (1.4) Update Loss
        total_adv_loss += adv_loss.item<float>();
        total_con_loss += con_loss.item<float>();
        total_lat_loss += lat_loss.item<float>();
        total_dis_real_loss += dis_real_loss.item<float>();
        total_dis_fake_loss += dis_fake_loss.item<float>();

        iteration++;
    }

    // (2) Calculate Average Loss
    ave_adv_loss = total_adv_loss / (float)iteration;
    ave_con_loss = total_con_loss / (float)iteration;
    ave_lat_loss = total_lat_loss / (float)iteration;
    ave_dis_real_loss = total_dis_real_loss / (float)iteration;
    ave_dis_fake_loss = total_dis_fake_loss / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "adv:" << ave_adv_loss << ' ' << std::flush;
    ofs << "con:" << ave_con_loss << ' ' << std::flush;
    ofs << "lat:" << ave_lat_loss << ' ' << std::flush;
    ofs << "D_Real:" << ave_dis_real_loss << ' ' << std::flush;
    ofs << "D_Fake:" << ave_dis_fake_loss << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer_rec.plot(/*base=*/epoch, /*value=*/{ave_con_loss});
    writer_lat.plot(/*base=*/epoch, /*value=*/{ave_lat_loss});
    writer_gan.plot(/*base=*/epoch, /*value=*/{ave_adv_loss, ave_dis_real_loss + ave_dis_fake_loss});
    writer_dis.plot(/*base=*/epoch, /*value=*/{ave_dis_real_loss + ave_dis_fake_loss, ave_dis_real_loss, ave_dis_fake_loss});

    // End Processing
    return;

}