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
#include "networks.hpp"                // ESRGAN_Generator, ESRGAN_Discriminator
#include "dataloader.hpp"              // DataLoader::ImageFolderPairWithPaths
#include "visualizer.hpp"              // visualizer::graph

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderPairWithPaths &valid_dataloader, torch::Device &device, torch::nn::L1Loss &criterion_L1, Loss &criterion_GAN, torch::nn::L1Loss &criterion_Con, ESRGAN_Generator &gen, ESRGAN_Discriminator &dis, MC_VGGNet &vgg, const size_t epoch, visualizer::graph &writer, visualizer::graph &writer_gen, visualizer::graph &writer_dis){

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_G_L1_loss, total_G_L1_loss;
    float ave_G_GAN_loss, total_G_GAN_loss;
    float ave_G_content_loss, total_G_content_loss;
    float ave_dis_real_loss, total_dis_real_loss;
    float ave_dis_fake_loss, total_dis_fake_loss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor lr, hr, sr;
    torch::Tensor content_hr, content_sr;
    torch::Tensor dis_real_out, dis_fake_out;
    torch::Tensor G_L1_loss, G_content_loss, G_GAN_loss;
    torch::Tensor dis_real_loss, dis_fake_loss;
    torch::Tensor label_real, label_fake;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    gen->eval();
    dis->eval();
    iteration = 0;
    total_G_L1_loss = 0.0; total_G_GAN_loss = 0.0; total_G_content_loss = 0.0; total_dis_real_loss = 0.0; total_dis_fake_loss = 0.0;
    while (valid_dataloader(mini_batch)){
        
        lr = std::get<0>(mini_batch).to(device);
        hr = std::get<1>(mini_batch).to(device);

        // (1.1) Set Target Label
        label_real = torch::full({lr.size(0)}, /*value*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
        label_fake = torch::full({lr.size(0)}, /*value*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

        // (1.2) Generator and Discriminator Forward
        sr = gen->forward(lr);
        dis_fake_out = dis->forward(sr).view({-1});
        dis_real_out = dis->forward(hr).view({-1});
        dis_fake_loss = criterion_GAN(dis_fake_out - dis_real_out.mean(), label_fake);
        dis_real_loss = criterion_GAN(dis_real_out - dis_fake_out.mean(), label_real);
        content_sr = vgg->forward(sr);
        content_hr = vgg->forward(hr);

        // (1.3) Generator Loss
        G_L1_loss = criterion_L1(sr, hr) * vm["eta"].as<float>();
        G_GAN_loss = criterion_GAN(dis_fake_out - dis_real_out.mean(), label_real) * vm["Lambda"].as<float>();
        G_content_loss = criterion_Con(content_sr, content_hr);

        // (1.4) Update Loss
        total_G_L1_loss += G_L1_loss.item<float>();
        total_G_GAN_loss += G_GAN_loss.item<float>();
        total_G_content_loss += G_content_loss.item<float>();
        total_dis_real_loss += dis_real_loss.item<float>();
        total_dis_fake_loss += dis_fake_loss.item<float>();

        iteration++;
    }

    // (2) Calculate Average Loss
    ave_G_L1_loss = total_G_L1_loss / (float)iteration;
    ave_G_GAN_loss = total_G_GAN_loss / (float)iteration;
    ave_G_content_loss = total_G_content_loss / (float)iteration;
    ave_dis_real_loss = total_dis_real_loss / (float)iteration;
    ave_dis_fake_loss = total_dis_fake_loss / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "G_L1:" << ave_G_L1_loss << ' ' << std::flush;
    ofs << "G_GAN:" << ave_G_GAN_loss << ' ' << std::flush;
    ofs << "G_Con:" << ave_G_content_loss << ' ' << std::flush;
    ofs << "D_Real:" << ave_dis_real_loss << ' ' << std::flush;
    ofs << "D_Fake:" << ave_dis_fake_loss << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_G_L1_loss + ave_G_GAN_loss + ave_G_content_loss, ave_dis_real_loss + ave_dis_fake_loss});
    writer_gen.plot(/*base=*/epoch, /*value=*/{ave_G_L1_loss + ave_G_GAN_loss + ave_G_content_loss, ave_G_L1_loss, ave_G_GAN_loss, ave_G_content_loss});
    writer_dis.plot(/*base=*/epoch, /*value=*/{ave_dis_real_loss + ave_dis_fake_loss, ave_dis_real_loss, ave_dis_fake_loss});

    // End Processing
    return;

}
