#include <iostream>                    // std::flush
#include <fstream>                     // std::ofstream
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // GAN_Generator, GAN_Discriminator
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Validation Function
// -------------------
void valid(po::variables_map &vm, DataLoader::ImageFolderWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, GAN_Generator &gen, GAN_Discriminator &dis, const size_t epoch, visualizer::graph &writer, visualizer::graph &writer_dis){

    constexpr std::string_view extension = "jpg";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    size_t iteration;
    size_t mini_batch_size;
    size_t max_counter;
    float ave_gen_loss, total_gen_loss;
    float ave_dis_real_loss, total_dis_real_loss;
    float ave_dis_fake_loss, total_dis_fake_loss;
    float value;
    std::stringstream ss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, fake_image, z;
    torch::Tensor dis_real_out, dis_fake_out;
    torch::Tensor gen_loss, dis_real_loss, dis_fake_loss;
    torch::Tensor label_real, label_fake;
    torch::Tensor output, outputs;

    // (1) Tensor Forward per Mini Batch
    gen->eval();
    dis->eval();
    iteration = 0;
    total_gen_loss = 0.0; total_dis_real_loss = 0.0; total_dis_fake_loss = 0.0;
    while (valid_dataloader(mini_batch)){

        image = std::get<0>(mini_batch).to(device);
        mini_batch_size = image.size(0);

        // (1.1) Set Target Label
        label_real = torch::full({(long int)mini_batch_size}, /*value=*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
        label_fake = torch::full({(long int)mini_batch_size}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

        // (1.2) Discriminator Training
        z = torch::randn({(long int)mini_batch_size, (long int)vm["nz"].as<size_t>()}).to(device);
        fake_image = gen->forward(z);
        dis_fake_out = dis->forward(fake_image.detach()).view({-1});
        dis_real_out = dis->forward(image).view({-1});
        dis_fake_loss = criterion(dis_fake_out, label_fake);
        dis_real_loss = criterion(dis_real_out, label_real);

        // (1.3) Generator Training
        dis_fake_out = dis->forward(fake_image).view({-1});
        gen_loss = criterion(dis_fake_out, label_real);

        // (1.4) Update Loss
        total_gen_loss += gen_loss.item<float>();
        total_dis_real_loss += dis_real_loss.item<float>();
        total_dis_fake_loss += dis_fake_loss.item<float>();

        iteration++;
    }

    // (2) Calculate Average Loss
    ave_gen_loss = total_gen_loss / (float)iteration;
    ave_dis_real_loss = total_dis_real_loss / (float)iteration;
    ave_dis_fake_loss = total_dis_fake_loss / (float)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + vm["dataset"].as<std::string>() + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << vm["epochs"].as<size_t>() << ' ' << std::flush;
    ofs << "G:" << ave_gen_loss << ' ' << std::flush;
    ofs << "D_Real:" << ave_dis_real_loss << ' ' << std::flush;
    ofs << "D_Fake:" << ave_dis_fake_loss << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ave_gen_loss, ave_dis_real_loss + ave_dis_fake_loss});
    writer_dis.plot(/*base=*/epoch, /*value=*/{ave_dis_real_loss + ave_dis_fake_loss, ave_dis_real_loss, ave_dis_fake_loss});

    // (4) Image Generation
    max_counter = (int)(vm["valid_sigma_max"].as<float>() / vm["valid_sigma_inter"].as<float>() * 2) + 1;
    z = torch::full({1, (long int)vm["nz"].as<size_t>()}, /*value=*/-vm["valid_sigma_max"].as<float>(), torch::TensorOptions().dtype(torch::kFloat)).to(device);
    outputs = gen->forward(z);
    for (size_t i = 1; i < max_counter; i++){
        value = -vm["valid_sigma_max"].as<float>() + (float)i * vm["valid_sigma_inter"].as<float>();
        z = torch::full({1, (long int)vm["nz"].as<size_t>()}, /*value=*/value, torch::TensorOptions().dtype(torch::kFloat)).to(device);
        output = gen->forward(z);
        outputs = torch::cat({outputs, output}, /*dim=*/0);
    }
    ss.str(""); ss.clear(std::stringstream::goodbit);
    ss << "checkpoints/" << vm["dataset"].as<std::string>() << "/samples/epoch_" << epoch << "-valid."  << extension;
    visualizer::save_image(outputs.detach(), ss.str(), /*range=*/output_range, /*cols=*/max_counter);

    // End Processing
    return;

}