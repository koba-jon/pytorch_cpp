#include <iostream>                    // std::cout
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <tuple>                       // std::tuple
#include <chrono>                      // std::chrono
#include <utility>                     // std::pair
#include <limits>                      // std::numeric_limits
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // MC_ResNet
#include "transforms.hpp"              // transforms_Compose
#include "datasets.hpp"                // datasets::ImageFolderWithPaths, datasets::ImageFolderPairWithPaths
#include "dataloader.hpp"              // DataLoader::ImageFolderWithPaths, DataLoader::ImageFolderPairWithPaths
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace F = torch::nn::functional;

// Function Prototype
std::tuple<torch::Tensor, torch::Tensor> PatchCoreScore(torch::Tensor feature, torch::Tensor coreset, long int K);
torch::Tensor gaussian_filter_2d(torch::Tensor x, double sigma);


// ---------------
// Test Function
// ---------------
void test(po::variables_map &vm, torch::Device &device, MC_ResNet &model, std::vector<transforms_Compose> &transform, std::vector<transforms_Compose> &transformGT){

    // (0) Initialization and Declaration
    float ave_anomaly_scoreN, ave_anomaly_scoreA, score_map_up_min, score_map_up_max;
    float *p, *p_pos, *p_neg;
    double seconds, ave_timeN, ave_timeA;
    std::string path, result_dir, heatmapN_dir, heatmapA_dir, heatmapN_norm_dir, heatmapA_norm_dir, fname;
    std::string datarootN, datarootA, datarootGT;
    std::ofstream ofs, ofs_image_score, ofs_pixel_scoreN, ofs_pixel_scoreA;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, std::vector<std::string>> dataN;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> dataA;
    torch::Tensor image, gt, heatmap;
    torch::Tensor coreset, feature;
    torch::Tensor score_map, score_map_up, anomaly_score;
    torch::Tensor s, mask, s_pos, s_neg;
    datasets::ImageFolderWithPaths datasetN;
    datasets::ImageFolderPairWithPaths datasetA;
    DataLoader::ImageFolderWithPaths dataloaderN;
    DataLoader::ImageFolderPairWithPaths dataloaderA;

    // (1) Get Test Normal Dataset
    datarootN = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_normal_dir"].as<std::string>();
    datasetN = datasets::ImageFolderWithPaths(datarootN, transform);
    dataloaderN = DataLoader::ImageFolderWithPaths(datasetN, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test normal images : " << datasetN.size() << std::endl;

    // (2) Get Test Anomaly Dataset
    datarootA = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_anomaly_dir"].as<std::string>();
    datarootGT = "datasets/" + vm["dataset"].as<std::string>() + '/' + vm["test_gt_dir"].as<std::string>();
    datasetA = datasets::ImageFolderPairWithPaths(datarootA, datarootGT, transform, transformGT);
    dataloaderA = DataLoader::ImageFolderPairWithPaths(datasetA, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test anomaly images : " << datasetA.size() << std::endl << std::endl;

    // (3) Get Model
    torch::load(model, vm["resnet_path"].as<std::string>(), device);
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/coreset.pth"; torch::load(coreset, path, device);
    coreset = coreset.to(device);

    // (4) Initialization of Value
    ave_anomaly_scoreN = 0.0;
    ave_anomaly_scoreA = 0.0;
    ave_timeN = 0.0;
    ave_timeA = 0.0;
    score_map_up_min = std::numeric_limits<float>::infinity();
    score_map_up_max = -std::numeric_limits<float>::infinity();

    // (5) Tensor Forward for test normal
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = vm["test_result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs_pixel_scoreN.open(result_dir + "/pixel_scoreN.txt", std::ios::out);
    ofs_pixel_scoreA.open(result_dir + "/pixel_scoreA.txt", std::ios::out);
    /*************************************************************************************/
    heatmapN_norm_dir = result_dir + "/testN-norm";  fs::create_directories(heatmapN_norm_dir);
    ofs.open(result_dir + "/timeN.txt", std::ios::out);
    ofs_image_score.open(result_dir + "/image_scoreN.txt", std::ios::out);
    while (dataloaderN(dataN)){
        
        image = std::get<0>(dataN).to(device);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();

        feature = model->forward(image);
        std::tie(score_map, anomaly_score) = PatchCoreScore(feature, coreset, vm["num_knn"].as<size_t>());
        score_map_up = F::interpolate(score_map, F::InterpolateFuncOptions().size(std::vector<long int>({image.size(2), image.size(3)})).mode(torch::kBilinear).align_corners(false));
        score_map_up = gaussian_filter_2d(score_map_up, 4.0);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;

        s = score_map_up.detach().to(torch::kCPU).view({-1}).contiguous();
        p = s.data_ptr<float>();
        for (long int i = 0; i < s.numel(); i++) ofs_pixel_scoreN << p[i] << std::endl;
        
        if (score_map_up_min > score_map_up.min().item<float>()) score_map_up_min = score_map_up.min().item<float>();
        if (score_map_up_max < score_map_up.max().item<float>()) score_map_up_max = score_map_up.max().item<float>();
        score_map_up = (score_map_up - score_map_up.min()) / (score_map_up.max() - score_map_up.min() + 1e-12);
        ave_anomaly_scoreN += anomaly_score.item<float>();
        ave_timeN += seconds;

        std::cout << '<' << std::get<1>(dataN).at(0) << "> anomaly_score:" << anomaly_score.item<float>() << " (time:" << seconds << ')' << std::endl;
        ofs << '<' << std::get<1>(dataN).at(0) << "> time:" << seconds << std::endl;
        ofs_image_score << anomaly_score.item<float>() << std::endl;

        fname = heatmapN_norm_dir + '/' + std::get<1>(dataN).at(0);
        heatmap = visualizer::create_heatmap(score_map_up.detach(), /*range=*/{0.0, 1.0});
        visualizer::save_image(heatmap.detach(), fname, /*range=*/{0.0, 1.0}, /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_anomaly_scoreN = ave_anomaly_scoreN / (double)datasetN.size();
    ave_timeN = ave_timeN / (double)datasetN.size();

    // (7) Average Output
    std::cout << "<All> anomaly_score:" << ave_anomaly_scoreN << " (time:" << ave_timeN << ')' << std::endl;
    ofs << "<All> time:" << ave_timeN << std::endl;
    ofs.close();
    ofs_image_score.close();

    // (8) Tensor Forward for test anomaly
    heatmapA_norm_dir = result_dir + "/testA-norm";  fs::create_directories(heatmapA_norm_dir);
    ofs.open(result_dir + "/timeA.txt", std::ios::out);
    ofs_image_score.open(result_dir + "/image_scoreA.txt", std::ios::out);
    while (dataloaderA(dataA)){
        
        image = std::get<0>(dataA).to(device);
        gt = std::get<1>(dataA).to(device);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();

        feature = model->forward(image);
        std::tie(score_map, anomaly_score) = PatchCoreScore(feature, coreset, vm["num_knn"].as<size_t>());
        score_map_up = F::interpolate(score_map, F::InterpolateFuncOptions().size(std::vector<long int>({image.size(2), image.size(3)})).mode(torch::kBilinear).align_corners(false));
        score_map_up = gaussian_filter_2d(score_map_up, 4.0);
        
        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;
        
        s = score_map_up.detach().to(torch::kCPU).view({-1}).contiguous();
        mask = (gt > 0.5).detach().to(torch::kCPU).view({-1}).contiguous();
        s_pos = s.masked_select(mask);
        s_neg = s.masked_select(mask.logical_not());
        p_pos = s_pos.data_ptr<float>();
        p_neg = s_neg.data_ptr<float>();
        for (long int i = 0; i < s_pos.numel(); i++) ofs_pixel_scoreA << p_pos[i] << std::endl;
        for (long int i = 0; i < s_neg.numel(); i++) ofs_pixel_scoreN << p_neg[i] << std::endl;
        
        if (score_map_up_min > score_map_up.min().item<float>()) score_map_up_min = score_map_up.min().item<float>();
        if (score_map_up_max < score_map_up.max().item<float>()) score_map_up_max = score_map_up.max().item<float>();
        score_map_up = (score_map_up - score_map_up.min()) / (score_map_up.max() - score_map_up.min() + 1e-12);
        ave_anomaly_scoreA += anomaly_score.item<float>();
        ave_timeA += seconds;

        std::cout << '<' << std::get<2>(dataA).at(0) << "> anomaly_score:" << anomaly_score.item<float>() << " (time:" << seconds << ')' << std::endl;
        ofs << '<' << std::get<2>(dataA).at(0) << "> time:" << seconds << std::endl;
        ofs_image_score << anomaly_score.item<float>() << std::endl;

        fname = heatmapA_norm_dir + '/' + std::get<2>(dataA).at(0);
        heatmap = visualizer::create_heatmap(score_map_up.detach(), /*range=*/{0.0, 1.0});
        visualizer::save_image(heatmap.detach(), fname, /*range=*/{0.0, 1.0}, /*cols=*/1, /*padding=*/0);

    }

    // (9) Calculate Average
    ave_anomaly_scoreA = ave_anomaly_scoreA / (double)datasetA.size();
    ave_timeA = ave_timeA / (double)datasetA.size();

    // (10) Average Output
    std::cout << "<All> anomaly_score:" << ave_anomaly_scoreA << " (time:" << ave_timeA << ')' << std::endl;
    ofs << "<All> time:" << ave_timeA << std::endl;
    ofs.close();
    ofs_image_score.close();
    /*************************************************************************************/
    ofs_pixel_scoreN.close();
    ofs_pixel_scoreA.close();

    // (11) Tensor Forward for test normal
    heatmapN_dir = result_dir + "/testN";  fs::create_directories(heatmapN_dir);
    while (dataloaderN(dataN)){

        image = std::get<0>(dataN).to(device);
        feature = model->forward(image);
        std::tie(score_map, anomaly_score) = PatchCoreScore(feature, coreset, vm["num_knn"].as<size_t>());
        score_map_up = F::interpolate(score_map, F::InterpolateFuncOptions().size(std::vector<long int>({image.size(2), image.size(3)})).mode(torch::kBilinear).align_corners(false));
        score_map_up = gaussian_filter_2d(score_map_up, 4.0);
        score_map_up = (score_map_up - score_map_up_min) / (score_map_up_max - score_map_up_min + 1e-12);

        fname = heatmapN_dir + '/' + std::get<1>(dataN).at(0);
        heatmap = visualizer::create_heatmap(score_map_up.detach(), /*range=*/{0.0, 1.0});
        visualizer::save_image(heatmap.detach(), fname, /*range=*/{0.0, 1.0}, /*cols=*/1, /*padding=*/0);

    }

    // (12) Tensor Forward for test anomaly
    heatmapA_dir = result_dir + "/testA";  fs::create_directories(heatmapA_dir);
    while (dataloaderA(dataA)){

        image = std::get<0>(dataA).to(device);
        feature = model->forward(image);
        std::tie(score_map, anomaly_score) = PatchCoreScore(feature, coreset, vm["num_knn"].as<size_t>());
        score_map_up = F::interpolate(score_map, F::InterpolateFuncOptions().size(std::vector<long int>({image.size(2), image.size(3)})).mode(torch::kBilinear).align_corners(false));
        score_map_up = gaussian_filter_2d(score_map_up, 4.0);
        score_map_up = (score_map_up - score_map_up_min) / (score_map_up_max - score_map_up_min + 1e-12);

        fname = heatmapA_dir + '/' + std::get<2>(dataA).at(0);
        heatmap = visualizer::create_heatmap(score_map_up.detach(), /*range=*/{0.0, 1.0});
        visualizer::save_image(heatmap.detach(), fname, /*range=*/{0.0, 1.0}, /*cols=*/1, /*padding=*/0);

    }

    // End Processing
    return;

}


// ---------------------------------------------
// Function to Calculate Anomaly Score
// ---------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> PatchCoreScore(torch::Tensor feature, torch::Tensor coreset, long int K){

    // feature: {1,D,H,W}
    // coreset: {Nc,D}

    long int dim, height, width, feature_size;
    torch::Tensor x, dist, knn_dist, _, pixel_score, score_map;
    torch::Tensor max_idx, max_d, alpha, beta, image_score;
    
    dim = feature.size(1);
    height = feature.size(2);
    width = feature.size(3);
    feature_size = height * width;

    x = feature.permute({0, 2, 3, 1}).contiguous().view({feature_size, dim});  // {H*W,D}
    dist = torch::cdist(x.unsqueeze(0), coreset.unsqueeze(0), 2.0).squeeze(0);  // {H*W,Nc}
    std::tie(knn_dist, _) = dist.topk(/*k=*/K, /*dim=*/1, /*largest=*/false, /*sorted=*/true);  // {H*W,K}
    pixel_score = knn_dist.select(1, 0);  // {H*W}
    score_map = pixel_score.view({1, 1, height, width});  // {1,1,H,W}

    max_idx = pixel_score.argmax();  // {}
    max_d = pixel_score.index({max_idx});  // {}
    alpha = torch::exp(max_d);  // {}
    beta = torch::exp(knn_dist.index({max_idx}).slice(/*dim=*/0, /*start=*/1)).sum();  // {}
    image_score = (1.0 - alpha / (beta + 1e-12)) * max_d;  // {}

    return {score_map, image_score};

}


// ---------------------------------------------
// Function to Apply Gaussian Filter
// ---------------------------------------------
torch::Tensor gaussian_filter_2d(torch::Tensor x, double sigma){

    int radius, k;
    torch::Tensor t, g1, g2, w, x_pad, out;

    radius = std::ceil(3.0 * sigma);
    k = 2 * radius + 1;

    t = torch::arange(-radius, radius + 1).to(torch::kFloat).to(x.device());
    g1 = torch::exp(-(t * t) / (2.0 * sigma * sigma));
    g1 = g1 / g1.sum();

    g2 = g1.unsqueeze(1).mm(g1.unsqueeze(0));
    g2 = g2 / g2.sum();

    w = g2.view({1, 1, k, k});
    x_pad = F::pad(x, F::PadFuncOptions({radius, radius, radius, radius}).mode(torch::kReflect));
    out = F::conv2d(x_pad, w, F::Conv2dFuncOptions().padding(0));

    return out;

}
