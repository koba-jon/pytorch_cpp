#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <string>
#include <tuple>
#include <vector>
#include <random>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "datasets.hpp"


// -----------------------
// namespace{DataLoader}
// -----------------------
namespace DataLoader{

    // -----------------------------------------------------
    // namespace{DataLoader} -> class{ImageFolderWithPaths}
    // -----------------------------------------------------
    class ImageFolderWithPaths{
    private:
        datasets::ImageFolderWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        size_t size;
        std::vector<size_t> index;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        ImageFolderWithPaths(){}
        ImageFolderWithPaths(datasets::ImageFolderWithPaths &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_);
        bool operator()(std::tuple<torch::Tensor, std::vector<std::string>> &data);
    };

    // -----------------------------------------------------
    // namespace{DataLoader} -> class{ImageFolderPairWithPaths}
    // -----------------------------------------------------
    class ImageFolderPairWithPaths{
    private:
        datasets::ImageFolderPairWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        size_t size;
        std::vector<size_t> index;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        ImageFolderPairWithPaths(){}
        ImageFolderPairWithPaths(datasets::ImageFolderPairWithPaths &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_);
        bool operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> &data);
    };

    // -----------------------------------------------------
    // namespace{DataLoader} -> class{ImageFolderSegmentWithPaths}
    // -----------------------------------------------------
    class ImageFolderSegmentWithPaths{
    private:
        datasets::ImageFolderSegmentWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        size_t size;
        std::vector<size_t> index;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        ImageFolderSegmentWithPaths(){}
        ImageFolderSegmentWithPaths(datasets::ImageFolderSegmentWithPaths &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_);
        bool operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> &data);
    };

}




#endif