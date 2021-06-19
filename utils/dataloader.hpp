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

    
    /*******************************************************************************/
    /*                                   Data 1d                                   */
    /*******************************************************************************/

    // -----------------------------------------------------
    // namespace{DataLoader} -> class{Data1dFolderWithPaths}
    // -----------------------------------------------------
    class Data1dFolderWithPaths{
    private:
        datasets::Data1dFolderWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;
        size_t size;
        std::vector<size_t> idx;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        Data1dFolderWithPaths(){}
        Data1dFolderWithPaths(datasets::Data1dFolderWithPaths &dataset_, const size_t batch_size_=1, const bool shuffle_=false, const size_t num_workers_=0, const bool pin_memory_=false, const bool drop_last_=false);
        bool operator()(std::tuple<torch::Tensor, std::vector<std::string>> &data);
        void reset();
        size_t get_count_max();
    };
    
    // ----------------------------------------------------------
    // namespace{DataLoader} -> class{Data1dFolderPairWithPaths}
    // ----------------------------------------------------------
    class Data1dFolderPairWithPaths{
    private:
        datasets::Data1dFolderPairWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;
        size_t size;
        std::vector<size_t> idx;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        Data1dFolderPairWithPaths(){}
        Data1dFolderPairWithPaths(datasets::Data1dFolderPairWithPaths &dataset_, const size_t batch_size_=1, const bool shuffle_=false, const size_t num_workers_=0, const bool pin_memory_=false, const bool drop_last_=false);
        bool operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> &data);
        void reset();
        size_t get_count_max();
    };


    /*******************************************************************************/
    /*                                   Data 2d                                   */
    /*******************************************************************************/

    // -----------------------------------------------------
    // namespace{DataLoader} -> class{ImageFolderWithPaths}
    // -----------------------------------------------------
    class ImageFolderWithPaths{
    private:
        datasets::ImageFolderWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;
        size_t size;
        std::vector<size_t> idx;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        ImageFolderWithPaths(){}
        ImageFolderWithPaths(datasets::ImageFolderWithPaths &dataset_, const size_t batch_size_=1, const bool shuffle_=false, const size_t num_workers_=0, const bool pin_memory_=false, const bool drop_last_=false);
        bool operator()(std::tuple<torch::Tensor, std::vector<std::string>> &data);
        void reset();
        size_t get_count_max();
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
        bool pin_memory;
        bool drop_last;
        size_t size;
        std::vector<size_t> idx;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        ImageFolderPairWithPaths(){}
        ImageFolderPairWithPaths(datasets::ImageFolderPairWithPaths &dataset_, const size_t batch_size_=1, const bool shuffle_=false, const size_t num_workers_=0, const bool pin_memory_=false, const bool drop_last_=false);
        bool operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> &data);
        void reset();
        size_t get_count_max();
    };

    // --------------------------------------------------------------------------
    // namespace{DataLoader} -> class{ImageFolderPairAndRandomSamplingWithPaths}
    // --------------------------------------------------------------------------
    class ImageFolderPairAndRandomSamplingWithPaths{
    private:
        datasets::ImageFolderPairAndRandomSamplingWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;
        size_t size;
        std::vector<size_t> idx;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
        std::uniform_int_distribution<int> int_rand;
    public:
        ImageFolderPairAndRandomSamplingWithPaths(){}
        ImageFolderPairAndRandomSamplingWithPaths(datasets::ImageFolderPairAndRandomSamplingWithPaths &dataset_, const size_t batch_size_=1, const bool shuffle_=false, const size_t num_workers_=0, const bool pin_memory_=false, const bool drop_last_=false);
        bool operator()(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::string>> &data);
        void reset();
        size_t get_count_max();
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
        bool pin_memory;
        bool drop_last;
        size_t size;
        std::vector<size_t> idx;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        ImageFolderSegmentWithPaths(){}
        ImageFolderSegmentWithPaths(datasets::ImageFolderSegmentWithPaths &dataset_, const size_t batch_size_=1, const bool shuffle_=false, const size_t num_workers_=0, const bool pin_memory_=false, const bool drop_last_=false);
        bool operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> &data);
        void reset();
        size_t get_count_max();
    };

    // -----------------------------------------------------
    // namespace{DataLoader} -> class{ImageFolderClassesWithPaths}
    // -----------------------------------------------------
    class ImageFolderClassesWithPaths{
    private:
        datasets::ImageFolderClassesWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;
        size_t size;
        std::vector<size_t> idx;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        ImageFolderClassesWithPaths(){}
        ImageFolderClassesWithPaths(datasets::ImageFolderClassesWithPaths &dataset_, const size_t batch_size_=1, const bool shuffle_=false, const size_t num_workers_=0, const bool pin_memory_=false, const bool drop_last_=false);
        bool operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> &data);
        void reset();
        size_t get_count_max();
    };
    
    // ----------------------------------------------------
    // namespace{DataLoader} -> class{ImageFolderBBWithPaths}
    // ----------------------------------------------------
    class ImageFolderBBWithPaths{
    private:
        datasets::ImageFolderBBWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;
        size_t size;
        std::vector<size_t> idx;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
    public:
        ImageFolderBBWithPaths(){}
        ImageFolderBBWithPaths(datasets::ImageFolderBBWithPaths &dataset_, const size_t batch_size_=1, const bool shuffle_=false, const size_t num_workers_=0, const bool pin_memory_=false, const bool drop_last_=false);
        bool operator()(std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> &data);
        void reset();
        size_t get_count_max();
    };

}




#endif