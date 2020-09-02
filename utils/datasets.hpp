#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <string>
#include <tuple>
#include <vector>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
// For Original Header
#include "transforms.hpp"


// -----------------------
// namespace{datasets}
// -----------------------
namespace datasets{

    // Function Prototype
    cv::Mat RGB_Loader(std::string &path);
    cv::Mat Index_Loader(std::string &path);

    // ----------------------------------------------------
    // namespace{datasets} -> class{ImageFolderWithPaths}
    // ----------------------------------------------------
    class ImageFolderWithPaths{
    private:
        std::vector<transforms::Compose*> transform;
        std::vector<std::string> paths, fnames;
    public:
        ImageFolderWithPaths(){}
        ImageFolderWithPaths(const std::string root, std::vector<transforms::Compose*> &transform_);
        void get(const size_t index, std::tuple<torch::Tensor, std::string> &data);
        size_t size();
    };

    // ----------------------------------------------------
    // namespace{datasets} -> class{ImageFolderPairWithPaths}
    // ----------------------------------------------------
    class ImageFolderPairWithPaths{
    private:
        std::vector<transforms::Compose*> transformI, transformO;
        std::vector<std::string> paths1, paths2, fnames1, fnames2;
    public:
        ImageFolderPairWithPaths(){}
        ImageFolderPairWithPaths(const std::string root1, const std::string root2, std::vector<transforms::Compose*> &transformI_, std::vector<transforms::Compose*> &transformO_);
        void get(const size_t index, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string> &data);
        size_t size();
    };

    // ------------------------------------------------------------------------
    // namespace{datasets} -> class{ImageFolderPairAndRandomSamplingWithPaths}
    // ------------------------------------------------------------------------
    class ImageFolderPairAndRandomSamplingWithPaths{
    private:
        std::vector<transforms::Compose*> transformI, transformO, transform_rand;
        std::vector<std::string> paths1, paths2, paths_rand, fnames1, fnames2, fnames_rand;
    public:
        ImageFolderPairAndRandomSamplingWithPaths(){}
        ImageFolderPairAndRandomSamplingWithPaths(const std::string root1, const std::string root2, const std::string root_rand, std::vector<transforms::Compose*> &transformI_, std::vector<transforms::Compose*> &transformO_, std::vector<transforms::Compose*> &transform_rand_);
        void get(const size_t index, const size_t index_rand, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::string, std::string, std::string> &data);
        size_t size();
        size_t size_rand();
    };

    // ----------------------------------------------------
    // namespace{datasets} -> class{ImageFolderSegmentWithPaths}
    // ----------------------------------------------------
    class ImageFolderSegmentWithPaths{
    private:
        std::vector<transforms::Compose*> transformI, transformO;
        std::vector<std::string> paths1, paths2, fnames1, fnames2;
        std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette;
    public:
        ImageFolderSegmentWithPaths(){}
        ImageFolderSegmentWithPaths(const std::string root1, const std::string root2, std::vector<transforms::Compose*> &transformI_, std::vector<transforms::Compose*> &transformO_);
        void get(const size_t index, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> &data);
        size_t size();
    };

}



#endif