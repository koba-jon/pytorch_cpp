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
        cv::Mat Loader(std::string &path);
        void get(const size_t index, std::tuple<torch::Tensor, std::string> &data);
        size_t size();
    };

    // ----------------------------------------------------
    // namespace{datasets} -> class{ImageFolderPairWithPaths}
    // ----------------------------------------------------
    class ImageFolderPairWithPaths{
    private:
        std::vector<transforms::Compose*> transform;
        std::vector<std::string> paths1, paths2, fnames;
    public:
        ImageFolderPairWithPaths(){}
        ImageFolderPairWithPaths(const std::string root1, const std::string root2, std::vector<transforms::Compose*> &transform_);
        cv::Mat Loader(std::string &path);
        void get(const size_t index, std::tuple<torch::Tensor, torch::Tensor, std::string> &data);
        size_t size();
    };

}



#endif