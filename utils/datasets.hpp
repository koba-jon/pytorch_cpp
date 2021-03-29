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
    void collect(const std::string root, const std::string sub, std::vector<std::string> &paths, std::vector<std::string> &fnames);
    torch::Tensor Data1d_Loader(std::string &path);
    cv::Mat RGB_Loader(std::string &path);
    cv::Mat Index_Loader(std::string &path);
    std::tuple<torch::Tensor, torch::Tensor> BoundingBox_Loader(std::string &path);


    /*******************************************************************************/
    /*                                   Data 1d                                   */
    /*******************************************************************************/

    // ----------------------------------------------------
    // namespace{datasets} -> class{Data1dFolderWithPaths}
    // ----------------------------------------------------
    class Data1dFolderWithPaths{
    private:
        std::vector<transforms_Compose> transform;
        std::vector<std::string> paths, fnames;
    public:
        Data1dFolderWithPaths(){}
        Data1dFolderWithPaths(const std::string root, std::vector<transforms_Compose> &transform_);
        void get(const size_t idx, std::tuple<torch::Tensor, std::string> &data);
        size_t size();
    };

    // --------------------------------------------------------
    // namespace{datasets} -> class{Data1dFolderPairWithPaths}
    // --------------------------------------------------------
    class Data1dFolderPairWithPaths{
    private:
        std::vector<transforms_Compose> transformI, transformO;
        std::vector<std::string> paths1, paths2, fnames1, fnames2;
    public:
        Data1dFolderPairWithPaths(){}
        Data1dFolderPairWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string> &data);
        size_t size();
    };


    /*******************************************************************************/
    /*                                   Data 2d                                   */
    /*******************************************************************************/

    // ----------------------------------------------------
    // namespace{datasets} -> class{ImageFolderWithPaths}
    // ----------------------------------------------------
    class ImageFolderWithPaths{
    private:
        std::vector<transforms_Compose> transform;
        std::vector<std::string> paths, fnames;
    public:
        ImageFolderWithPaths(){}
        ImageFolderWithPaths(const std::string root, std::vector<transforms_Compose> &transform_);
        void get(const size_t idx, std::tuple<torch::Tensor, std::string> &data);
        size_t size();
    };

    // ----------------------------------------------------
    // namespace{datasets} -> class{ImageFolderPairWithPaths}
    // ----------------------------------------------------
    class ImageFolderPairWithPaths{
    private:
        std::vector<transforms_Compose> transformI, transformO;
        std::vector<std::string> paths1, paths2, fnames1, fnames2;
    public:
        ImageFolderPairWithPaths(){}
        ImageFolderPairWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string> &data);
        size_t size();
    };

    // ------------------------------------------------------------------------
    // namespace{datasets} -> class{ImageFolderPairAndRandomSamplingWithPaths}
    // ------------------------------------------------------------------------
    class ImageFolderPairAndRandomSamplingWithPaths{
    private:
        std::vector<transforms_Compose> transformI, transformO, transform_rand;
        std::vector<std::string> paths1, paths2, paths_rand, fnames1, fnames2, fnames_rand;
    public:
        ImageFolderPairAndRandomSamplingWithPaths(){}
        ImageFolderPairAndRandomSamplingWithPaths(const std::string root1, const std::string root2, const std::string root_rand, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_, std::vector<transforms_Compose> &transform_rand_);
        void get(const size_t idx, const size_t idx_rand, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::string, std::string, std::string> &data);
        size_t size();
        size_t size_rand();
    };

    // ----------------------------------------------------
    // namespace{datasets} -> class{ImageFolderSegmentWithPaths}
    // ----------------------------------------------------
    class ImageFolderSegmentWithPaths{
    private:
        std::vector<transforms_Compose> transformI, transformO;
        std::vector<std::string> paths1, paths2, fnames1, fnames2;
        std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette;
    public:
        ImageFolderSegmentWithPaths(){}
        ImageFolderSegmentWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> &data);
        size_t size();
    };
    
    // ----------------------------------------------------------
    // namespace{datasets} -> class{ImageFolderClassesWithPaths}
    // ----------------------------------------------------------
    class ImageFolderClassesWithPaths{
    private:
        std::vector<transforms_Compose> transform;
        std::vector<std::string> paths, fnames;
        std::vector<size_t> class_ids;
    public:
        ImageFolderClassesWithPaths(){}
        ImageFolderClassesWithPaths(const std::string root, std::vector<transforms_Compose> &transform_, const std::vector<std::string> class_names);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string> &data);
        size_t size();
    };
    
    // ----------------------------------------------------
    // namespace{datasets} -> class{ImageFolderBBWithPaths}
    // ----------------------------------------------------
    class ImageFolderBBWithPaths{
    private:
        std::vector<transforms_Compose> transformBB, transformI;
        std::vector<std::string> paths1, paths2, fnames1, fnames2;
        void deepcopy(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2);
    public:
        ImageFolderBBWithPaths(){}
        ImageFolderBBWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformBB_, std::vector<transforms_Compose> &transformI_);
        void get(const size_t idx, std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>, std::string, std::string> &data);
        size_t size();
    };

}



#endif