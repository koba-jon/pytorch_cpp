#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <algorithm>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
// For Original Header
#include "transforms.hpp"
#include "datasets.hpp"

namespace fs = boost::filesystem;


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::ImageFolderWithPaths::ImageFolderWithPaths(const std::string root, std::vector<transforms::Compose*> &transform_){
    fs::path ROOT = fs::path(root);
    for (auto &p : boost::make_iterator_range(fs::directory_iterator(ROOT), {})){
        if (!fs::is_directory(p)){
            std::stringstream rpath, fname;
            rpath << p.path().string();
            fname << p.path().filename().string();
            this->paths.push_back(rpath.str());
            this->fnames.push_back(fname.str());
        }
    }
    sort(this->paths.begin(), this->paths.end());
    sort(this->fnames.begin(), this->fnames.end());
    this->transform = transform_;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderWithPaths} -> function{Loader}
// -------------------------------------------------------------------------
cv::Mat datasets::ImageFolderWithPaths::Loader(std::string &path){
    cv::Mat BGR, RGB;
    BGR = cv::imread(path, cv::IMREAD_COLOR); // path ===> color image {B,G,R}
    cv::cvtColor(BGR, RGB, cv::COLOR_BGR2RGB);  // {0,1,2} = {B,G,R} ===> {0,1,2} = {R,G,B}
    return RGB.clone();
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::ImageFolderWithPaths::get(const size_t index, std::tuple<torch::Tensor, std::string> &data){
    cv::Mat image_Mat = this->Loader(this->paths.at(index));
    torch::Tensor image = transforms::apply(this->transform, image_Mat);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    std::string fname = this->fnames.at(index);
    data = {image, fname};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::ImageFolderWithPaths::size(){
    return this->fnames.size();
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::ImageFolderPairWithPaths::ImageFolderPairWithPaths(const std::string root1, const std::string root2, std::vector<transforms::Compose*> &transformI_, std::vector<transforms::Compose*> &transformO_){

    fs::path ROOT = fs::path(root1);
    for (auto &p : boost::make_iterator_range(fs::directory_iterator(ROOT), {})){
        if (!fs::is_directory(p)){
            std::stringstream path1, fname;
            path1 << p.path().string();
            fname << p.path().filename().string();
            this->paths1.push_back(path1.str());
            this->fnames.push_back(fname.str());
        }
    }
    sort(this->paths1.begin(), this->paths1.end());
    sort(this->fnames.begin(), this->fnames.end());

    for (auto &f : this->fnames){
        std::string path2 = root2 + '/' + f;
        this->paths2.push_back(path2);
    }

    this->transformI = transformI_;
    this->transformO = transformO_;

}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairWithPaths} -> function{Loader}
// -------------------------------------------------------------------------
cv::Mat datasets::ImageFolderPairWithPaths::Loader(std::string &path){
    cv::Mat BGR, RGB;
    BGR = cv::imread(path, cv::IMREAD_COLOR); // path ===> color image {B,G,R}
    cv::cvtColor(BGR, RGB, cv::COLOR_BGR2RGB);  // {0,1,2} = {B,G,R} ===> {0,1,2} = {R,G,B}
    return RGB.clone();
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::ImageFolderPairWithPaths::get(const size_t index, std::tuple<torch::Tensor, torch::Tensor, std::string> &data){
    cv::Mat image_Mat1 = this->Loader(this->paths1.at(index));
    cv::Mat image_Mat2 = this->Loader(this->paths2.at(index));
    torch::Tensor image1 = transforms::apply(this->transformI, image_Mat1);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    torch::Tensor image2 = transforms::apply(this->transformO, image_Mat2);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    std::string fname = this->fnames.at(index);
    data = {image1, image2, fname};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::ImageFolderPairWithPaths::size(){
    return this->fnames.size();
}