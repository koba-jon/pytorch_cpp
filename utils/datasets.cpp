#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <png++/png.hpp>
// For Original Header
#include "transforms.hpp"
#include "datasets.hpp"

namespace fs = boost::filesystem;


// -----------------------------------------------
// namespace{datasets} -> function{collect}
// -----------------------------------------------
void datasets::collect(const std::string root, const std::string sub, std::vector<std::string> &paths, std::vector<std::string> &fnames){
    fs::path ROOT = fs::path(root);
    for (auto &p : boost::make_iterator_range(fs::directory_iterator(ROOT), {})){
        if (!fs::is_directory(p)){
            std::stringstream rpath, fname;
            rpath << p.path().string();
            fname << p.path().filename().string();
            paths.push_back(rpath.str());
            fnames.push_back(sub + fname.str());
        }
        else{
            std::stringstream subsub;
            subsub << p.path().leaf().string();
            datasets::collect(root + '/' + subsub.str(), sub + subsub.str() + '/', paths, fnames);
        }
    }
    return;
}


// -----------------------------------------------
// namespace{datasets} -> function{Data1d_Loader}
// -----------------------------------------------
torch::Tensor datasets::Data1d_Loader(std::string &path){

    float data_one;
    std::ifstream ifs;
    std::vector<float> data_src;
    torch::Tensor data;

    // Get Data
    ifs.open(path);
    while (1){
        ifs >> data_one;
        if (ifs.eof()) break;
        data_src.push_back(data_one);
    }
    ifs.close();

    // Get Tensor
    data = torch::from_blob(data_src.data(), {(long int)data_src.size()}, torch::kFloat).clone();

    return data;

}


// -----------------------------------------------
// namespace{datasets} -> function{RGB_Loader}
// -----------------------------------------------
cv::Mat datasets::RGB_Loader(std::string &path){
    cv::Mat BGR, RGB;
    BGR = cv::imread(path, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);  // path ===> color image {B,G,R}
    if (BGR.empty()) {
        std::cerr << "Error : Couldn't open the image '" << path << "'." << std::endl;
        std::exit(1);
    }
    cv::cvtColor(BGR, RGB, cv::COLOR_BGR2RGB);  // {0,1,2} = {B,G,R} ===> {0,1,2} = {R,G,B}
    return RGB.clone();
}


// -----------------------------------------------
// namespace{datasets} -> function{Index_Loader}
// -----------------------------------------------
cv::Mat datasets::Index_Loader(std::string &path){

    size_t i, j;    
    size_t width, height;
    cv::Mat Index;

    png::image<png::index_pixel> Index_png(path);  // path ===> index image

    width = Index_png.get_width();
    height = Index_png.get_height();
    Index = cv::Mat(cv::Size(width, height), CV_32SC1);
    for (j = 0; j < height; j++){
        for (i = 0; i < width; i++){
            Index.at<int>(j, i) = (int)Index_png[j][i];
        }
    }

    return Index.clone();

}


// ----------------------------------------------------
// namespace{datasets} -> function{BoundingBox_Loader}
// ----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> datasets::BoundingBox_Loader(std::string &path){
    
    FILE *fp;
    int state;
    long int id_data;
    float cx_data, cy_data, w_data, h_data;
    torch::Tensor id, cx, cy, w, h, coord;
    torch::Tensor ids, coords;
    std::tuple<torch::Tensor, torch::Tensor> BBs;

    if ((fp = fopen(path.c_str(), "r")) == NULL){
        std::cerr << "Error : Couldn't open the file '" << path << "'." << std::endl;
        std::exit(1);
    }

    state = 0;
    while (fscanf(fp, "%ld %f %f %f %f", &id_data, &cx_data, &cy_data, &w_data, &h_data) != EOF){

        id = torch::full({1}, id_data, torch::TensorOptions().dtype(torch::kLong));  // id{1}
        cx = torch::full({1, 1}, cx_data, torch::TensorOptions().dtype(torch::kFloat));  // cx{1,1}
        cy = torch::full({1, 1}, cy_data, torch::TensorOptions().dtype(torch::kFloat));  // cy{1,1}
        w = torch::full({1, 1}, w_data, torch::TensorOptions().dtype(torch::kFloat));  // w{1,1}
        h = torch::full({1, 1}, h_data, torch::TensorOptions().dtype(torch::kFloat));  // h{1,1}
        coord = torch::cat({cx, cy, w, h}, /*dim=*/1);  // cx{1,1} + cy{1,1} + w{1,1} + h{1,1} ===> coord{1,4}
        
        switch (state){
            case 0:
                ids = id;  // id{1} ===> ids{1}
                coords = coord;  // coord{1,4} ===> coords{1,4}
                state = 1;
                break;
            default:
                ids = torch::cat({ids, id}, /*dim=*/0);  // ids{i} + id{1} ===> ids{i+1}
                coords = torch::cat({coords, coord}, /*dim=*/0);  // coords{i,4} + coord{1,4} ===> coords{i+1,4}
        }

    }
    fclose(fp);

    if (ids.numel() > 0){
        ids = ids.contiguous().detach().clone();
        coords = coords.contiguous().detach().clone();
    }
    BBs = {ids, coords};  // {BB_n} (ids), {BB_n,4} (coordinates)

    return BBs;

}



/*******************************************************************************/
/*                                   Data 1d                                   */
/*******************************************************************************/


// -------------------------------------------------------------------------
// namespace{datasets} -> class{Data1dFolderWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::Data1dFolderWithPaths::Data1dFolderWithPaths(const std::string root, std::vector<transforms_Compose> &transform_){
    datasets::collect(root, "", this->paths, this->fnames);
    std::sort(this->paths.begin(), this->paths.end());
    std::sort(this->fnames.begin(), this->fnames.end());
    this->transform = transform_;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{Data1dFolderWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::Data1dFolderWithPaths::get(const size_t idx, std::tuple<torch::Tensor, std::string> &data){
    torch::Tensor tensor_in = datasets::Data1d_Loader(this->paths.at(idx));
    torch::Tensor tensor_out = transforms::applyT(this->transform, tensor_in);  // Tensor Data ==={Normalize,etc.}===> Tensor Data
    std::string fname = this->fnames.at(idx);
    data = {tensor_out.detach().clone(), fname};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{Data1dFolderWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::Data1dFolderWithPaths::size(){
    return this->fnames.size();
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{Data1dFolderPairWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::Data1dFolderPairWithPaths::Data1dFolderPairWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_){

    datasets::collect(root1, "", this->paths1, this->fnames1);
    std::sort(this->paths1.begin(), this->paths1.end());
    std::sort(this->fnames1.begin(), this->fnames1.end());

    datasets::collect(root2, "", this->paths2, this->fnames2);
    std::sort(this->paths2.begin(), this->paths2.end());
    std::sort(this->fnames2.begin(), this->fnames2.end());

    this->transformI = transformI_;
    this->transformO = transformO_;

}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{Data1dFolderPairWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::Data1dFolderPairWithPaths::get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string> &data){
    torch::Tensor tensor1_in = datasets::Data1d_Loader(this->paths1.at(idx));
    torch::Tensor tensor2_in = datasets::Data1d_Loader(this->paths2.at(idx));
    torch::Tensor tensor1_out = transforms::applyT(this->transformI, tensor1_in);  // Tensor Data ==={Normalize,etc.}===> Tensor Data
    torch::Tensor tensor2_out = transforms::applyT(this->transformO, tensor2_in);  // Tensor Data ==={Normalize,etc.}===> Tensor Data
    std::string fname1 = this->fnames1.at(idx);
    std::string fname2 = this->fnames2.at(idx);
    data = {tensor1_out.detach().clone(), tensor2_out.detach().clone(), fname1, fname2};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{Data1dFolderPairWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::Data1dFolderPairWithPaths::size(){
    return this->fnames1.size();
}



/*******************************************************************************/
/*                                   Data 2d                                   */
/*******************************************************************************/


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::ImageFolderWithPaths::ImageFolderWithPaths(const std::string root, std::vector<transforms_Compose> &transform_){
    datasets::collect(root, "", this->paths, this->fnames);
    std::sort(this->paths.begin(), this->paths.end());
    std::sort(this->fnames.begin(), this->fnames.end());
    this->transform = transform_;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::ImageFolderWithPaths::get(const size_t idx, std::tuple<torch::Tensor, std::string> &data){
    cv::Mat image_Mat = datasets::RGB_Loader(this->paths.at(idx));
    torch::Tensor image = transforms::apply(this->transform, image_Mat);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    std::string fname = this->fnames.at(idx);
    data = {image.detach().clone(), fname};
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
datasets::ImageFolderPairWithPaths::ImageFolderPairWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_){

    datasets::collect(root1, "", this->paths1, this->fnames1);
    std::sort(this->paths1.begin(), this->paths1.end());
    std::sort(this->fnames1.begin(), this->fnames1.end());

    datasets::collect(root2, "", this->paths2, this->fnames2);
    std::sort(this->paths2.begin(), this->paths2.end());
    std::sort(this->fnames2.begin(), this->fnames2.end());

    this->transformI = transformI_;
    this->transformO = transformO_;

}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::ImageFolderPairWithPaths::get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string> &data){
    cv::Mat image_Mat1 = datasets::RGB_Loader(this->paths1.at(idx));
    cv::Mat image_Mat2 = datasets::RGB_Loader(this->paths2.at(idx));
    torch::Tensor image1 = transforms::apply(this->transformI, image_Mat1);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    torch::Tensor image2 = transforms::apply(this->transformO, image_Mat2);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    std::string fname1 = this->fnames1.at(idx);
    std::string fname2 = this->fnames2.at(idx);
    data = {image1.detach().clone(), image2.detach().clone(), fname1, fname2};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::ImageFolderPairWithPaths::size(){
    return this->fnames1.size();
}


// ----------------------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairAndRandomSamplingWithPaths} -> constructor
// ----------------------------------------------------------------------------------------
datasets::ImageFolderPairAndRandomSamplingWithPaths::ImageFolderPairAndRandomSamplingWithPaths(const std::string root1, const std::string root2, const std::string root_rand, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_, std::vector<transforms_Compose> &transform_rand_){

    datasets::collect(root1, "", this->paths1, this->fnames1);
    std::sort(this->paths1.begin(), this->paths1.end());
    std::sort(this->fnames1.begin(), this->fnames1.end());

    datasets::collect(root2, "", this->paths2, this->fnames2);
    std::sort(this->paths2.begin(), this->paths2.end());
    std::sort(this->fnames2.begin(), this->fnames2.end());

    datasets::collect(root_rand, "", this->paths_rand, this->fnames_rand);
    std::sort(this->paths_rand.begin(), this->paths_rand.end());
    std::sort(this->fnames_rand.begin(), this->fnames_rand.end());

    this->transformI = transformI_;
    this->transformO = transformO_;
    this->transform_rand = transform_rand_;

}


// ------------------------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairAndRandomSamplingWithPaths} -> function{get}
// ------------------------------------------------------------------------------------------
void datasets::ImageFolderPairAndRandomSamplingWithPaths::get(const size_t idx, const size_t idx_rand, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::string, std::string, std::string> &data){
    cv::Mat image_Mat1 = datasets::RGB_Loader(this->paths1.at(idx));
    cv::Mat image_Mat2 = datasets::RGB_Loader(this->paths2.at(idx));
    cv::Mat image_Mat_rand = datasets::RGB_Loader(this->paths_rand.at(idx_rand));
    torch::Tensor image1 = transforms::apply(this->transformI, image_Mat1);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    torch::Tensor image2 = transforms::apply(this->transformO, image_Mat2);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    torch::Tensor image_rand = transforms::apply(this->transform_rand, image_Mat_rand);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    std::string fname1 = this->fnames1.at(idx);
    std::string fname2 = this->fnames2.at(idx);
    std::string fname_rand = this->fnames_rand.at(idx_rand);
    data = {image1.detach().clone(), image2.detach().clone(), image_rand.detach().clone(), fname1, fname2, fname_rand};
    return;
}


// -------------------------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairAndRandomSamplingWithPaths} -> function{size}
// -------------------------------------------------------------------------------------------
size_t datasets::ImageFolderPairAndRandomSamplingWithPaths::size(){
    return this->fnames1.size();
}


// -----------------------------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderPairAndRandomSamplingWithPaths} -> function{size_rand}
// -----------------------------------------------------------------------------------------------
size_t datasets::ImageFolderPairAndRandomSamplingWithPaths::size_rand(){
    return this->fnames_rand.size();
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderSegmentWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::ImageFolderSegmentWithPaths::ImageFolderSegmentWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_){

    datasets::collect(root1, "", this->paths1, this->fnames1);
    std::sort(this->paths1.begin(), this->paths1.end());
    std::sort(this->fnames1.begin(), this->fnames1.end());

    std::string f_png;
    std::string::size_type pos;
    for (auto &f : this->fnames1){
        if ((pos = f.find_last_of(".")) == std::string::npos){
            f_png = f + ".png";
        }
        else{
            f_png = f.substr(0, pos) + ".png";
        }
        std::string path2 = root2 + '/' + f_png;
        this->fnames2.push_back(f_png);
        this->paths2.push_back(path2);
    }

    this->transformI = transformI_;
    this->transformO = transformO_;

    png::image<png::index_pixel> Index_png(paths2.at(0));
    png::palette pal = Index_png.get_palette();
    for (auto &p : pal){
        this->label_palette.push_back({(unsigned char)p.red, (unsigned char)p.green, (unsigned char)p.blue});
    }

}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderSegmentWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::ImageFolderSegmentWithPaths::get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> &data){
    cv::Mat image_Mat1 = datasets::RGB_Loader(this->paths1.at(idx));
    cv::Mat image_Mat2 = datasets::Index_Loader(this->paths2.at(idx));
    torch::Tensor image1 = transforms::apply(this->transformI, image_Mat1);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    torch::Tensor image2 = transforms::apply(this->transformO, image_Mat2);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    std::string fname1 = this->fnames1.at(idx);
    std::string fname2 = this->fnames2.at(idx);
    data = {image1.detach().clone(), image2.detach().clone(), fname1, fname2, this->label_palette};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderSegmentWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::ImageFolderSegmentWithPaths::size(){
    return this->fnames1.size();
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderClassesWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::ImageFolderClassesWithPaths::ImageFolderClassesWithPaths(const std::string root, std::vector<transforms_Compose> &transform_, const std::vector<std::string> class_names){
    
    std::string class_name, class_root;
    
    for (size_t i = 0; i < class_names.size(); i++){
        
        std::vector<std::string> paths_tmp, fnames_tmp;
        class_name = class_names.at(i);
        class_root = root + '/' + class_name;
        
        datasets::collect(class_root, class_name + '/', paths_tmp, fnames_tmp);
        std::sort(paths_tmp.begin(), paths_tmp.end());
        std::sort(fnames_tmp.begin(), fnames_tmp.end());
        std::copy(paths_tmp.begin(), paths_tmp.end(), std::back_inserter(this->paths));
        std::copy(fnames_tmp.begin(), fnames_tmp.end(), std::back_inserter(this->fnames));

        std::vector<size_t> class_ids_tmp(paths_tmp.size(), i);
        std::copy(class_ids_tmp.begin(), class_ids_tmp.end(), std::back_inserter(this->class_ids));

    }

    this->transform = transform_;

}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderClassesWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::ImageFolderClassesWithPaths::get(const size_t idx, std::tuple<torch::Tensor,  torch::Tensor, std::string> &data){
    cv::Mat image_Mat = datasets::RGB_Loader(this->paths.at(idx));
    torch::Tensor image = transforms::apply(this->transform, image_Mat);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    torch::Tensor class_id = torch::full({}, (long int)this->class_ids.at(idx), torch::TensorOptions().dtype(torch::kLong));
    std::string fname = this->fnames.at(idx);
    data = {image.detach().clone(), class_id.detach().clone(), fname};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderClassesWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::ImageFolderClassesWithPaths::size(){
    return this->fnames.size();
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderBBWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::ImageFolderBBWithPaths::ImageFolderBBWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformBB_, std::vector<transforms_Compose> &transformI_){

    datasets::collect(root1, "", this->paths1, this->fnames1);
    std::sort(this->paths1.begin(), this->paths1.end());
    std::sort(this->fnames1.begin(), this->fnames1.end());

    std::string f_txt;
    std::string::size_type pos;
    for (auto &f : this->fnames1){
        if ((pos = f.find_last_of(".")) == std::string::npos){
            f_txt = f + ".txt";
        }
        else{
            f_txt = f.substr(0, pos) + ".txt";
        }
        std::string path2 = root2 + '/' + f_txt;
        this->fnames2.push_back(f_txt);
        this->paths2.push_back(path2);
    }

    this->transformBB = transformBB_;
    this->transformI = transformI_;

}


// --------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderBBWithPaths} -> function{deepcopy}
// --------------------------------------------------------------------------
void datasets::ImageFolderBBWithPaths::deepcopy(cv::Mat &data_in1, std::tuple<torch::Tensor, torch::Tensor> &data_in2, cv::Mat &data_out1, std::tuple<torch::Tensor, torch::Tensor> &data_out2){
    data_in1.copyTo(data_out1);
    if (std::get<0>(data_in2).numel() > 0){
        data_out2 = {std::get<0>(data_in2).clone(), std::get<1>(data_in2).clone()};
    }
    else{
        data_out2 = {torch::Tensor(), torch::Tensor()};
    }
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderBBWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::ImageFolderBBWithPaths::get(const size_t idx, std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>, std::string, std::string> &data){

    cv::Mat image_Mat, image_Mat_mid;
    std::tuple<torch::Tensor, torch::Tensor> BBs, BBs_mid;
    torch::Tensor image;
    std::string fname1, fname2;

    image_Mat = datasets::RGB_Loader(this->paths1.at(idx));
    BBs = datasets::BoundingBox_Loader(this->paths2.at(idx));

    for (size_t i = 0; i < this->transformBB.size(); i++){
        this->deepcopy(image_Mat, BBs, image_Mat_mid, BBs_mid);
        this->transformBB.at(i)->forward(image_Mat_mid, BBs_mid, image_Mat, BBs);
    }

    image = transforms::apply(this->transformI, image_Mat);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    fname1 = this->fnames1.at(idx);
    fname2 = this->fnames2.at(idx);

    data = {image.detach().clone(), BBs, fname1, fname2};

    return;
    
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderBBWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::ImageFolderBBWithPaths::size(){
    return this->fnames1.size();
}
