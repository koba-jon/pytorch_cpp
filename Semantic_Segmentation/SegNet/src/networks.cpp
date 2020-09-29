#include <tuple>
#include <vector>
#include <typeinfo>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;


// ----------------------------------------------------------------------
// struct{SegNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
SegNetImpl::SegNetImpl(po::variables_map &vm){

    constexpr size_t num_downs_ = 5;
    this->num_downs = num_downs_;
    
    size_t nc = vm["nc"].as<size_t>();
    size_t nz = vm["nz"].as<size_t>();
    size_t feature = vm["nf"].as<size_t>();
    size_t feature_out = vm["nf"].as<size_t>();
    std::vector<size_t> enc_features = {nc, feature, feature*2, feature*4, feature*8, nz};
    std::vector<size_t> dec_features = {nz, feature*8, feature*4, feature*2, feature, feature_out};
    std::vector<size_t> enc_n_layers = {2, 2, 2, 2, 2};
    std::vector<size_t> dec_n_layers = {2, 2, 2, 2, 1};
    std::vector<bool> enc_use_dropout;
    std::vector<bool> dec_use_dropout;
    if (vm["no_dropout"].as<bool>()){
        enc_use_dropout = {false, false, false, false, false};
        dec_use_dropout = {false, false, false, false, false};
    }
    else{
        enc_use_dropout = {false, false, true, true, true};
        dec_use_dropout = {true, true, true, false, false};
    }
    
    for (size_t i = 0; i < this->num_downs; i++){
        this->encoder->push_back(
            DownSamplingImpl(enc_features.at(i), enc_features.at(i + 1), enc_n_layers.at(i), enc_use_dropout.at(i))
        );
        this->decoder->push_back(
            UpSamplingImpl(dec_features.at(i), dec_features.at(i + 1), dec_n_layers.at(i), dec_use_dropout.at(i))
        );
    }
    register_module("encoder", this->encoder);
    register_module("decoder", this->decoder);

    this->classifier = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(feature_out, vm["class_num"].as<size_t>(), 3).stride(1).padding(1).bias(true))
    );
    register_module("classifier", this->classifier);

}


// ----------------------------------------------------------------------
// struct{SegNetImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor SegNetImpl::forward(torch::Tensor x){

    torch::Tensor feature, out;
    std::vector<torch::Tensor> indices;
    std::vector<std::vector<long int>> unpool_sizes;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<long int>> tensor_with_alpha;

    feature = x;
    for (size_t i = 0; i < this->num_downs; i++){
        tensor_with_alpha = this->encoder[i]->as<DownSamplingImpl>()->forward(feature);  // {IC,256,256} ===> {8F,8,8}
        feature = std::get<0>(tensor_with_alpha);
        indices.push_back(std::get<1>(tensor_with_alpha));
        unpool_sizes.push_back(std::get<2>(tensor_with_alpha));
    }
    for (size_t i = 0; i < this->num_downs; i++){
        feature = this->decoder[i]->as<UpSamplingImpl>()->forward(feature, indices.at(this->num_downs - i - 1), unpool_sizes.at(this->num_downs - i - 1));  // {8F,8,8} ===> {F,256,256}
    }

    out = this->classifier->forward(feature);  // {F,256,256} ===> {OC,256,256}
    out = F::log_softmax(out, /*dim=*/1);

    return out;
}


// ----------------------------------------------------------------------
// struct{DownSamplingImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
DownSamplingImpl::DownSamplingImpl(const size_t in_nc, const size_t out_nc, const size_t n_layers, const bool use_dropout){
    
    this->features = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 3).stride(1).padding(1).bias(true)),
        nn::BatchNorm2d(out_nc),
        nn::ReLU(nn::ReLUOptions().inplace(true))
    );
    for (size_t i = 1; i < n_layers; i++){
        this->features->push_back(nn::Conv2d(nn::Conv2dOptions(out_nc, out_nc, 3).stride(1).padding(1).bias(true)));
        this->features->push_back(nn::BatchNorm2d(out_nc));
        this->features->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    if (use_dropout){
        this->features->push_back(nn::Dropout(0.5));
    }
    register_module("features", this->features);

    this->pool = nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2));
    register_module("pool", this->pool);

}


// ----------------------------------------------------------------------
// struct{DownSamplingImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, std::vector<long int>> DownSamplingImpl::forward(torch::Tensor x){
    
    size_t sizes_size;
    torch::Tensor feature;
    std::vector<long int> sizes_list;
    std::tuple<torch::Tensor, torch::Tensor> tensor_with_indices;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<long int>> out;
    
    feature = this->features->forward(x);
    tensor_with_indices = this->pool->forward_with_indices(feature);
    sizes_size = feature.sizes().size();
    for (size_t i = 0; i < sizes_size; i++){
        sizes_list.push_back(feature.size(i));
    }
    out = {std::get<0>(tensor_with_indices), std::get<1>(tensor_with_indices), sizes_list};
    
    return out;
}


// ----------------------------------------------------------------------
// struct{UpSamplingImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
UpSamplingImpl::UpSamplingImpl(const size_t in_nc, const size_t out_nc, const size_t n_layers, const bool use_dropout){
    
    for (size_t i = 0; i < n_layers - 1; i++){
        this->features->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, in_nc, 3).stride(1).padding(1).bias(true)));
        this->features->push_back(nn::BatchNorm2d(in_nc));
        this->features->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    }
    this->features->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 3).stride(1).padding(1).bias(true)));
    this->features->push_back(nn::BatchNorm2d(out_nc));
    this->features->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
    if (use_dropout){
        this->features->push_back(nn::Dropout(0.5));
    }
    register_module("features", this->features);

    this->unpool = nn::MaxUnpool2d(nn::MaxUnpool2dOptions(/*kernel_size=*/2).stride(2));
    register_module("unpool", this->unpool);

}


// ----------------------------------------------------------------------
// struct{UpSamplingImpl}(nn::Module) -> function{forward}
// ----------------------------------------------------------------------
torch::Tensor UpSamplingImpl::forward(torch::Tensor x, torch::Tensor indices, const std::vector<long int> unpool_sizes){
    torch::Tensor feature = this->unpool->forward(x, indices, unpool_sizes);
    torch::Tensor out = this->features->forward(feature);
    return out;
}


// ----------------------------
// function{weights_init}
// ----------------------------
void weights_init(nn::Module &m){
    if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl))) {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::BatchNorm2d)) || (typeid(m) == typeid(nn::BatchNorm2dImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/1.0, /*std=*/0.02);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}
