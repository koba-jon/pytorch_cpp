#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <omp.h>
// For Original Header
#include "datasets.hpp"
#include "dataloader.hpp"


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderWithPaths} -> constructor
// --------------------------------------------------------------------
DataLoader::ImageFolderWithPaths::ImageFolderWithPaths(datasets::ImageFolderWithPaths &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_){

    this->dataset = dataset_;
    this->batch_size = batch_size_;
    this->shuffle = shuffle_;
    this->num_workers = num_workers_;

    this->size = this->dataset.size();
    this->idx = std::vector<size_t>(this->size);
    for (size_t i = 0; i < this->size; i++){
        this->idx.at(i) = i;
    }

    this->count = 0;
    this->count_max = std::ceil((float)this->size / (float)this->batch_size);

    this->mt.seed(std::rand());

}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderWithPaths} -> operator
// --------------------------------------------------------------------
bool DataLoader::ImageFolderWithPaths::operator()(std::tuple<torch::Tensor, std::vector<std::string>> &data){

    // (0) Initialization and Declaration
    size_t i;
    size_t idx_start = this->batch_size * this->count;
    size_t idx_end = std::min(this->size, (idx_start + this->batch_size));
    size_t mini_batch_size = idx_end - idx_start;
    torch::Tensor data1, tensor;
    std::vector<std::string> data2;
    std::tuple<torch::Tensor, std::string> group;
    std::tuple<torch::Tensor, std::string> *data_before;

    // (1) Special Handling on Certain Count
    if ((this->count == 0) && this->shuffle){
        std::shuffle(this->idx.begin(), this->idx.end(), this->mt);
    }
    else if(this->count == this->count_max){
        this->count = 0;
        return false;
    }

    // (2) Get Mini Batch Data
    data_before = new std::tuple<torch::Tensor, std::string>[mini_batch_size];
    // (2.1) Get Mini Batch Data using Single Thread
    if (this->num_workers == 0){
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }
    // (2.2) Get Mini Batch Data using Multi Thread
    else{
        omp_set_num_threads(this->num_workers);
        #pragma omp parallel for
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }

    // (3) Organize Data
    data1 = std::get<0>(data_before[0]);
    data1 = torch::unsqueeze(data1, /*dim=*/0);
    data2.push_back(std::get<1>(data_before[0]));
    for (i = 1; i < mini_batch_size; i++){
        group = data_before[i];
        tensor = std::get<0>(group);
        tensor = torch::unsqueeze(tensor, /*dim=*/0);  // {C,H,W} ===> {1,C,H,W}
        data1 = torch::cat({data1, tensor}, /*dim=*/0);  // {i,C,H,W} + {1,C,H,W} ===> {i+1,C,H,W}
        data2.push_back(std::get<1>(group));
    }

    // Post Processing
    this->count++;
    data = {data1.contiguous().detach().clone(), data2};  // {N,C,H,W} (images), {N} (fnames)
    delete[] data_before;

    // End Processing
    return true;
    
}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderPairWithPaths} -> constructor
// --------------------------------------------------------------------
DataLoader::ImageFolderPairWithPaths::ImageFolderPairWithPaths(datasets::ImageFolderPairWithPaths &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_){

    this->dataset = dataset_;
    this->batch_size = batch_size_;
    this->shuffle = shuffle_;
    this->num_workers = num_workers_;

    this->size = this->dataset.size();
    this->idx = std::vector<size_t>(this->size);
    for (size_t i = 0; i < this->size; i++){
        this->idx.at(i) = i;
    }

    this->count = 0;
    this->count_max = std::ceil((float)this->size / (float)this->batch_size);

    this->mt.seed(std::rand());

}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderPairWithPaths} -> operator
// --------------------------------------------------------------------
bool DataLoader::ImageFolderPairWithPaths::operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> &data){

    // (0) Initialization and Declaration
    size_t i;
    size_t idx_start = this->batch_size * this->count;
    size_t idx_end = std::min(this->size, (idx_start + this->batch_size));
    size_t mini_batch_size = idx_end - idx_start;
    torch::Tensor data1, data2, tensor1, tensor2;
    std::vector<std::string> data3, data4;
    std::tuple<torch::Tensor, torch::Tensor, std::string, std::string> group;
    std::tuple<torch::Tensor, torch::Tensor, std::string, std::string> *data_before;

    // (1) Special Handling on Certain Count
    if ((this->count == 0) && this->shuffle){
        std::shuffle(this->idx.begin(), this->idx.end(), this->mt);
    }
    else if(this->count == this->count_max){
        this->count = 0;
        return false;
    }

    // (2) Get Mini Batch Data
    data_before = new std::tuple<torch::Tensor, torch::Tensor, std::string, std::string>[mini_batch_size];
    // (2.1) Get Mini Batch Data using Single Thread
    if (this->num_workers == 0){
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }
    // (2.2) Get Mini Batch Data using Multi Thread
    else{
        omp_set_num_threads(this->num_workers);
        #pragma omp parallel for
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }

    // (3) Organize Data
    data1 = std::get<0>(data_before[0]);
    data1 = torch::unsqueeze(data1, /*dim=*/0);
    data2 = std::get<1>(data_before[0]);
    data2 = torch::unsqueeze(data2, /*dim=*/0);
    data3.push_back(std::get<2>(data_before[0]));
    data4.push_back(std::get<3>(data_before[0]));
    for (i = 1; i < mini_batch_size; i++){
        group = data_before[i];
        tensor1 = std::get<0>(group);
        tensor1 = torch::unsqueeze(tensor1, /*dim=*/0);  // {C,H,W} ===> {1,C,H,W}
        data1 = torch::cat({data1, tensor1}, /*dim=*/0);  // {i,C,H,W} + {1,C,H,W} ===> {i+1,C,H,W}
        tensor2 = std::get<1>(group);
        tensor2 = torch::unsqueeze(tensor2, /*dim=*/0);  // {C,H,W} ===> {1,C,H,W}
        data2 = torch::cat({data2, tensor2}, /*dim=*/0);  // {i,C,H,W} + {1,C,H,W} ===> {i+1,C,H,W}
        data3.push_back(std::get<2>(group));
        data4.push_back(std::get<3>(group));
    }

    // Post Processing
    this->count++;
    data = {data1.contiguous().detach().clone(), data2.contiguous().detach().clone(), data3, data4};  // {N,C,H,W} (images1), {N,C,H,W} (images2), {N} (fnames1), {N} (fnames2)
    delete[] data_before;

    // End Processing
    return true;
    
}


// ------------------------------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderPairAndRandomSamplingWithPaths} -> constructor
// ------------------------------------------------------------------------------------------
DataLoader::ImageFolderPairAndRandomSamplingWithPaths::ImageFolderPairAndRandomSamplingWithPaths(datasets::ImageFolderPairAndRandomSamplingWithPaths &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_){

    this->dataset = dataset_;
    this->batch_size = batch_size_;
    this->shuffle = shuffle_;
    this->num_workers = num_workers_;

    this->size = this->dataset.size();
    this->idx = std::vector<size_t>(this->size);
    for (size_t i = 0; i < this->size; i++){
        this->idx.at(i) = i;
    }

    this->count = 0;
    this->count_max = std::ceil((float)this->size / (float)this->batch_size);

    this->mt.seed(std::rand());
    this->int_rand = std::uniform_int_distribution<>(/*min=*/0, /*max=*/this->dataset.size_rand() - 1);

}


// ---------------------------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderPairAndRandomSamplingWithPaths} -> operator
// ---------------------------------------------------------------------------------------
bool DataLoader::ImageFolderPairAndRandomSamplingWithPaths::operator()(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::string>> &data){

    // (0) Initialization and Declaration
    size_t i;
    size_t idx_start = this->batch_size * this->count;
    size_t idx_end = std::min(this->size, (idx_start + this->batch_size));
    size_t mini_batch_size = idx_end - idx_start;
    torch::Tensor data1, data2, data3, tensor1, tensor2, tensor3;
    std::vector<std::string> data4, data5, data6;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::string, std::string, std::string> group;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::string, std::string, std::string> *data_before;

    // (1) Special Handling on Certain Count
    if ((this->count == 0) && this->shuffle){
        std::shuffle(this->idx.begin(), this->idx.end(), this->mt);
    }
    else if(this->count == this->count_max){
        this->count = 0;
        return false;
    }

    // (2) Get Mini Batch Data
    data_before = new std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::string, std::string, std::string>[mini_batch_size];
    // (2.1) Get Mini Batch Data using Single Thread
    if (this->num_workers == 0){
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), this->int_rand(this->mt), data_before[i]);
        }
    }
    // (2.2) Get Mini Batch Data using Multi Thread
    else{
        omp_set_num_threads(this->num_workers);
        #pragma omp parallel for
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), this->int_rand(this->mt), data_before[i]);
        }
    }

    // (3) Organize Data
    data1 = std::get<0>(data_before[0]);
    data1 = torch::unsqueeze(data1, /*dim=*/0);
    data2 = std::get<1>(data_before[0]);
    data2 = torch::unsqueeze(data2, /*dim=*/0);
    data3 = std::get<2>(data_before[0]);
    data3 = torch::unsqueeze(data3, /*dim=*/0);
    data4.push_back(std::get<3>(data_before[0]));
    data5.push_back(std::get<4>(data_before[0]));
    data6.push_back(std::get<5>(data_before[0]));
    for (i = 1; i < mini_batch_size; i++){
        group = data_before[i];
        tensor1 = std::get<0>(group);
        tensor1 = torch::unsqueeze(tensor1, /*dim=*/0);  // {C,H,W} ===> {1,C,H,W}
        data1 = torch::cat({data1, tensor1}, /*dim=*/0);  // {i,C,H,W} + {1,C,H,W} ===> {i+1,C,H,W}
        tensor2 = std::get<1>(group);
        tensor2 = torch::unsqueeze(tensor2, /*dim=*/0);  // {C,H,W} ===> {1,C,H,W}
        data2 = torch::cat({data2, tensor2}, /*dim=*/0);  // {i,C,H,W} + {1,C,H,W} ===> {i+1,C,H,W}
        tensor3 = std::get<2>(group);
        tensor3 = torch::unsqueeze(tensor3, /*dim=*/0);  // {C,H,W} ===> {1,C,H,W}
        data3 = torch::cat({data3, tensor3}, /*dim=*/0);  // {i,C,H,W} + {1,C,H,W} ===> {i+1,C,H,W}
        data4.push_back(std::get<3>(group));
        data5.push_back(std::get<4>(group));
        data6.push_back(std::get<5>(group));
    }

    // Post Processing
    this->count++;
    data = {data1.contiguous().detach().clone(), data2.contiguous().detach().clone(), data3.contiguous().detach().clone(), data4, data5, data6};  // {N,C,H,W} (images1), {N,C,H,W} (images2), {N,C,H,W} (images_rand), {N} (fnames1), {N} (fnames2), {N} (fnames_rand)
    delete[] data_before;

    // End Processing
    return true;
    
}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderSegmentWithPaths} -> constructor
// --------------------------------------------------------------------
DataLoader::ImageFolderSegmentWithPaths::ImageFolderSegmentWithPaths(datasets::ImageFolderSegmentWithPaths &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_){

    this->dataset = dataset_;
    this->batch_size = batch_size_;
    this->shuffle = shuffle_;
    this->num_workers = num_workers_;

    this->size = this->dataset.size();
    this->idx = std::vector<size_t>(this->size);
    for (size_t i = 0; i < this->size; i++){
        this->idx.at(i) = i;
    }

    this->count = 0;
    this->count_max = std::ceil((float)this->size / (float)this->batch_size);

    this->mt.seed(std::rand());

}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderSegmentWithPaths} -> operator
// --------------------------------------------------------------------
bool DataLoader::ImageFolderSegmentWithPaths::operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> &data){

    // (0) Initialization and Declaration
    size_t i;
    size_t idx_start = this->batch_size * this->count;
    size_t idx_end = std::min(this->size, (idx_start + this->batch_size));
    size_t mini_batch_size = idx_end - idx_start;
    torch::Tensor data1, data2, tensor1, tensor2;
    std::vector<std::string> data3, data4;
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> data5;
    std::tuple<torch::Tensor, torch::Tensor, std::string, std::string, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> group;
    std::tuple<torch::Tensor, torch::Tensor, std::string, std::string, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> *data_before;

    // (1) Special Handling on Certain Count
    if ((this->count == 0) && this->shuffle){
        std::shuffle(this->idx.begin(), this->idx.end(), this->mt);
    }
    else if(this->count == this->count_max){
        this->count = 0;
        return false;
    }

    // (2) Get Mini Batch Data
    data_before = new std::tuple<torch::Tensor, torch::Tensor, std::string, std::string, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>>[mini_batch_size];
    // (2.1) Get Mini Batch Data using Single Thread
    if (this->num_workers == 0){
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }
    // (2.2) Get Mini Batch Data using Multi Thread
    else{
        omp_set_num_threads(this->num_workers);
        #pragma omp parallel for
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }

    // (3) Organize Data
    data1 = std::get<0>(data_before[0]);
    data1 = torch::unsqueeze(data1, /*dim=*/0);
    data2 = std::get<1>(data_before[0]);
    data2 = torch::unsqueeze(data2, /*dim=*/0);
    data3.push_back(std::get<2>(data_before[0]));
    data4.push_back(std::get<3>(data_before[0]));
    data5 = std::get<4>(data_before[0]);
    for (i = 1; i < mini_batch_size; i++){
        group = data_before[i];
        tensor1 = std::get<0>(group);
        tensor1 = torch::unsqueeze(tensor1, /*dim=*/0);  // {C,H,W} ===> {1,C,H,W}
        data1 = torch::cat({data1, tensor1}, /*dim=*/0);  // {i,C,H,W} + {1,C,H,W} ===> {i+1,C,H,W}
        tensor2 = std::get<1>(group);
        tensor2 = torch::unsqueeze(tensor2, /*dim=*/0);  // {H,W} ===> {1,H,W}
        data2 = torch::cat({data2, tensor2}, /*dim=*/0);  // {i,H,W} + {1,H,W} ===> {i+1,H,W}
        data3.push_back(std::get<2>(group));
        data4.push_back(std::get<3>(group));
    }

    // Post Processing
    this->count++;
    data = {data1.contiguous().detach().clone(), data2.contiguous().detach().clone(), data3, data4, data5};  // {N,C,H,W} (images1), {N,H,W} (images2), {N} (fnames1), {N} (fnames2), {L} (label_palette)
    delete[] data_before;

    // End Processings
    return true;
    
}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderClassesWithPaths} -> constructor
// --------------------------------------------------------------------
DataLoader::ImageFolderClassesWithPaths::ImageFolderClassesWithPaths(datasets::ImageFolderClassesWithPaths &dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_){

    this->dataset = dataset_;
    this->batch_size = batch_size_;
    this->shuffle = shuffle_;
    this->num_workers = num_workers_;

    this->size = this->dataset.size();
    this->idx = std::vector<size_t>(this->size);
    for (size_t i = 0; i < this->size; i++){
        this->idx.at(i) = i;
    }

    this->count = 0;
    this->count_max = std::ceil((float)this->size / (float)this->batch_size);

    this->mt.seed(std::rand());

}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{ImageFolderClassesWithPaths} -> operator
// --------------------------------------------------------------------
bool DataLoader::ImageFolderClassesWithPaths::operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> &data){

    // (0) Initialization and Declaration
    size_t i;
    size_t idx_start = this->batch_size * this->count;
    size_t idx_end = std::min(this->size, (idx_start + this->batch_size));
    size_t mini_batch_size = idx_end - idx_start;
    torch::Tensor data1, data2, tensor1, tensor2;
    std::vector<std::string> data3;
    std::tuple<torch::Tensor, torch::Tensor, std::string> group;
    std::tuple<torch::Tensor, torch::Tensor, std::string> *data_before;

    // (1) Special Handling on Certain Count
    if ((this->count == 0) && this->shuffle){
        std::shuffle(this->idx.begin(), this->idx.end(), this->mt);
    }
    else if(this->count == this->count_max){
        this->count = 0;
        return false;
    }

    // (2) Get Mini Batch Data
    data_before = new std::tuple<torch::Tensor, torch::Tensor, std::string>[mini_batch_size];
    // (2.1) Get Mini Batch Data using Single Thread
    if (this->num_workers == 0){
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }
    // (2.2) Get Mini Batch Data using Multi Thread
    else{
        omp_set_num_threads(this->num_workers);
        #pragma omp parallel for
        for (i = 0; i < mini_batch_size; i++){
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }

    // (3) Organize Data
    data1 = std::get<0>(data_before[0]);
    data1 = torch::unsqueeze(data1, /*dim=*/0);
    data2 = std::get<1>(data_before[0]);
    data2 = torch::unsqueeze(data2, /*dim=*/0);
    data3.push_back(std::get<2>(data_before[0]));
    for (i = 1; i < mini_batch_size; i++){
        group = data_before[i];
        tensor1 = std::get<0>(group);
        tensor1 = torch::unsqueeze(tensor1, /*dim=*/0);  // {C,H,W} ===> {1,C,H,W}
        data1 = torch::cat({data1, tensor1}, /*dim=*/0);  // {i,C,H,W} + {1,C,H,W} ===> {i+1,C,H,W}
        tensor2 = std::get<1>(group);
        tensor2 = torch::unsqueeze(tensor2, /*dim=*/0);  // {} ===> {1}
        data2 = torch::cat({data2, tensor2}, /*dim=*/0);  // {i} + {1} ===> {i+1}
        data3.push_back(std::get<2>(group));
    }

    // Post Processing
    this->count++;
    data = {data1.contiguous().detach().clone(), data2.contiguous().detach().clone(), data3};  // {N,C,H,W} (images), {N} (class ids), {N} (fnames)
    delete[] data_before;

    // End Processing
    return true;
    
}