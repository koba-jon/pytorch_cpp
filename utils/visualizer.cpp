#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <png++/png.hpp>
// For Original Header
#include "visualizer.hpp"

// Define Namespace
namespace fs = std::filesystem;


// ----------------------------------------------------------
// namespace{visualizer} -> function{save_image}
// ----------------------------------------------------------
void visualizer::save_image(const torch::Tensor image, const std::string path, const std::pair<float, float> range, const size_t cols, const size_t padding, const size_t bits){

    // (0) Initialization and Declaration
    size_t i, j, k, l;
    size_t i_dev, j_dev;
    size_t width, height, channels, mini_batch_size;
    size_t width_out, height_out;
    size_t ncol, nrow;
    int mtype_in, mtype_out;
    cv::Mat float_mat, normal_mat, bit_mat;
    cv::Mat sample, output;
    std::vector<cv::Mat> samples;
    torch::Tensor tensor_sq, tensor_per, tensor_con;

    // (1) Get Tensor Size
    mini_batch_size = image.size(0);
    channels = image.size(1);
    height = image.size(2);
    width = image.size(3);

    // (2) Judge the number of channels and bits
    if (channels == 1){
        mtype_in = CV_32FC1;
        if (bits == 8){
            mtype_out = CV_8UC1;
        }
        else if (bits == 16){
            mtype_out = CV_16UC1;
        }
        else{
            std::cerr << "Error : Bits of the image to be saved is inappropriate." << std::endl;
            std::exit(1);
        }
    }
    else if (channels == 3){
        mtype_in = CV_32FC3;
        if (bits == 8){
            mtype_out = CV_8UC3;
        }
        else if (bits == 16){
            mtype_out = CV_16UC3;
        }
        else{
            std::cerr << "Error : Bits of the image to be saved is inappropriate." << std::endl;
            std::exit(1);
        }
    }
    else{
        std::cerr << "Error : Channels of the image to be saved is inappropriate." << std::endl;
        std::exit(1);
    }

    // (3) Add images to the array
    i = 0;
    samples = std::vector<cv::Mat>(mini_batch_size);
    auto mini_batch = image.clamp(/*min=*/range.first, /*max=*/range.second).to(torch::kCPU).chunk(mini_batch_size, /*dim=*/0);  // {N,C,H,W} ===> {1,C,H,W} + {1,C,H,W} + ...
    for (auto &tensor : mini_batch){
        tensor_sq = torch::squeeze(tensor, /*dim=*/0);  // {1,C,H,W} ===> {C,H,W}
        tensor_per = tensor_sq.permute({1, 2, 0});  // {C,H,W} ===> {H,W,C}
        tensor_con = tensor_per.contiguous();
        float_mat = cv::Mat(cv::Size(width, height), mtype_in, tensor_con.data_ptr<float>());  // torch::Tensor ===> cv::Mat
        normal_mat = (float_mat - range.first) / (float)(range.second - range.first);  // [range.first, range.second] ===> [0,1]
        bit_mat = normal_mat * (std::pow(2.0, bits) - 1.0);  // [0,1] ===> [0,255] or [0,65535]
        bit_mat.convertTo(sample, mtype_out);  // {32F} ===> {8U} or {16U}
        if (channels == 3){
            cv::cvtColor(sample, sample, cv::COLOR_RGB2BGR);  // {R,G,B} ===> {B,G,R}
        }
        sample.copyTo(samples.at(i));
        i++;
    }

    // (4) Output Image Information
    ncol = (mini_batch_size < cols) ? mini_batch_size : cols;
    width_out = width * ncol + padding * (ncol + 1);
    nrow = 1 + (mini_batch_size - 1) / ncol;
    height_out =  height * nrow + padding * (nrow + 1);

    // (5) Value Substitution for Output Image
    output = cv::Mat(cv::Size(width_out, height_out), mtype_out, cv::Scalar::all(0));
    for (l = 0; l < mini_batch_size; l++){
        sample = samples.at(l);
        i_dev = (l % ncol) * width + padding * (l % ncol + 1);
        j_dev = (l / ncol) * height + padding * (l / ncol + 1);
        for (j = 0; j < height; j++){
            for (i = 0; i < width; i++){
                for (k = 0; k < sample.elemSize(); k++){
                    output.data[(j + j_dev) * output.step + (i + i_dev) * output.elemSize() + k] = sample.data[j * sample.step + i * sample.elemSize() + k];
                }
            }
        }
    }

    // (6) Image Output
    cv::imwrite(path, output);

    // End Processing
    return;

}


// ----------------------------------------------------------
// namespace{visualizer} -> function{save_label}
// ----------------------------------------------------------
void visualizer::save_label(const torch::Tensor label, const std::string path, const std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette, const size_t cols, const size_t padding){

    // (0) Initialization and Declaration
    size_t i, j, k;
    size_t i_dev, j_dev;
    size_t width, height, mini_batch_size;
    size_t width_out, height_out;
    size_t ncol, nrow;
    unsigned char R, G, B;
    cv::Mat sample;
    std::vector<cv::Mat> samples;
    torch::Tensor tensor_sq, tensor_per, tensor_con;
    png::image<png::index_pixel> output;
    png::palette pal;
    std::tuple<unsigned char, unsigned char, unsigned char> pal_one;

    // (1) Get Tensor Size
    mini_batch_size = label.size(0);
    height = label.size(2);
    width = label.size(3);

    // (2) Add images to the array
    i = 0;
    samples = std::vector<cv::Mat>(mini_batch_size);
    auto mini_batch = label.to(torch::kCPU).chunk(mini_batch_size, /*dim=*/0);  // {N,1,H,W} ===> {1,1,H,W} + {1,1,H,W} + ...
    for (auto &tensor : mini_batch){
        tensor_sq = torch::squeeze(tensor, /*dim=*/0);  // {1,1,H,W} ===> {1,H,W}
        tensor_per = tensor_sq.permute({1, 2, 0});  // {1,H,W} ===> {H,W,1}
        tensor_con = tensor_per.to(torch::kInt).contiguous();
        sample = cv::Mat(cv::Size(width, height), CV_32SC1, tensor_con.data_ptr<int>());  // torch::Tensor ===> cv::Mat
        sample.copyTo(samples.at(i));
        i++;
    }

    // (3) Output Image Information
    ncol = (mini_batch_size < cols) ? mini_batch_size : cols;
    width_out = width * ncol + padding * (ncol + 1);
    nrow = 1 + (mini_batch_size - 1) / ncol;
    height_out =  height * nrow + padding * (nrow + 1);

    // (4) Palette and Value Initialization for Index Image
    output = png::image<png::index_pixel>(width_out, height_out);
    pal = png::palette(label_palette.size());
    for (i = 0; i < pal.size(); i++){
        pal_one = label_palette.at(i);
        R = std::get<0>(pal_one);
        G = std::get<1>(pal_one);
        B = std::get<2>(pal_one);
        pal[i] = png::color(R, G, B);
    }
    output.set_palette(pal);
    for (j = 0; j < height_out; j++){
        for (i = 0; i < width_out; i++){
            output[j][i] = 0;
        }
    }

    // (5) Value Substitution for Output Image
    for (k = 0; k < mini_batch_size; k++){
        sample = samples.at(k);
        i_dev = (k % ncol) * width + padding * (k % ncol + 1);
        j_dev = (k / ncol) * height + padding * (k / ncol + 1);
        for (j = 0; j < height; j++){
            for (i = 0; i < width; i++){
                output[j+j_dev][i+i_dev] = sample.at<int>(j, i);
            }
        }
    }

    // (6) Image Output
    output.write(path);

    // End Processing
    return;

}


// ----------------------------------------------------------
// namespace{visualizer} -> function{create_heatmap}
// ----------------------------------------------------------
torch::Tensor visualizer::create_heatmap(const torch::Tensor image, const std::pair<float, float> range){

    torch::Tensor gray, R, G, B, heatmap;

    gray = (image.clamp(/*min=*/range.first, /*max=*/range.second) - range.first) / (float)(range.second - range.first);  // image[range.first, range.second] ===> gray[0,1]
    R = (gray < 0.5) * 0.0 + (0.5 <= gray) * (gray < 0.75) * (4.0 * gray - 2.0) + (0.75 <= gray) * 1.0;  // 0 (0 <= x < 0.5), 4x-2 (0.5 <= x < 0.75), 1 (0.75 <= x <= 1)
    G = (gray < 0.25) * (4.0 * gray) + (0.25 <= gray) * (gray < 0.75) * 1.0 + (0.75 <= gray) * (-4.0 * gray + 4.0); // 4x (0 <= x < 0.25), 1 (0.25 <= x < 0.75), -4x+4 (0.75 <= x <= 1)
    B = (gray < 0.25) * 1.0 + (0.25 <= gray) * (gray < 0.5) * (-4.0 * gray + 2.0) + (0.5 <= gray) * 0.0; // 1 (0 <= x < 0.25), -4x+2 (0.25 <= x < 0.5), 0 (0.5 <= x <= 1)
    heatmap = torch::cat({R, G, B}, /*dim=*/1);  // R{N,1,H,W} + G{N,1,H,W} + B{N,1,H,W} ===> heatmap{N,C,H,W}

    return heatmap;

}


// ----------------------------------------------------------
// namespace{visualizer} -> function{draw_detections}
// ----------------------------------------------------------
cv::Mat visualizer::draw_detections(const torch::Tensor image, const std::tuple<torch::Tensor, torch::Tensor> label, const std::vector<std::string> class_names, const std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette, const std::pair<float, float> range){

    constexpr size_t bits = 8;
    constexpr size_t line_thickness = 2;
    constexpr size_t text_thickness = 2;
    constexpr double font_size = 0.8;
    
    // (0) Initialization and Declaration
    size_t width, height, channels;
    size_t BB_n;
    size_t x_min, y_min, x_max, y_max;
    float x_min_f, y_min_f, x_max_f, y_max_f;
    long int id;
    int baseline;
    std::tuple<unsigned char, unsigned char, unsigned char> pal;
    unsigned char R, G, B;
    float cx, cy, w, h;
    cv::Size text_size;
    std::string class_name;
    cv::Mat mat, sample;
    torch::Tensor tensor;
    torch::Tensor ids, coords;
    
    // (1) Get Tensor Size
    channels = image.size(0);
    height = image.size(1);
    width = image.size(2);

    // (2) Judge the number of channels
    if ((channels != 1) && (channels != 3)){
        std::cerr << "Error : Channels of the image is inappropriate." << std::endl;
        std::exit(1);
    }
    
    // (3) Convert "torch::Tensor" into "cv::Mat"
    tensor = image.clamp(/*min=*/range.first, /*max=*/range.second).to(torch::kCPU);  // GPU ===> CPU
    tensor = tensor.expand({3, (long int)height, (long int)width});  // {C,H,W} ===> {3,H,W}
    tensor = tensor.permute({1, 2, 0});  // {3,H,W} ===> {H,W,3}
    tensor = tensor.contiguous();
    mat = cv::Mat(cv::Size(width, height), CV_32FC3, tensor.data_ptr<float>());  // torch::Tensor ===> cv::Mat
    mat = (mat - range.first) / (float)(range.second - range.first);  // [range.first, range.second] ===> [0,1]
    mat = mat * (std::pow(2.0, bits) - 1.0);  // [0,1] ===> [0,255]
    mat.convertTo(sample, CV_8UC3);  // {32F} ===> {8U}
    cv::cvtColor(sample, sample, cv::COLOR_RGB2BGR);  // {R,G,B} ===> {B,G,R}

    // (4) Draw Bounding Box with class names
    ids = std::get<0>(label);
    coords = std::get<1>(label);
    BB_n = ids.numel();
    if (BB_n > 0){
        for(size_t b = 0; b < BB_n; b++){

            // (4.1) Get parameters
            id = ids[b].item<long int>();  // id[0,CN)
            cx = coords[b][0].item<float>();  // cx[0.0,1.0)
            cy = coords[b][1].item<float>();  // cy[0.0,1.0)
            w = coords[b][2].item<float>();  // w[0.0,1.0)
            h = coords[b][3].item<float>();  // h[0.0,1.0)
            x_min_f = (cx - 0.5 * w) * (float)width + 0.5;
            y_min_f = (cy - 0.5 * h) * (float)height + 0.5;
            x_max_f = (cx + 0.5 * w) * (float)width + 0.5;
            y_max_f = (cy + 0.5 * h) * (float)height + 0.5;
            x_min = (size_t)((x_min_f > 0.0) ? x_min_f : 0.0);
            y_min = (size_t)((y_min_f > 0.0) ? y_min_f : 0.0);
            x_max = (size_t)((x_max_f < (float)(width - 1)) ? x_max_f : (float)(width - 1));
            y_max = (size_t)((y_max_f < (float)(height - 1)) ? y_max_f : (float)(height - 1));
            pal = label_palette.at(id);
            R = std::get<0>(pal);
            G = std::get<1>(pal);
            B = std::get<2>(pal);

            // (4.2) Draw bounding box
            cv::rectangle(sample, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(B, G, R), /*thickness=*/line_thickness);

            // (4.3) Draw class names
            class_name = class_names.at(id);
            text_size = cv::getTextSize(class_name, cv::FONT_HERSHEY_SIMPLEX, /*fontScale=*/font_size, /*thickness=*/text_thickness, &baseline);
            cv::rectangle(sample, cv::Point(x_min, y_min), cv::Point(x_min + text_size.width + line_thickness, y_min + text_size.height + line_thickness + baseline), cv::Scalar(B, G, R), /*thickness=*/-1);
            cv::putText(sample, class_name, cv::Point(x_min + line_thickness, y_min + 2 * baseline + line_thickness), cv::FONT_HERSHEY_SIMPLEX, /*fontScale=*/font_size, cv::Scalar(0, 0, 0), /*thickness=*/text_thickness, /*lineType=*/cv::LINE_8);

        }
    }

    // End Processing
    return sample;

}


// ----------------------------------------------------------
// namespace{visualizer} -> function{draw_detections_des}
// ----------------------------------------------------------
cv::Mat visualizer::draw_detections_des(const torch::Tensor image, const std::tuple<torch::Tensor, torch::Tensor> label, const torch::Tensor prob, const std::vector<std::string> class_names, const std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette, const std::pair<float, float> range){

    constexpr size_t bits = 8;
    constexpr size_t line_thickness_max = 5;
    constexpr size_t line_thickness_min = 1;
    constexpr size_t text_thickness_max = 2;
    constexpr size_t text_thickness_min = 1;
    constexpr double font_size_max = 0.5;
    constexpr double font_size_min = 0.3;
    
    // (0) Initialization and Declaration
    size_t width, height, channels;
    size_t BB_n;
    size_t x_min, y_min, x_max, y_max;
    size_t line_thickness, text_thickness;
    double font_size;
    double rate;
    float x_min_f, y_min_f, x_max_f, y_max_f;
    long int id;
    int baseline;
    std::tuple<unsigned char, unsigned char, unsigned char> pal;
    unsigned char R, G, B;
    float cx, cy, w, h;
    cv::Size text_size;
    std::string class_name;
    cv::Mat mat, sample;
    torch::Tensor tensor;
    torch::Tensor ids, coords;
    torch::Tensor ids_sorted, coords_sorted, probs_sorted, probs_idx;
    std::tuple<torch::Tensor, torch::Tensor> probs_sorted_with_idx;
    
    // (1) Get Tensor Size
    channels = image.size(0);
    height = image.size(1);
    width = image.size(2);

    // (2) Judge the number of channels
    if ((channels != 1) && (channels != 3)){
        std::cerr << "Error : Channels of the image is inappropriate." << std::endl;
        std::exit(1);
    }
    
    // (3) Convert "torch::Tensor" into "cv::Mat"
    tensor = image.clamp(/*min=*/range.first, /*max=*/range.second).to(torch::kCPU);  // GPU ===> CPU
    tensor = tensor.expand({3, (long int)height, (long int)width});  // {C,H,W} ===> {3,H,W}
    tensor = tensor.permute({1, 2, 0});  // {3,H,W} ===> {H,W,3}
    tensor = tensor.contiguous();
    mat = cv::Mat(cv::Size(width, height), CV_32FC3, tensor.data_ptr<float>());  // torch::Tensor ===> cv::Mat
    mat = (mat - range.first) / (float)(range.second - range.first);  // [range.first, range.second] ===> [0,1]
    mat = mat * (std::pow(2.0, bits) - 1.0);  // [0,1] ===> [0,255]
    mat.convertTo(sample, CV_8UC3);  // {32F} ===> {8U}
    cv::cvtColor(sample, sample, cv::COLOR_RGB2BGR);  // {R,G,B} ===> {B,G,R}

    // (4) Draw Bounding Box with class names
    ids = std::get<0>(label);
    coords = std::get<1>(label);
    BB_n = ids.numel();
    if (BB_n > 0){

        probs_sorted_with_idx = prob.sort(/*dim=*/0, /*descending=*/false);  // probs_sorted_with_idx(probs{object}, idx{object})
        probs_sorted = std::get<0>(probs_sorted_with_idx);  // probs_sorted{object}
        probs_idx = std::get<1>(probs_sorted_with_idx);  // probs_idx{object}
        ids_sorted = ids.gather(/*dim=*/0, /*index=*/probs_idx);  // ids_sorted{object}
        coords_sorted = coords.gather(/*dim=*/0, /*index=*/probs_idx.unsqueeze(/*dim=*/-1).expand({probs_idx.size(0), 4})).contiguous();  // coords_sorted{object,4}

        for(size_t b = 0; b < BB_n; b++){

            // (4.1) Get parameters
            id = ids_sorted[b].item<long int>();  // id[0,CN)
            cx = coords_sorted[b][0].item<float>();  // cx[0.0,1.0)
            cy = coords_sorted[b][1].item<float>();  // cy[0.0,1.0)
            w = coords_sorted[b][2].item<float>();  // w[0.0,1.0)
            h = coords_sorted[b][3].item<float>();  // h[0.0,1.0)
            x_min_f = (cx - 0.5 * w) * (float)width + 0.5;
            y_min_f = (cy - 0.5 * h) * (float)height + 0.5;
            x_max_f = (cx + 0.5 * w) * (float)width + 0.5;
            y_max_f = (cy + 0.5 * h) * (float)height + 0.5;
            x_min = (size_t)((x_min_f > 0.0) ? x_min_f : 0.0);
            y_min = (size_t)((y_min_f > 0.0) ? y_min_f : 0.0);
            x_max = (size_t)((x_max_f < (float)(width - 1)) ? x_max_f : (float)(width - 1));
            y_max = (size_t)((y_max_f < (float)(height - 1)) ? y_max_f : (float)(height - 1));
            pal = label_palette.at(id);
            R = std::get<0>(pal);
            G = std::get<1>(pal);
            B = std::get<2>(pal);

            // (4.2) Calculate size
            rate = probs_sorted[b].item<float>();
            line_thickness = line_thickness_min + (size_t)((double)(line_thickness_max - line_thickness_min) * rate + 0.5);
            text_thickness = text_thickness_min + (size_t)((double)(text_thickness_max - text_thickness_min) * rate + 0.5);
            font_size = font_size_min + (font_size_max - font_size_min) * rate;

            // (4.3) Draw bounding box
            cv::rectangle(sample, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(B, G, R), /*thickness=*/line_thickness);

            // (4.4) Draw class names
            class_name = class_names.at(id);
            text_size = cv::getTextSize(class_name, cv::FONT_HERSHEY_SIMPLEX, /*fontScale=*/font_size, /*thickness=*/text_thickness, &baseline);
            cv::rectangle(sample, cv::Point(x_min, y_min), cv::Point(x_min + text_size.width + line_thickness, y_min + text_size.height + line_thickness + baseline), cv::Scalar(B, G, R), /*thickness=*/-1);
            cv::putText(sample, class_name, cv::Point(x_min + line_thickness, y_min + 2 * baseline + line_thickness), cv::FONT_HERSHEY_SIMPLEX, /*fontScale=*/font_size, cv::Scalar(0, 0, 0), /*thickness=*/text_thickness, /*lineType=*/cv::LINE_8);

        }
    }

    // End Processing
    return sample;

}


// ----------------------------------------------------------
// namespace{visualizer} -> class{graph} -> constructor
// ----------------------------------------------------------
visualizer::graph::graph(const std::string dir_, const std::string gname_, const std::vector<std::string> label_){
    this->flag = false;
    this->dir = dir_;
    this->data_dir = this->dir + "/data";
    this->gname = gname_;
    this->graph_fname= this->dir + '/' + this->gname + ".png";
    this->data_fname= this->data_dir + '/' + this->gname + ".dat";
    this->label = label_;
    fs::create_directories(this->dir);
    fs::create_directories(this->data_dir);
}


// ----------------------------------------------------------
// namespace{visualizer} -> class{graph} -> function{plot}
// ----------------------------------------------------------
void visualizer::graph::plot(const float base, const std::vector<float> value){

    // (1) Value Output
    std::ofstream ofs(this->data_fname, std::ios::app);
    ofs << base << std::flush;
    for (auto &v : value){
        ofs << ' ' << v << std::flush;
    }
    ofs << std::endl;
    ofs.close();

    // (2) Graph Output
    if (this->flag){
        FILE* gp;
        gp = popen("gnuplot -persist", "w");
        fprintf(gp, "set terminal png\n");
        fprintf(gp, "set output '%s'\n", this->graph_fname.c_str());
        fprintf(gp, "plot ");
        for (size_t i = 0; i < this->label.size() - 1; i++){
            fprintf(gp, "'%s' using 1:%zu ti '%s' with lines,", this->data_fname.c_str(), i + 2, this->label.at(i).c_str());
        }
        fprintf(gp, "'%s' using 1:%zu ti '%s' with lines\n", this->data_fname.c_str(), this->label.size() + 1, this->label.at(this->label.size() - 1).c_str());
        pclose(gp);
    }

    // (3) Setting for after the Second Time
    this->flag = true;

    // End Processing
    return;

}