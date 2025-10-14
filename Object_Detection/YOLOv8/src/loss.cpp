#include <tuple>
#include <algorithm>
#include <array>
#include <utility>
#include <limits>
#include <cmath>
#include <vector>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "losses.hpp"

namespace nn = torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;


// -----------------------------------
// class{Loss} -> constructor
// -----------------------------------
Loss::Loss(const long int class_num_){
    this->class_num = class_num_;
    this->BCE = nn::BCEWithLogitsLoss(nn::BCEWithLogitsLossOptions().reduction(torch::kMean));
    this->balance = {4.0, 1.0, 0.4};
}


// -------------------------------------
// class{Loss} -> function{build_target}
// -------------------------------------
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> Loss::build_target(std::vector<torch::Tensor> &inputs, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target){

    constexpr float g = 0.5;
    torch::Device device = inputs.at(0).device();
    size_t scales = inputs.size();

    // (1) Build target tensor
    torch::Tensor ids, coords, target_tensor;
    std::vector<float> buffer;
    /*******************************************************/
    for (size_t n = 0; n < target.size(); n++){
        std::tie(ids, coords) = target[n];
        if (ids.numel() != 0){
            for (long int b = 0; b < ids.size(0); b++){
                buffer.push_back(n);
                buffer.push_back(ids[b].item<long int>());
                buffer.push_back(coords[b][0].item<float>());
                buffer.push_back(coords[b][1].item<float>());
                buffer.push_back(coords[b][2].item<float>());
                buffer.push_back(coords[b][3].item<float>());
            }
        }
    }
    /*******************************************************/
    if (buffer.empty()){
        target_tensor = torch::empty({0, 6}, torch::kFloat).to(device);
    }
    else{
        target_tensor = torch::from_blob(buffer.data(), {(long int)(buffer.size() / 6), 6}, torch::kFloat).clone().to(device);  // {T,6}
    }

    // (2) Build target
    long int nt;
    torch::Tensor gain, ai, off, xyxy_gain, t, r, ones, j, gxy, gxi, jk, lm, offsets, b, c, gwh, a, gij, gi, gj;
    std::vector<torch::Tensor> scale_indices_b, scale_indices_a, scale_indices_gj, scale_indices_gi, scale_tbox, scale_tclass;
    /*******************************************************/
    nt = target_tensor.size(0);
    gain = torch::ones({7}).to(device);  // {7}
    ai = torch::arange(1).to(device).to(torch::kFloat).view({1, 1}).repeat({1, nt}).unsqueeze(-1);  // {1,T,1}
    target_tensor = torch::cat({target_tensor.view({1, nt, 6}).repeat({1, 1, 1}), ai}, 2);  // {1,T,7}
    off = torch::tensor({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}, torch::kFloat).to(device) * g;  // {5,2}
    /*******************************************************/
    scale_indices_b = std::vector<torch::Tensor>(scales);
    scale_indices_a = std::vector<torch::Tensor>(scales);
    scale_indices_gj = std::vector<torch::Tensor>(scales);
    scale_indices_gi = std::vector<torch::Tensor>(scales);
    scale_tbox = std::vector<torch::Tensor>(scales);
    scale_tclass = std::vector<torch::Tensor>(scales);
    /*******************************************************/
    for (size_t i = 0; i < scales; i++){

        if (nt == 0){
            scale_indices_b[i] = torch::empty({0}, torch::kLong).to(device);
            scale_indices_a[i] = torch::empty({0}, torch::kLong).to(device);
            scale_indices_gj[i] = torch::empty({0}, torch::kLong).to(device);
            scale_indices_gi[i] =  torch::empty({0}, torch::kLong).to(device);
            scale_tbox[i] = torch::empty({0, 4}, torch::kFloat).to(device);
            scale_tclass[i] = torch::empty({0}, torch::kLong).to(device);
            continue;
        }

        xyxy_gain = torch::tensor({(float)inputs[i].size(2), (float)inputs[i].size(1), (float)inputs[i].size(2), (float)inputs[i].size(1)}, torch::kFloat).to(device);  // {4}
        gain.index_put_({Slice(2, 6)}, xyxy_gain);  // {7}
        t = target_tensor * gain.view({1, 1, 7});  // {B,T,7}
        t = t.view({-1, 7}).contiguous();  // {T',7}
        
        gxy = t.index({Slice(), Slice(2, 4)});  // {T',2}
        gxi = gain.index({Slice(2, 4)}).unsqueeze(0) - gxy;  // {T',2}
        jk = (gxy - torch::floor(gxy) < g) & (gxy > 1.0);  // {T',2}
        lm = (gxi - torch::floor(gxi) < g) & (gxi > 1.0);  // {T',2}
        ones = torch::ones_like(jk.index({Slice(), 0}), torch::kBool);  // {T'}
        j = torch::stack({ones, jk.index({Slice(), 0}), jk.index({Slice(), 1}), lm.index({Slice(), 0}), lm.index({Slice(), 1})});  // {5,T'}
        t = t.unsqueeze(0).repeat({5, 1, 1}).index({j});  // {K,7}
        offsets = off.unsqueeze(1).repeat({1, gxy.size(0), 1}).index({j});  // {K,2}

        b = t.index({Slice(), 0}).to(torch::kLong);  // {K}
        c = t.index({Slice(), 1}).to(torch::kLong);  // {K}
        gxy = t.index({Slice(), Slice(2, 4)});  // {K,2}
        gwh = t.index({Slice(), Slice(4, 6)});  // {K,2}
        a = t.index({Slice(), 6}).to(torch::kLong);  // {K}
        gij = (gxy - offsets).to(torch::kLong);  // {K,2}
        gi = gij.index({Slice(), 0});  // {K}
        gj = gij.index({Slice(), 1});  // {K}

        scale_indices_b[i] = b;
        scale_indices_a[i] = a;
        scale_indices_gj[i] = gj.clamp(0, inputs[i].size(1) - 1);
        scale_indices_gi[i] = gi.clamp(0, inputs[i].size(2) - 1);
        scale_tbox[i] = torch::cat({gxy - gij.to(torch::kFloat), gwh}, 1);
        scale_tclass[i] = c;

    }
    
    return {scale_indices_b, scale_indices_a, scale_indices_gj, scale_indices_gi, scale_tbox, scale_tclass};
    
}


// -------------------------------------
// class{Loss} -> function{compute_iou}
// -------------------------------------
torch::Tensor Loss::bbox_iou(torch::Tensor box1, torch::Tensor box2){
    
    constexpr float eps = 1e-7;
    constexpr float PI = 3.14159265358979;

    torch::Tensor x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2;
    torch::Tensor width, height, inter, area1, area2, unions, iou;
    torch::Tensor rho2, cw, ch, c2, v, alpha, ciou;

    // (1) Compute for box1
    x1_1 = box1.index({Slice(), 0}) - box1.index({Slice(), 2}) * 0.5;
    y1_1 = box1.index({Slice(), 1}) - box1.index({Slice(), 3}) * 0.5;
    x2_1 = box1.index({Slice(), 0}) + box1.index({Slice(), 2}) * 0.5;
    y2_1 = box1.index({Slice(), 1}) + box1.index({Slice(), 3}) * 0.5;

    // (2) Compute for box2
    x1_2 = box2.index({Slice(), 0}) - box2.index({Slice(), 2}) * 0.5;
    y1_2 = box2.index({Slice(), 1}) - box2.index({Slice(), 3}) * 0.5;
    x2_2 = box2.index({Slice(), 0}) + box2.index({Slice(), 2}) * 0.5;
    y2_2 = box2.index({Slice(), 1}) + box2.index({Slice(), 3}) * 0.5;

    // (3) Compute area of the intersections from the coordinates
    width = (torch::min(x2_1, x2_2) - torch::max(x1_1, x1_2)).clamp_min(0.0);
    height = (torch::min(y2_1, y2_2) - torch::max(y1_1, y1_2)).clamp_min(0.0);
    inter = width * height;

    // (4) Compute area of the bounding boxes and IoU
    area1 = (x2_1 - x1_1).clamp_min(0.0) * (y2_1 - y1_1).clamp_min(0.0);
    area2 = (x2_2 - x1_2).clamp_min(0.0) * (y2_2 - y1_2).clamp_min(0.0);
    unions = area1 + area2 - inter;
    iou = inter / (unions + eps);

    // (5) Compute cIoU
    rho2 = torch::pow(box1.index({Slice(), 0}) - box2.index({Slice(), 0}), 2.0) + torch::pow(box1.index({Slice(), 1}) - box2.index({Slice(), 1}), 2.0);
    cw = torch::max(x2_1, x2_2) - torch::min(x1_1, x1_2);
    ch = torch::max(y2_1, y2_2) - torch::min(y1_1, y1_2);
    c2 = torch::pow(cw, 2.0) + torch::pow(ch, 2.0) + eps;
    v = 4.0 / (PI * PI) * torch::pow(torch::atan(box2.index({Slice(), 2}) / (box2.index({Slice(), 3}) + eps)) - torch::atan(box1.index({Slice(), 2}) / (box1.index({Slice(), 3}) + eps)), 2.0);
    alpha = v / (1.0 - iou + v + eps);
    ciou = iou - (rho2 / c2 + v * alpha);

    return ciou;

}


// -------------------------
// class{Loss} -> operator
// -------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Loss::operator()(std::vector<torch::Tensor> &inputs, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target){

    torch::Device device = inputs.at(0).device();
    size_t scales = inputs.size();

    long int n;
    torch::Tensor loss_box, loss_obj, loss_class;
    torch::Tensor b, a, gj, gi, input, tobj, ps, grid_size, pxy, pwh, pcls, pbox, iou, t, obji;
    std::vector<torch::Tensor> scale_indices_b, scale_indices_a, scale_indices_gj, scale_indices_gi, scale_tbox, scale_tclass;

    loss_box = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_obj = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_class = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_box.requires_grad_(true);
    loss_obj.requires_grad_(true);
    loss_class.requires_grad_(true);
    std::tie(scale_indices_b, scale_indices_a, scale_indices_gj, scale_indices_gi, scale_tbox, scale_tclass) = this->build_target(inputs, target);
    for (size_t i = 0; i < scales; i++){
        b = scale_indices_b[i];
        a = scale_indices_a[i];
        gj = scale_indices_gj[i];
        gi = scale_indices_gi[i];
        input = inputs[i].view({inputs[i].size(0), inputs[i].size(1), inputs[i].size(2), 1, 5 + this->class_num}).permute({0, 3, 1, 2, 4}).contiguous();  // {N,G,G,5+CN} ===> {N,G,G,1,5+CN} ===> {N,1,G,G,5+CN}
        tobj = torch::zeros({input.size(0), input.size(1), input.size(2), input.size(3)}, torch::kFloat).to(device);  // {N,B,G,G}

        n = b.size(0);
        if (n > 0){
            ps = input.index({b, a, gj, gi});  // {K,5+CN}
            pxy = ps.index({Slice(), Slice(0, 2)});  // {K,2}
            pwh = ps.index({Slice(), Slice(2, 4)});  // {K,2}
            pcls = ps.index({Slice(), Slice(5, torch::indexing::None)});  // {K,CN}

            grid_size = torch::tensor({{inputs[i].size(2), inputs[i].size(1)}}, torch::kFloat).to(device);  // {1,2}
            pxy = pxy.sigmoid() * 2.0 - 0.5;  // {K,2}
            pwh = (pwh.sigmoid() * 2.0).pow(2.0) * grid_size;  // {K,2}
            pbox = torch::cat({pxy, pwh}, 1);  // {K,4}
            iou = this->bbox_iou(pbox, scale_tbox[i]).squeeze();  // {K}
            loss_box = loss_box + (1.0 - iou).mean();  // {}

            tobj.index_put_({b, a, gj, gi}, iou.detach().clamp(0.0, 1.0));  // {N,B,G,G}

            if (this->class_num > 1){
                t = torch::zeros_like(pcls);  // {K,CN}
                t.index_put_({torch::arange(n, torch::kLong).to(device), scale_tclass[i]}, 1.0);  // {K,CN}
                loss_class = loss_class + this->BCE(pcls, t); // {}
            }
        }

        obji = this->BCE(input.index({Slice(), Slice(), Slice(), Slice(), 4}), tobj);  // {}
        loss_obj = loss_obj + obji * this->balance[i];  // {}

    }

    return {loss_box, loss_obj, loss_class};

}
