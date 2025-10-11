#include <tuple>
#include <algorithm>
#include <cmath>
#include <vector>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "losses.hpp"

namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;


// -----------------------------------
// class{Loss} -> constructor
// -----------------------------------
Loss::Loss(const std::vector<std::vector<std::tuple<float, float>>> anchors_, const long int class_num_, const float anchor_thresh_){

    long int scales = anchors_.size();
    this->na = anchors_.at(0).size();
    this->class_num = class_num_;
    this->anchor_thresh = anchor_thresh_;

    this->anchors = torch::zeros({scales, this->na, 2}, torch::TensorOptions().dtype(torch::kFloat));
    for (long int i = 0; i < scales; i++){
        for (long int j = 0; j < this->na; j++){
            this->anchors.index_put_({i, j, 0}, std::get<0>(anchors_.at(i).at(j)));
            this->anchors.index_put_({i, j, 1}, std::get<1>(anchors_.at(i).at(j)));
        }
    }

    this->balance = {4.0, 1.0, 0.4};
    if (this->balance.size() != (size_t)scales){
        this->balance = std::vector<float>(scales, 1.0);
    }

}


// ----------------------------------------
// class{Loss} -> function{format_target}
// ----------------------------------------
torch::Tensor Loss::format_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>> target, torch::Device device){

    torch::Tensor ids, coords;
    std::vector<float> buffer;
    torch::Tensor out;

    for (size_t n = 0; n < target.size(); n++){
        std::tie(ids, coords) = target.at(n);
        if (ids.numel() == 0) continue;
        ids = ids.to(torch::kLong);
        coords = coords.to(torch::kFloat);
        for (long int b = 0; b < ids.size(0); b++){
            buffer.push_back(n);
            buffer.push_back(ids[b].item<long int>());
            buffer.push_back(coords[b][0].item<float>());
            buffer.push_back(coords[b][1].item<float>());
            buffer.push_back(coords[b][2].item<float>());
            buffer.push_back(coords[b][3].item<float>());
        }
    }

    if (buffer.empty()){
        out = torch::empty({0, 6}, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    }
    else{
        out = torch::from_blob(buffer.data(), {(long int)(buffer.size() / 6), 6}, torch::kFloat).clone().to(device);
    }

    return out;

}


// -------------------------------------
// class{Loss} -> function{build_target}
// -------------------------------------
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> Loss::build_target(torch::Tensor targets, std::vector<std::array<long int, 2>> grid_shapes, std::vector<torch::Tensor> scaled_anchors, torch::Device device){

    size_t scales;
    long int nt;
    std::vector<torch::Tensor> result_indices, result_tbox, result_tclass, result_anchors;
    torch::Tensor gain, ai, targets_expanded, ai_expand, anchor_idx, anchors, wh, ratio, max_ratio, mask, t, b, cls, gxy, gij, gi, gj;

    scales = grid_shapes.size();
    result_indices = std::vector<torch::Tensor>(scales);
    result_tbox = std::vector<torch::Tensor>(scales);
    result_tclass = std::vector<torch::Tensor>(scales);
    result_anchors = std::vector<torch::Tensor>(scales);

    gain = torch::ones({7}).to(device);
    nt = targets.size(0);
    
    if (nt > 0){
        ai = torch::arange(this->na).to(device).view({(long int)this->na, 1, 1});
        targets_expanded = targets.repeat({(long int)this->na, 1});
        targets_expanded = targets_expanded.view({(long int)this->na, nt, 6});
        ai_expand = ai.expand({(long int)this->na, nt, 1});
        targets_expanded = torch::cat({targets_expanded, ai_expand}, 2).view({-1, 7});
    }
    else{
        targets_expanded = torch::empty({0, 7}).to(device);
    }

    for (size_t i = 0; i < scales; i++){
        gain.index_put_({2}, grid_shapes.at(i).at(1));
        gain.index_put_({3}, grid_shapes.at(i).at(0));
        gain.index_put_({4}, grid_shapes.at(i).at(1));
        gain.index_put_({5}, grid_shapes.at(i).at(0));

        t = targets_expanded * gain;
        if (t.size(0) == 0){
            result_indices.at(i) = torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kLong)).to(device);
            result_tbox.at(i) = torch::empty({0, 4}).to(device);
            result_tclass.at(i) = torch::empty({0, this->class_num}).to(device);
            result_anchors.at(i) = torch::empty({0, 2}).to(device);
            continue;
        }

        anchor_idx = t.index({Slice(), 6}).to(torch::kLong);
        anchors = scaled_anchors.at(i).index_select(0, anchor_idx);

        wh = t.index({Slice(), Slice(4, 6)});
        ratio = torch::max(wh / anchors, anchors / wh);
        max_ratio = std::get<0>(ratio.max(1, false));
        mask = (max_ratio < this->anchor_thresh);

        if (!mask.any().item<bool>()){
            result_indices.at(i) = torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kLong)).to(device);
            result_tbox.at(i) = torch::empty({0, 4}).to(device);
            result_tclass.at(i) = torch::empty({0, this->class_num}).to(device);
            result_anchors.at(i) = torch::empty({0, 2}).to(device);
            continue;
        }

        t = t.index({mask});
        anchors = anchors.index({mask});

        b = t.index({Slice(), 0}).to(torch::kLong);
        cls = t.index({Slice(), 1}).to(torch::kLong);
        gxy = t.index({Slice(), Slice(2, 4)});
        gij = torch::floor(gxy);
        gi = gij.index({Slice(), 0}).to(torch::kLong).clamp(0, grid_shapes.at(i).at(1) - 1);
        gj = gij.index({Slice(), 1}).to(torch::kLong).clamp(0, grid_shapes.at(i).at(0) - 1);

        result_indices.at(i) = torch::stack({b, t.index({Slice(), 6}).to(torch::kLong), gj, gi}, 1);
        result_tbox.at(i) = torch::cat({gxy - gij, t.index({Slice(), Slice(4, 6)})}, 1);
        if (this->class_num > 1){
            result_tclass.at(i) = torch::one_hot(cls, this->class_num).to(torch::kFloat).to(device);
        }
        else{
            result_tclass.at(i) = torch::empty({cls.size(0), 0}).to(device);
        }
        result_anchors.at(i) = anchors;

    }

    return {result_indices, result_tbox, result_tclass, result_anchors};  // target_class{N,G,G,CN}, target_coord{N,G,G,4}, target_mask{N,G,G}
    
}


// -------------------------------------
// class{Loss} -> function{compute_iou}
// -------------------------------------
torch::Tensor Loss::bbox_iou(torch::Tensor box1, torch::Tensor box2){
    
    constexpr float eps = 1e-7;

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
    inter = width * height;  // width{BB1,BB2,object}, height{BB1,BB2,object} ===> inter{BB1,BB2,object}

    // (4) Compute area of the bounding boxes and IoU
    area1 = (x2_1 - x1_1).clamp_min(0.0) * (y2_1 - y1_1).clamp_min(0.0);
    area2 = (x2_2 - x1_2).clamp_min(0.0) * (y2_2 - y1_2).clamp_min(0.0);
    unions = area1 + area2 - inter;  // area1{BB1,BB2,object}, area2{BB1,BB2,object}, inter{BB1,BB2,object} ===> unions{BB1,BB2,object}
    iou = inter / (unions + eps);  // inter{BB1,BB2,object}, unions{BB1,BB2,object} ===> IoU{BB1,BB2,object}

    // (5) Compute cIoU
    rho2 = torch::pow(box1.index({Slice(), 0}) - box2.index({Slice(), 0}), 2.0) + torch::pow(box1.index({Slice(), 1}) - box2.index({Slice(), 1}), 2.0);
    cw = torch::max(x2_1, x2_2) - torch::min(x1_1, x1_2);
    ch = torch::max(y2_1, y2_2) - torch::min(y1_1, y1_2);
    c2 = torch::pow(cw, 2.0) + torch::pow(ch, 2.0) + eps;
    v = 4.0 / (M_PI * M_PI) * torch::pow(torch::atan(box2.index({Slice(), 2}) / (box2.index({Slice(), 3}) + eps)) - torch::atan(box1.index({Slice(), 2}) / (box1.index({Slice(), 3}) + eps)), 2.0);
    alpha = v / (1.0 - iou + v + eps);
    ciou = iou - (rho2 / c2 + v * alpha);

    return ciou;

}


// -------------------------
// class{Loss} -> operator
// -------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Loss::operator()(std::vector<torch::Tensor> &inputs, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target, const std::tuple<float, float> image_sizes){

    torch::Device device = inputs.at(0).device();
    size_t scales = inputs.size();

    torch::Tensor targets_tensor, input, pred, anchors_level;
    std::vector<torch::Tensor> result_indices, result_tbox, result_tclass, result_anchors;
    torch::Tensor loss_box, loss_obj, loss_class;
    torch::Tensor obj_pred, obj_target, indices, tbox, tclass, tanchors;
    torch::Tensor b, a, gj, gi, ps, pxy, pwh, pbox, iou, pred_class, obj_loss;
    std::vector<std::array<long int, 2>> grid_shapes;
    std::vector<torch::Tensor> scaled_anchors;
    std::vector<torch::Tensor> predictions;
    float stride_x, stride_y;
    long int width, height;
    long int mini_batch_size, gy, gx;

    targets_tensor = this->format_target(target, device);
    grid_shapes = std::vector<std::array<long int, 2>>(scales);
    scaled_anchors = std::vector<torch::Tensor>(scales);
    predictions = std::vector<torch::Tensor>(scales);
    std::tie(width, height) = image_sizes;

    for (size_t i = 0; i < scales; i++){

        input = inputs.at(i);
        mini_batch_size = input.size(0);
        gy = input.size(1);
        gx = input.size(2);
        grid_shapes.at(i) = {gy, gx};

        pred = input.view({mini_batch_size, gy, gx, this->na, this->class_num + 5}).permute({0, 3, 1, 2, 4}).contiguous();
        predictions.at(i) = pred;

        stride_x = width / (float)gx;
        stride_y = height / (float)gy;
        anchors_level = this->anchors.index({(long int)i}).to(device).clone();
        anchors_level.index_put_({Slice(), 0}, anchors_level.index({Slice(), 0}) / stride_x);
        anchors_level.index_put_({Slice(), 1}, anchors_level.index({Slice(), 1}) / stride_y);
        scaled_anchors.at(i) = anchors_level;
    }
    std::tie(result_indices, result_tbox, result_tclass, result_anchors) = this->build_target(targets_tensor, grid_shapes, scaled_anchors, device);

    loss_box = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_obj = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_class = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_box.requires_grad_(true);
    loss_obj.requires_grad_(true);
    loss_class.requires_grad_(true);
    for (size_t i = 0; i < scales; i++){

        pred = predictions.at(i);
        obj_pred = pred.index({Slice(), Slice(), Slice(), Slice(), 4});
        obj_target = torch::zeros_like(obj_pred);

        indices = result_indices.at(i);
        tbox = result_tbox.at(i);
        tclass = result_tclass.at(i);
        tanchors = result_anchors.at(i).to(device);

        if (indices.size(0) > 0){
            b = indices.index({Slice(), 0}).to(torch::kLong);
            a = indices.index({Slice(), 1}).to(torch::kLong);
            gj = indices.index({Slice(), 2}).to(torch::kLong);
            gi = indices.index({Slice(), 3}).to(torch::kLong);

            ps = pred.index({b, a, gj, gi});
            pxy = ps.index({Slice(), Slice(0, 2)}).sigmoid() * 2.0 - 0.5;
            pwh = torch::pow(ps.index({Slice(), Slice(2, 4)}).sigmoid() * 2.0, 2) * tanchors;
            pbox = torch::cat({pxy, pwh}, 1);

            iou = this->bbox_iou(pbox, tbox.to(device));
            loss_box = loss_box + (1.0 - iou).mean();

            obj_target.index_put_({b, a, gj, gi}, iou.detach().clamp(0.0, 1.0));

            if ((this->class_num > 1) && (tclass.numel() > 0)){
                pred_class = ps.index({Slice(), Slice(5, 5 + this->class_num)});
                loss_class = loss_class + F::binary_cross_entropy_with_logits(pred_class, tclass.to(device), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kMean));
            }

        }

        obj_loss = F::binary_cross_entropy_with_logits(obj_pred, obj_target, F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kMean));
        loss_obj = loss_obj + obj_loss * this->balance.at(i);

    }


    return {loss_box, loss_obj, loss_class};

}
