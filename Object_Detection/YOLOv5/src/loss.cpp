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

namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;


// -----------------------------------
// class{Loss} -> constructor
// -----------------------------------
Loss::Loss(const std::vector<std::vector<std::tuple<float, float>>> anchors_, const long int class_num_, const float anchor_thresh_, const LossHyperparameters &hyp_, const bool autobalance_){

    long int scales = anchors_.size();
    this->na = anchors_.at(0).size();
    this->class_num = class_num_;
    this->anchor_thresh = anchor_thresh_;
    this->hyp = hyp_;
    this->hyp.anchor_t = anchor_thresh_;
    this->autobalance = autobalance_;
    this->ssi_initialized = false;
    this->ssi = 0;
    std::tie(this->cp, this->cn) = this->smooth_BCE(this->hyp.label_smoothing);
    this->focal_gamma = this->hyp.focal_gamma;
    this->focal_alpha = this->hyp.focal_alpha;
    this->class_pos_weight = this->hyp.class_pos_weight;
    this->obj_pos_weight = this->hyp.obj_pos_weight;

    this->anchors = torch::zeros({scales, this->na, 2}, torch::TensorOptions().dtype(torch::kFloat));
    for (long int i = 0; i < scales; i++){
        for (long int j = 0; j < this->na; j++){
            this->anchors.index_put_({i, j, 0}, std::get<0>(anchors_.at(i).at(j)));
            this->anchors.index_put_({i, j, 1}, std::get<1>(anchors_.at(i).at(j)));
        }
    }

    std::vector<float> base_balance = {4.0, 1.0, 0.25, 0.06, 0.02};
    if ((size_t)scales <= base_balance.size()){
        this->balance = std::vector<float>(base_balance.begin(), base_balance.begin() + scales);
    }
    else{
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
    torch::Tensor gain, ai, targets_expanded, ai_expand, anchor_idx, anchors, wh, ratio, max_ratio, mask, t;

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
            result_tclass.at(i) = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong)).to(device);
            result_anchors.at(i) = torch::empty({0, 2}).to(device);
            continue;
        }

        anchor_idx = t.index({Slice(), 6}).to(torch::kLong);
        anchors = scaled_anchors.at(i).index_select(0, anchor_idx);

        wh = t.index({Slice(), Slice(4, 6)});
        ratio = torch::max(wh / anchors, anchors / wh);
        max_ratio = std::get<0>(ratio.max(1, false));
        mask = (max_ratio < this->hyp.anchor_t);

        if (!mask.any().item<bool>()){
            result_indices.at(i) = torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kLong)).to(device);
            result_tbox.at(i) = torch::empty({0, 4}).to(device);
            result_tclass.at(i) = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong)).to(device);
            result_anchors.at(i) = torch::empty({0, 2}).to(device);
            continue;
        }

        t = t.index({mask});
        anchors = anchors.index({mask});

        std::vector<long int> b_vec, a_vec, gi_vec, gj_vec, cls_vec;
        std::vector<float> tbox_vec, anchor_vec;
        long int b_val, cls_val, anchor_id, gi, gj, gi_new, gj_new;
        float gx, gy, gw, gh, anchor_w, anchor_h, fx, fy;
        std::vector<std::pair<int, int>> offsets;

        for (long int idx = 0; idx < t.size(0); idx++){
            b_val = (long int)(t.index({idx, 0}).item<float>());
            cls_val = (long int)(t.index({idx, 1}).item<float>());
            gx = t.index({idx, 2}).item<float>();
            gy = t.index({idx, 3}).item<float>();
            gw = t.index({idx, 4}).item<float>();
            gh = t.index({idx, 5}).item<float>();
            anchor_id = (long int)(t.index({idx, 6}).item<float>());
            anchor_w = anchors.index({idx, 0}).item<float>();
            anchor_h = anchors.index({idx, 1}).item<float>();

            gi = (long int)(std::floor(gx));
            gj = (long int)(std::floor(gy));
            fx = gx - gi;
            fy = gy - gj;

            offsets = {{0, 0}};
            if ((fx < 0.5) && (gi > 0)) offsets.emplace_back(-1, 0);
            if ((fy < 0.5) && (gj > 0)) offsets.emplace_back(0, -1);
            if (((1.0 - fx) < 0.5) && (gi < grid_shapes.at(i).at(1) - 1)) offsets.emplace_back(1, 0);
            if (((1.0 - fy) < 0.5) && (gj < grid_shapes.at(i).at(0) - 1)) offsets.emplace_back(0, 1);

            for (const auto &off : offsets){

                gi_new = gi + off.first;
                gj_new = gj + off.second;

                if ((gi_new < 0) || (gi_new >= grid_shapes.at(i).at(1)) || (gj_new < 0) || (gj_new >= grid_shapes.at(i).at(0))) continue;

                b_vec.push_back(b_val);
                a_vec.push_back(anchor_id);
                gi_vec.push_back(gi_new);
                gj_vec.push_back(gj_new);
                tbox_vec.push_back(gx - gi_new);
                tbox_vec.push_back(gy - gj_new);
                tbox_vec.push_back(gw);
                tbox_vec.push_back(gh);
                cls_vec.push_back(cls_val);
                anchor_vec.push_back(anchor_w);
                anchor_vec.push_back(anchor_h);

            }
        }

        if (b_vec.empty()){
            result_indices.at(i) = torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kLong)).to(device);
            result_tbox.at(i) = torch::empty({0, 4}).to(device);
            result_tclass.at(i) = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong)).to(device);
            result_anchors.at(i) = torch::empty({0, 2}).to(device);
            continue;
        }

        long int total;
        torch::Tensor b_tensor, a_tensor, gi_tensor, gj_tensor;

        total = b_vec.size();

        b_tensor = torch::from_blob(b_vec.data(), {total}, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
        a_tensor = torch::from_blob(a_vec.data(), {total}, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
        gi_tensor = torch::from_blob(gi_vec.data(), {total}, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
        gj_tensor = torch::from_blob(gj_vec.data(), {total}, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
        result_indices.at(i) = torch::stack({b_tensor, a_tensor, gj_tensor, gi_tensor}, 1);

        result_tbox.at(i) = torch::from_blob(tbox_vec.data(), {total, 4}, torch::TensorOptions().dtype(torch::kFloat)).clone().to(device);
        result_tclass.at(i) = torch::from_blob(cls_vec.data(), {total}, torch::TensorOptions().dtype(torch::kLong)).clone().to(device);
        result_anchors.at(i) = torch::from_blob(anchor_vec.data(), {total, 2}, torch::TensorOptions().dtype(torch::kFloat)).clone().to(device);

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


// -------------------------------------
// class{Loss} -> function{smooth_BCE}
// -------------------------------------
std::pair<float, float> Loss::smooth_BCE(float eps){

    float cp, cn;

    eps = std::clamp(eps, 0.0f, 1.0f);
    cp = 1.0 - 0.5 * eps;
    cn = 0.5 * eps;
    return {cp, cn};

}


// -----------------------------------------------
// class{Loss} -> function{binary_cross_entropy}
// -----------------------------------------------
torch::Tensor Loss::binary_cross_entropy(torch::Tensor input, torch::Tensor target, float pos_weight){

    torch::Tensor loss, pred_prob, p_t, alpha_factor, modulating_factor;

    auto options = F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone);
    if (std::abs(pos_weight - 1.0) > 1e-6){
        options = options.pos_weight(torch::full_like(input, pos_weight));
    }
   loss = F::binary_cross_entropy_with_logits(input, target, options);

    if (this->focal_gamma > 0.0){
        pred_prob = torch::sigmoid(input);
        p_t = target * pred_prob + (1.0f - target) * (1.0f - pred_prob);
        alpha_factor = target * this->focal_alpha + (1.0f - target) * (1.0f - this->focal_alpha);
        modulating_factor = torch::pow(1.0f - p_t, this->focal_gamma);
        loss = loss * alpha_factor * modulating_factor;
    }

    return loss.mean();

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
    std::vector<float> strides(scales, 0.0);
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
        strides.at(i) = (stride_x + stride_y) * 0.5;
        anchors_level = this->anchors.index({(long int)i}).to(device).clone();
        anchors_level.index_put_({Slice(), 0}, anchors_level.index({Slice(), 0}) / stride_x);
        anchors_level.index_put_({Slice(), 1}, anchors_level.index({Slice(), 1}) / stride_y);
        scaled_anchors.at(i) = anchors_level;
    }
    std::tie(result_indices, result_tbox, result_tclass, result_anchors) = this->build_target(targets_tensor, grid_shapes, scaled_anchors, device);


    float target_stride, best_diff, diff;
    size_t best_idx;

    if (!this->ssi_initialized && (!strides.empty())){
        target_stride = 16.0f;
        best_idx = 0;
        best_diff = std::numeric_limits<float>::max();
        for (size_t i = 0; i < strides.size(); i++){
            diff = std::fabs(strides.at(i) - target_stride);
            if (diff < best_diff){
                best_diff = diff;
                best_idx = i;
            }
        }
        this->ssi = best_idx;
        this->ssi_initialized = true;
    }

    if (this->autobalance && (this->ssi < this->balance.size())){
        this->balance.at(this->ssi) = 1.0f;
    }

    torch::Tensor target_class;
    float obj_item;
    float base;

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
                target_class = torch::full_like(pred_class, this->cn);
                target_class.scatter_(1, tclass.to(torch::kLong).unsqueeze(1), this->cp);
                loss_class = loss_class + this->binary_cross_entropy(pred_class, target_class, this->class_pos_weight);
            }

        }

        obj_loss = this->binary_cross_entropy(obj_pred, obj_target, this->obj_pos_weight);
        loss_obj = loss_obj + obj_loss * this->balance.at(i);
        if (this->autobalance){
            obj_item = obj_loss.detach().item<float>() + 1e-6;
            this->balance.at(i) = this->balance.at(i) * 0.9999 + 0.0001 / obj_item;
        }

    }

    if (this->autobalance && (this->ssi < this->balance.size())){
        base = this->balance.at(this->ssi);
        if (base > 0.0){
            for (auto &val : this->balance){
                val /= base;
            }
        }
        this->balance.at(this->ssi) = 1.0;
    }


    return {loss_box, loss_obj, loss_class};

}
