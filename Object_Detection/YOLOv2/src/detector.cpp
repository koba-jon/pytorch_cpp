#include <tuple>
#include <vector>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "detector.hpp"

// Define Namespace
namespace F = torch::nn::functional;


// ------------------------------------
// class{YOLODetector} -> constructor
// ------------------------------------
YOLODetector::YOLODetector(const std::vector<std::tuple<float, float>> anchors_, const long int class_num_, const float prob_thresh_, const float nms_thresh_){
    std::vector<float> anchors(anchors_.size() * 2);
    for (size_t i = 0; i < anchors_.size(); i++){
        anchors.at(i * 2) = std::get<0>(anchors_.at(i));
        anchors.at(i * 2 + 1) = std::get<1>(anchors_.at(i));
    }
    this->class_num = class_num_;
    this->na = anchors_.size();
    this->prob_thresh = prob_thresh_;
    this->nms_thresh = nms_thresh_;
    this->anchors_wh = torch::from_blob(anchors.data(), {1, 1, this->na, 2}, torch::kFloat).clone();  // anchors_wh{1,1,A,2}
}


// ----------------------------------------------------
// class{YOLODetector} -> function{NonMaximumSuppression}
// ----------------------------------------------------
torch::Tensor YOLODetector::NonMaximumSuppression(torch::Tensor &coord, torch::Tensor &conf){

    int idx;
    torch::Tensor coord_per;
    torch::Tensor x_min, y_min, x_max, y_max;
    torch::Tensor areas;
    torch::Tensor nms_idx_sorted, nms_idx, idx_keep;
    torch::Tensor left, right, top, bottom;
    torch::Tensor width, height;
    torch::Tensor inter, area1, area2;
    torch::Tensor unions, IoU;
    std::vector<int> nms_idx_vec;

    // (1) Set parameters
    coord_per = coord.permute({1, 0}).contiguous();  // coord{object,4} ===> coord_per{4,object}
    x_min = coord_per[0] - 0.5 * coord_per[2];  // x_min{object}
    y_min = coord_per[1] - 0.5 * coord_per[3];  // y_min{object}
    x_max = coord_per[0] + 0.5 * coord_per[2];  // x_max{object}
    y_max = coord_per[1] + 0.5 * coord_per[3];  // y_max{object}
    areas = (x_max - x_min) * (y_max - y_min);  // areas{object}
    nms_idx_sorted = std::get<1>(conf.sort(/*dim=*/0, /*descending=*/true));  // nms_idx_sorted{object}

    // (2) Extract index of bounding box
    while (nms_idx_sorted.numel() > 0){

        // (2.1) Extract index for top confidence
        idx = (int)nms_idx_sorted[0].item<long int>();  // index for top confidence
        nms_idx_vec.push_back(idx);
        
        // (2.2) Compute coordinate of the intersections
        if (nms_idx_sorted.numel() == 1) break;

        // (2.3) Compute coordinate of the intersections
        nms_idx_sorted = nms_idx_sorted.split_with_sizes(/*split_sizes=*/{1, nms_idx_sorted.numel() - 1}, /*dim=*/0).at(1);  // nms_idx_sorted{num} ===> {num-1}
        left = x_min.gather(/*dim=*/0, /*index=*/nms_idx_sorted).clamp_min(x_min[idx].item<float>());  // x_min{object} ===> left{num-1}
        top = y_min.gather(/*dim=*/0, /*index=*/nms_idx_sorted).clamp_min(y_min[idx].item<float>());  // y_min{object} ===> top{num-1}
        right = x_max.gather(/*dim=*/0, /*index=*/nms_idx_sorted).clamp_max(x_max[idx].item<float>());  // x_max{object} ===> right{num-1}
        bottom = y_max.gather(/*dim=*/0, /*index=*/nms_idx_sorted).clamp_max(y_max[idx].item<float>());  // y_max{object} ===> bottom{num-1}

        // (2.4) Compute area of the intersections from the coordinates
        width = (right - left).clamp_min(0.0);  // right{num-1}, left{num-1} ===> width{num-1}
        height = (bottom - top).clamp_min(0.0);  // bottom{num-1}, top{num-1} ===> height{num-1}
        inter = width * height;  // width{num-1}, height{num-1} ===> inter{num-1}

        // (2.5) Compute area of the bounding boxes
        area1 = areas[idx];  // areas{object} ===> area1{}
        area2 = areas.gather(/*dim=*/0, /*index=*/nms_idx_sorted);  // areas{object} ===> area2{num-1}

        // (2.6) Compute IoU from the areas
        unions = area1 + area2 - inter;  // area1{}, area2{num-1}, inter{num-1} ===> unions{num-1}
        IoU = inter / unions;  // inter{num-1}, unions{num-1} ===> IoU{num-1}

        // (2.7) Remove bounding box whose IoU is higher than the threshold
        idx_keep = (IoU <= this->nms_thresh).nonzero().squeeze(/*dim=*/-1);  // IoU{num-1} ===> idx_keep{num_next}
        if (idx_keep.numel() == 0) break;
        nms_idx_sorted = nms_idx_sorted.gather(/*dim=*/0, /*index=*/idx_keep);  // nms_idx_sorted{num-1} ===> {num_next}

    }

    // (3) Set index of final bounding box
    nms_idx = torch::from_blob(nms_idx_vec.data(), {(long int)nms_idx_vec.size()}, torch::kInt).to(torch::kLong).to(conf.device()).clone();  // {nms-object}

    return nms_idx;

}


// ----------------------------------------------------
// class{YOLODetector} -> function{get_label_palette}
// ----------------------------------------------------
std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> YOLODetector::get_label_palette(){

    constexpr double H_max = 360.0;

    double step, H;
    unsigned char R, G, B;
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette;

    step = H_max / (double)this->class_num;
    label_palette = std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>(this->class_num);
    for (long int i = 0; i < this->class_num; i++){
        H = step * (double)i;
        if (H <= 60.0){
            R = 255;
            G = (unsigned char)(H / 60.0 * 255.0 + 0.5);
            B = 0;
        }
        else if (H <= 120.0){
            R = (unsigned char)((120.0 - H) / 60.0 * 255.0 + 0.5);
            G = 255;
            B = 0;
        }
        else if (H <= 180.0){
            R = 0;
            G = 255;
            B = (unsigned char)((H - 120.0) / 60.0 * 255.0 + 0.5);
        }
        else if (H <= 240.0){
            R = 0;
            G = (unsigned char)((240.0 - H) / 60.0 * 255.0 + 0.5);
            B = 255;
        }
        else if (H <= 300.0){
            R = (unsigned char)((H - 240.0) / 60.0 * 255.0 + 0.5);
            G = 0;
            B = 255;
        }
        else {
            R = 255;
            G = 0;
            B = (unsigned char)((360.0 - H) / 60.0 * 255.0 + 0.5);
        }
        label_palette.at(i) = {R, G, B};
    }

    return label_palette;
    
}


// ------------------------------------
// class{YOLODetector} -> operator
// ------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> YOLODetector::operator()(const torch::Tensor pred){

    torch::Device device = pred.device();
    long int ng = pred.size(0);

    // (1) Activate predicted tensor
    std::vector<torch::Tensor> pred_split;
    torch::Tensor arange, x0, y0, x0y0;
    torch::Tensor pred_view, pred_class, pred_xy, pred_wh, pred_conf, pred_coord, pred_coord_per;
    /*************************************************************************/
    arange = torch::arange(/*start=*/0.0, /*end=*/(float)ng, /*step=*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);  // arange{G} = [0,1,2,...,G-1]
    x0 = arange.view({1, ng, 1, 1}).expand({ng, ng, 1, 1});  // arange{G} ===> {1,G,1,1} ===> x0{G,G,1,1}
    y0 = arange.view({ng, 1, 1, 1}).expand({ng, ng, 1, 1});  // arange{G} ===> {G,1,1,1} ===> y0{G,G,1,1}
    x0y0 = torch::cat({x0, y0}, /*dim=*/3);  // x0{G,G,1,1} + y0{G,G,1,1} ===> x0y0{G,G,1,2}
    /*************************************************************************/
    pred_view = pred.view({ng, ng, this->na, this->class_num + 5});  // pred{G,G,A*(CN+5)} ===> pred_view{G,G,A,CN+5}
    pred_split = pred_view.split_with_sizes(/*split_sizes=*/{this->class_num, 2, 2, 1}, /*dim=*/3);  // pred_view{G,G,A,CN+5} ===> pred_split({G,G,A,CN}, {G,G,A,2}, {G,G,A,2}, {G,G,A,1})
    pred_class = F::softmax(pred_split.at(0), /*dim=*/3);  // pred_class{G,G,A,CN}
    pred_xy = (torch::sigmoid(pred_split.at(1)) + x0y0) / (float)ng;  // pred_xy{G,G,A,2}
    pred_wh = torch::exp(pred_split.at(2)) * this->anchors_wh.to(device) / (float)ng;  // pred_wh{G,G,A,2}
    pred_conf = torch::sigmoid(pred_split.at(3)).squeeze(-1);  // pred_conf{G,G,A}
    pred_coord = torch::cat({pred_xy, pred_wh}, /*dim=*/3);  // pred_xy{G,G,A,2} + pred_wh{G,G,A,2} ===> pred_coord{G,G,A,4}
    pred_coord_per = pred_coord.permute({3, 0, 1, 2}).contiguous();  // pred_coord{G,G,A,4} ===> pred_coord_per{4,G,G,A}

    // (2) Get object mask
    torch::Tensor class_score, ids, prob, obj_mask;
    std::tuple<torch::Tensor, torch::Tensor> class_score_with_idx;
    /*************************************************************************/
    class_score_with_idx = pred_class.max(/*dim=*/3, /*keepdim=*/false);  // pred_class{G,G,A,CN} ===> class_score_with_idx({G,G,A}, {G,G,A})
    class_score = std::get<0>(class_score_with_idx);  // class_score{G,G,A}
    ids = std::get<1>(class_score_with_idx);  // ids{G,G,A}
    prob = pred_conf * class_score;  // pred_conf{G,G,A}, class_score{G,G,A} ===> prob{G,G,A}
    obj_mask = (prob >= this->prob_thresh);  // prob{G,G,A} ===> obj_mask{G,G,A}

    // (3) Get ids
    torch::Tensor obj_ids;
    /*************************************************************************/
    obj_ids = ids.masked_select(/*mask=*/obj_mask);  // ids{G,G,A} ===> obj_ids{object}

    // (4) Get coordinates
    torch::Tensor obj_cx, obj_cy, obj_w, obj_h, obj_coord;
    /*************************************************************************/
    obj_cx = pred_coord_per[0].masked_select(/*mask=*/obj_mask);  // pred_coord_per{4,G,G,A} ===> obj_cx{object}
    obj_cy = pred_coord_per[1].masked_select(/*mask=*/obj_mask);  // pred_coord_per{4,G,G,A} ===> obj_cy{object}
    obj_w = pred_coord_per[2].masked_select(/*mask=*/obj_mask);  // pred_coord_per{4,G,G,A} ===> obj_w{object}
    obj_h = pred_coord_per[3].masked_select(/*mask=*/obj_mask);  // pred_coord_per{4,G,G,A} ===> obj_h{object}
    obj_coord = torch::cat({obj_cx.unsqueeze(/*dim=*/-1), obj_cy.unsqueeze(/*dim=*/-1), obj_w.unsqueeze(/*dim=*/-1), obj_h.unsqueeze(/*dim=*/-1)}, /*dim=*/1);  // obj_coord{object,4}

    // (5) Get confidences
    torch::Tensor obj_conf;
    /*************************************************************************/
    obj_conf = pred_conf.masked_select(/*mask=*/obj_mask);  // pred_conf{G,G,A} ===> obj_conf{object}

    // (6) Get probabilities
    torch::Tensor obj_prob;
    /*************************************************************************/
    obj_prob = prob.masked_select(/*mask=*/obj_mask);  // prob{G,G,A} ===> obj_prob{object}

    // (7) Apply Non-Maximum Suppression
    torch::Tensor nms_idx;
    torch::Tensor final_ids, final_coord, final_prob;
    /*************************************************************************/
    if (obj_ids.numel() > 0){
        nms_idx = this->NonMaximumSuppression(obj_coord, obj_conf);  // obj_coord{object,4}, obj_conf{object} ===> nms_idx{nms-object}
        final_ids = obj_ids.gather(/*dim=*/0, /*index=*/nms_idx).contiguous().detach().clone();  // obj_ids{object} ===> final_ids{nms-object}
        final_coord = obj_coord.gather(/*dim=*/0, /*index=*/nms_idx.unsqueeze(/*dim=*/-1).expand({nms_idx.size(0), 4})).contiguous().detach().clone();  // obj_coord{object,4} ===> final_coord{nms-object,4}
        final_prob = obj_prob.gather(/*dim=*/0, /*index=*/nms_idx).contiguous().detach().clone();  // obj_prob{object} ===> final_prob{nms-object}
    }

    return {final_ids, final_coord, final_prob};  // {nms-object} (ids), {nms-object,4} (coordinates), {nms-object} (probabilities)
    
}

