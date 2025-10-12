#include <tuple>
#include <vector>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "detector.hpp"


// ------------------------------------
// class{YOLODetector} -> constructor
// ------------------------------------
YOLODetector::YOLODetector(const std::vector<std::vector<std::tuple<float, float>>> anchors_, const long int class_num_, const float prob_thresh_, const float nms_thresh_){

    long int scales = anchors_.size();
    this->na = anchors_.at(0).size();

    this->anchors = torch::zeros({scales, 1, 1, this->na, 2}, torch::TensorOptions().dtype(torch::kFloat));  // {S,A,2}
    for (long int i = 0; i < scales; i++){
        for (long int j = 0; j < this->na; j++){
            this->anchors.index_put_({i, 0, 0, j, 0}, std::get<0>(anchors_.at(i).at(j)));
            this->anchors.index_put_({i, 0, 0, j, 1}, std::get<1>(anchors_.at(i).at(j)));
        }
    }

    this->class_num = class_num_;
    this->prob_thresh = prob_thresh_;
    this->nms_thresh = nms_thresh_;

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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> YOLODetector::operator()(const std::vector<torch::Tensor> preds, const std::tuple<float, float> image_sizes){

    torch::Device device = preds.at(0).device();
    size_t scales = preds.size();

    // (1) Set image size
    std::vector<float> image_size_vec(2);
    torch::Tensor image_size;
    /*************************************************************************/
    image_size_vec.at(0) = std::get<0>(image_sizes);
    image_size_vec.at(1) = std::get<1>(image_sizes);
    image_size = torch::from_blob(image_size_vec.data(), {1, 1, 1, 2}, torch::kFloat).to(device).clone();  // image_size{1,1,1,2}

    // (2) Set object tensor
    std::vector<torch::Tensor> obj_id_vec, obj_coord_vec, obj_conf_vec, obj_prob_vec;

    for (size_t i = 0; i < scales; i++){

        torch::Tensor pred = preds.at(i);
        long int ng = pred.size(0);

        // (3.1) Activate predicted tensor
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
        pred_split = pred_view.split_with_sizes(/*split_sizes=*/{2, 2, 1, this->class_num}, /*dim=*/3);  // pred_view{G,G,A,CN+5} ===> pred_split({G,G,A,CN}, {G,G,A,2}, {G,G,A,2}, {G,G,A,1})
        pred_xy = (torch::sigmoid(pred_split.at(0)) * 2.0 - 0.5 + x0y0) / (float)ng;  // pred_xy{G,G,A,2}
        pred_wh = (torch::sigmoid(pred_split.at(1)) * 2.0).pow(2.0) * this->anchors[i].to(device) / image_size;  // pred_wh{G,G,A,2}
        pred_conf = torch::sigmoid(pred_split.at(2)).squeeze(-1);  // pred_conf{G,G,A}
        pred_class = torch::sigmoid(pred_split.at(3));  // pred_class{G,G,A,CN}
        pred_coord = torch::cat({pred_xy, pred_wh}, /*dim=*/3);  // pred_xy{G,G,A,2} + pred_wh{G,G,A,2} ===> pred_coord{G,G,A,4}
        pred_coord_per = pred_coord.permute({3, 0, 1, 2}).contiguous();  // pred_coord{G,G,A,4} ===> pred_coord_per{4,G,G,A}

        // (3.2) Get object mask
        torch::Tensor class_score, id, prob, obj_mask;
        std::tuple<torch::Tensor, torch::Tensor> class_score_with_idx;
        /*************************************************************************/
        class_score_with_idx = pred_class.max(/*dim=*/3, /*keepdim=*/false);  // pred_class{G,G,A,CN} ===> class_score_with_idx({G,G,A}, {G,G,A})
        class_score = std::get<0>(class_score_with_idx);  // class_score{G,G,A}
        id = std::get<1>(class_score_with_idx);  // id{G,G,A}
        prob = pred_conf * class_score;  // pred_conf{G,G,A}, class_score{G,G,A} ===> prob{G,G,A}
        obj_mask = (prob >= this->prob_thresh);  // prob{G,G,A} ===> obj_mask{G,G,A}

        // (4.1) Get ids
        torch::Tensor obj_id;
        /*************************************************************************/
        obj_id = id.masked_select(/*mask=*/obj_mask);  // id{G,G,A} ===> obj_id{object}
        obj_id_vec.push_back(obj_id);  // obj_id_vec{scales,{object}}

        // (4.2) Get coordinates
        torch::Tensor obj_cx, obj_cy, obj_w, obj_h, obj_coord;
        /*************************************************************************/
        obj_cx = pred_coord_per[0].masked_select(/*mask=*/obj_mask);  // pred_coord_per{4,G,G,A} ===> obj_cx{object}
        obj_cy = pred_coord_per[1].masked_select(/*mask=*/obj_mask);  // pred_coord_per{4,G,G,A} ===> obj_cy{object}
        obj_w = pred_coord_per[2].masked_select(/*mask=*/obj_mask);  // pred_coord_per{4,G,G,A} ===> obj_w{object}
        obj_h = pred_coord_per[3].masked_select(/*mask=*/obj_mask);  // pred_coord_per{4,G,G,A} ===> obj_h{object}
        obj_coord = torch::cat({obj_cx.unsqueeze(/*dim=*/-1), obj_cy.unsqueeze(/*dim=*/-1), obj_w.unsqueeze(/*dim=*/-1), obj_h.unsqueeze(/*dim=*/-1)}, /*dim=*/1);  // obj_coord{object,4}
        obj_coord_vec.push_back(obj_coord);  // obj_coord_vec{scales,{object,4}}

        // (4.3) Get confidences
        torch::Tensor obj_conf;
        /*************************************************************************/
        obj_conf = pred_conf.masked_select(/*mask=*/obj_mask);  // pred_conf{G,G,A} ===> obj_conf{object}
        obj_conf_vec.push_back(obj_conf);  // obj_conf_vec{scales,{object}}

        // (4.4) Get probabilities
        torch::Tensor obj_prob;
        /*************************************************************************/
        obj_prob = prob.masked_select(/*mask=*/obj_mask);  // prob{G,G,A} ===> obj_prob{object}
        obj_prob_vec.push_back(obj_prob);  // obj_prob_vec{scales,{object}}
    
    }

    // (5) Concatenate object tensor at all scales
    torch::Tensor obj_ids, obj_coords, obj_confs, obj_probs;
    /*************************************************************************/
    obj_ids = obj_id_vec.at(0);  // obj_id_vec[0]{object} ===> obj_ids{object}
    obj_coords = obj_coord_vec.at(0);  // obj_coord_vec[0]{object,4} ===> obj_coords{objects,4}
    obj_confs = obj_conf_vec.at(0);  // obj_conf_vec[0]{object} ===> obj_confs{objects}
    obj_probs = obj_prob_vec.at(0);  // obj_prob_vec[0]{object} ===> obj_probs{objects}
    for (size_t i = 1; i < obj_id_vec.size(); i++){
        obj_ids = torch::cat({obj_ids, obj_id_vec.at(i)}, /*dim=*/0);  // obj_ids{objects} + obj_id_vec[i]{object} ===> obj_ids{objects + object}
        obj_coords = torch::cat({obj_coords, obj_coord_vec.at(i)}, /*dim=*/0);  // obj_coords{objects,4} + obj_coord_vec[i]{object,4} ===> obj_coords{objects + object,4}
        obj_confs = torch::cat({obj_confs, obj_conf_vec.at(i)}, /*dim=*/0);  // obj_confs{objects} + obj_conf_vec[i]{object} ===> obj_confs{objects + object}
        obj_probs = torch::cat({obj_probs, obj_prob_vec.at(i)}, /*dim=*/0);  // obj_probs{objects} + obj_prob_vec[i]{object} ===> obj_probs{objects + object}
    }

    // (6) Apply Non-Maximum Suppression
    torch::Tensor nms_idx;
    torch::Tensor final_ids, final_coords, final_probs;
    /*************************************************************************/
    if (obj_ids.numel() > 0){
        nms_idx = this->NonMaximumSuppression(obj_coords, obj_confs);  // obj_coords{objects,4}, obj_confs{objects} ===> nms_idx{nms-object}
        final_ids = obj_ids.gather(/*dim=*/0, /*index=*/nms_idx).contiguous().detach().clone();  // obj_ids{objects} ===> final_ids{nms-object}
        final_coords = obj_coords.gather(/*dim=*/0, /*index=*/nms_idx.unsqueeze(/*dim=*/-1).expand({nms_idx.size(0), 4})).contiguous().detach().clone();  // obj_coords{objects,4} ===> final_coords{nms-object,4}
        final_probs = obj_probs.gather(/*dim=*/0, /*index=*/nms_idx).contiguous().detach().clone();  // obj_probs{objects} ===> final_probs{nms-object}
    }

    return {final_ids, final_coords, final_probs};  // {nms-object} (ids), {nms-object,4} (coordinates), {nms-object} (probabilities)
    
}

