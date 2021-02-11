#include <tuple>
#include <vector>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "losses.hpp"


// -----------------------------------
// class{Loss} -> constructor
// -----------------------------------
Loss::Loss(const long int class_num_, const long int ng_, const long int nb_){
    this->class_num = class_num_;
    this->ng = ng_;
    this->nb = nb_;
}


// -------------------------------------
// class{Loss} -> function{make_target}
// -------------------------------------
std::tuple<torch::Tensor, torch::Tensor> Loss::make_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target){

    size_t i, j;
    size_t BB_n;
    size_t mini_batch_size;
    long int id;
    float cx, cy, w, h, x0, y0;
    float cell_size;
    torch::Tensor target_class;
    torch::Tensor target_coord;
    torch::Tensor ids, coords;
    std::tuple<torch::Tensor, torch::Tensor> data; 

    mini_batch_size = target.size();  // mini batch size
    cell_size = 1.0 / (float)this->ng;  // cell_size = 1/G
    target_class = torch::full({(long int)mini_batch_size, this->ng, this->ng, this->class_num}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat));  // target_class{N,G,G,CN}
    target_coord = torch::full({(long int)mini_batch_size, this->ng, this->ng, 5}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat));  // target_coord{N,G,G,5}

    for (size_t n = 0; n < mini_batch_size; n++){
        data = target.at(n);
        ids = std::get<0>(data);  // {BB_n}
        coords = std::get<1>(data);  // {BB_n,4}
        BB_n = ids.size(0);
        for (size_t b = 0; b < BB_n; b++){
            id = ids[b].item<long int>();  // id[0,CN)
            cx = coords[b][0].item<float>();  // cx[0.0,1.0)
            cy = coords[b][1].item<float>();  // cy[0.0,1.0)
            w = coords[b][2].item<float>();  // w[0.0,1.0)
            h = coords[b][3].item<float>();  // h[0.0,1.0)
            i = (size_t)(cx / cell_size);  // i[0,G)
            j = (size_t)(cy / cell_size);  // j[0,G)
            x0 = (float)i * cell_size;  // x0[0.0,1.0)
            y0 = (float)j * cell_size;  // y0[0.0,1.0)
            target_class[n][j][i][id].data() = 1.0;  // class likelihood = 1.0
            target_coord[n][j][i][0].data() = (cx - x0) / cell_size;  // center of x[0.0,1.0)
            target_coord[n][j][i][1].data() = (cy - y0) / cell_size;  // center of y[0.0,1.0)
            target_coord[n][j][i][2].data() = w;  // width[0.0,1.0)
            target_coord[n][j][i][3].data() = h;  // height[0.0,1.0)
            target_coord[n][j][i][4].data() = 1.0;  // object confidence = 1.0
        }
    }
    
    return {target_class.detach().clone(), target_coord.detach().clone()};  // target_class{N,G,G,CN}, target_coord{N,G,G,5}
    
}


// -------------------------------------
// class{Loss} -> function{rescale}
// -------------------------------------
torch::Tensor Loss::rescale(torch::Tensor &BBs){
    torch::Tensor x_min, y_min, x_max, y_max, output;
    x_min = BBs[0] / (float)this->ng - 0.5 * BBs[2];  // x_min{BB,object}
    y_min = BBs[1] / (float)this->ng - 0.5 * BBs[3];  // y_min{BB,object}
    x_max = BBs[0] / (float)this->ng + 0.5 * BBs[2];  // x_max{BB,object}
    y_max = BBs[1] / (float)this->ng + 0.5 * BBs[3];  // y_max{BB,object}
    output = torch::cat({x_min.unsqueeze(/*dim=*/0), y_min.unsqueeze(/*dim=*/0), x_max.unsqueeze(/*dim=*/0), y_max.unsqueeze(/*dim=*/0)}, /*dim=*/0);  // output{4,BB,object}
    return output;
}


// -------------------------------------
// class{Loss} -> function{compute_iou}
// -------------------------------------
torch::Tensor Loss::compute_iou(torch::Tensor &BBs1, torch::Tensor &BBs2){

    long int BB1, BB2, obj;
    torch::Tensor left, right, top, bottom;
    torch::Tensor width, height;
    torch::Tensor inter, area1, area2;
    torch::Tensor unions, IoU;

    BB1 = BBs1.size(1);
    BB2 = BBs2.size(1);
    obj = BBs1.size(2);

    // (1) Compute left coordinate of the intersections
    left = std::get<0>(  // left{BB1,BB2,object}
        torch::cat(
            {
                BBs1[0].unsqueeze(/*dim=*/1).unsqueeze(/*dim=*/0).expand({1,BB1,BB2,obj}),  // BBs1{4,BB1,object} ===> {1,BB1,BB2,object}
                BBs2[0].unsqueeze(/*dim=*/0).unsqueeze(/*dim=*/0).expand({1,BB1,BB2,obj})   // BBs2{4,BB2,object} ===> {1,BB1,BB2,object}
            }
            , /*dim=*/0
        )
        .max(/*dim=*/0, /*keepdim=*/false)
    );

    // (2) Compute top coordinate of the intersections
    top = std::get<0>(  // top{BB1,BB2,object}
        torch::cat(
            {
                BBs1[1].unsqueeze(/*dim=*/1).unsqueeze(/*dim=*/0).expand({1,BB1,BB2,obj}),  // BBs1{4,BB1,object} ===> {1,BB1,BB2,object}
                BBs2[1].unsqueeze(/*dim=*/0).unsqueeze(/*dim=*/0).expand({1,BB1,BB2,obj})   // BBs2{4,BB2,object} ===> {1,BB1,BB2,object}
            }
            , /*dim=*/0
        )
        .max(/*dim=*/0, /*keepdim=*/false)
    );

    // (3) Compute right coordinate of the intersections
    right = std::get<0>(  // right{BB1,BB2,object}
        torch::cat(
            {
                BBs1[2].unsqueeze(/*dim=*/1).unsqueeze(/*dim=*/0).expand({1,BB1,BB2,obj}),  // BBs1{4,BB1,object} ===> {1,BB1,BB2,object}
                BBs2[2].unsqueeze(/*dim=*/0).unsqueeze(/*dim=*/0).expand({1,BB1,BB2,obj})   // BBs2{4,BB2,object} ===> {1,BB1,BB2,object}
            }
            , /*dim=*/0
        )
        .min(/*dim=*/0, /*keepdim=*/false)
    );

    // (4) Compute bottom coordinate of the intersections
    bottom = std::get<0>(  // bottom{BB1,BB2,object}
        torch::cat(
            {
                BBs1[3].unsqueeze(/*dim=*/1).unsqueeze(/*dim=*/0).expand({1,BB1,BB2,obj}),  // BBs1{4,BB1,object} ===> {1,BB1,BB2,object}
                BBs2[3].unsqueeze(/*dim=*/0).unsqueeze(/*dim=*/0).expand({1,BB1,BB2,obj})   // BBs2{4,BB2,object} ===> {1,BB1,BB2,object}
            }
            , /*dim=*/0
        )
        .min(/*dim=*/0, /*keepdim=*/false)
    );

    // (5) Compute area of the intersections from the coordinates
    width = (right - left).clamp_min(/*min=*/0.0);  // right{BB1,BB2,object}, left{BB1,BB2,object} ===> width{BB1,BB2,object}
    height = (bottom - top).clamp_min(/*min=*/0.0);  // bottom{BB1,BB2,object}, top{BB1,BB2,object} ===> height{BB1,BB2,object}
    inter = width * height;  // width{BB1,BB2,object}, height{BB1,BB2,object} ===> inter{BB1,BB2,object}

    // (6) Compute area of the bounding boxes
    area1 = (BBs1[2] - BBs1[0]) * (BBs1[3] - BBs1[1]);  // area1{BB1,object}
    area2 = (BBs2[2] - BBs2[0]) * (BBs2[3] - BBs2[1]);  // area2{BB2,object}
    area1 =  area1.unsqueeze(/*dim=*/1).expand_as(inter);  // area1{BB1,object} ===> {BB1,BB2,object}
    area2 =  area2.unsqueeze(/*dim=*/0).expand_as(inter);  // area2{BB2,object} ===> {BB1,BB2,object}

    // (7) Compute IoU from the areas
    unions = area1 + area2 - inter;  // area1{BB1,BB2,object}, area2{BB1,BB2,object}, inter{BB1,BB2,object} ===> unions{BB1,BB2,object}
    IoU = inter / unions;  // inter{BB1,BB2,object}, unions{BB1,BB2,object} ===> IoU{BB1,BB2,object}

    return IoU;

}


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Loss::operator()(torch::Tensor &input, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target){

    // -----------------------------------
    // 1. Preparation
    // -----------------------------------

    static auto criterion = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kSum));
    torch::Device device = input.device();

    // (1) Set input tensor
    std::vector<torch::Tensor> input_new;
    torch::Tensor input_class, input_coord, input_coord_new, input_coord_per, input_conf;
    /*************************************************************************/
    input_new = input.split_with_sizes(/*split_sizes=*/{this->class_num, this->nb * 5}, /*dim=*/3);  // input{N,G,G,FF} ===> input_new({N,G,G,CN}, {N,G,G,BB*5})
    input_class = input_new.at(0);  // input_class{N,G,G,CN}
    input_coord = input_new.at(1);  // input_coord{N,G,G,BB*5}
    input_coord_new = input_coord.view({input_coord.size(0), input_coord.size(1), input_coord.size(2), this->nb, 5});  // input_coord{N,G,G,BB*5} ===> input_coord_new{N,G,G,BB,5}
    input_coord_per = input_coord_new.permute({4, 0, 1, 2, 3}).contiguous();  // input_coord_new{N,G,G,BB,5} ===> input_coord_per{5,N,G,G,BB}
    input_conf = input_coord_per[4];  // input_coord_per{5,N,G,G,BB} ===> input_conf{N,G,G,BB}

    // (2) Set target tensor
    std::tuple<torch::Tensor, torch::Tensor> target_new;
    torch::Tensor target_class, target_coord, target_coord_new, target_coord_per, target_conf;
    /*************************************************************************/
    target_new = this->make_target(target);  // target{N, ({BB_n}, {BB_n,4}) } ===> target_new({N,G,G,CN}, {N,G,G,5})
    target_class = std::get<0>(target_new).to(device);  // target_class{N,G,G,CN}
    target_coord = std::get<1>(target_new).to(device);  // target_coord{N,G,G,5}
    target_coord_new = target_coord.unsqueeze(/*dim=*/3);  // target_coord{N,G,G,5} ===> target_coord_new{N,G,G,1,5}
    target_coord_per = target_coord.permute({3, 0, 1, 2}).contiguous();  // target_coord{N,G,G,5} ===> target_coord_per{5,N,G,G}
    target_conf = target_coord_per[4];  // target_coord_per{5,N,G,G} ===> target_conf{N,G,G}
    
    // (3) Extract response coordinate
    bool obj_flag;
    torch::Tensor target_conf_coord;
    torch::Tensor input_obj_coord_mask, input_obj_coord, input_BBs_normalized, input_BBs;
    torch::Tensor target_obj_coord_mask, target_obj_coord, target_BBs_normalized, target_BBs;
    torch::Tensor IoU, max_IoU, max_idx, response_idx;
    torch::Tensor input_response_coord, target_response_coord;
    std::tuple<torch::Tensor, torch::Tensor> max_IoU_with_idx;
    /*************************************************************************/
    target_conf_coord = target_conf.unsqueeze(/*dim=*/-1).unsqueeze(/*dim=*/-1);  // target_conf{N,G,G} ===> target_conf_coord{N,G,G,1,1}
    input_obj_coord_mask = (target_conf_coord > 0.0).expand({target_conf_coord.size(0), target_conf_coord.size(1), target_conf_coord.size(2), this->nb, 5});  // target_conf_coord{N,G,G,1,1} ===> input_obj_coord_mask{N,G,G,BB,5}
    target_obj_coord_mask = (target_conf_coord > 0.0).expand({target_conf_coord.size(0), target_conf_coord.size(1), target_conf_coord.size(2), 1, 5});  // target_conf_coord{N,G,G,1,1} ===> target_obj_coord_mask{N,G,G,1,5}
    input_obj_coord = input_coord_new.masked_select(/*mask=*/input_obj_coord_mask).view({-1, this->nb, 5});  // input_coord_new{N,G,G,BB,5} ===> input_obj_coord{object,BB,5}
    target_obj_coord = target_coord_new.masked_select(/*mask=*/target_obj_coord_mask).view({-1, 1, 5});  // target_coord_new{N,G,G,1,5} ===> target_obj_coord{object,1,5}
    obj_flag = (input_obj_coord.numel() > 0);
    if (obj_flag){
        input_BBs_normalized = input_obj_coord.permute({2, 1, 0}).contiguous();  // input_obj_coord{object,BB,5} ===> input_BBs_normalized{5,BB,object}
        target_BBs_normalized = target_obj_coord.permute({2, 1, 0}).contiguous();  // target_obj_coord{object,1,5} ===> target_BBs_normalized{5,1,object}
        input_BBs = this->rescale(input_BBs_normalized);  // input_BBs_normalized{5,BB,object} ===rescale===> input_BBs{4,BB,object}
        target_BBs = this->rescale(target_BBs_normalized);  // target_BBs_normalized{5,1,object} ===rescale===> target_BBs{4,1,object}
        IoU = this->compute_iou(input_BBs, target_BBs).squeeze(/*dim=*/1);  // input_BBs{4,BB,object}, target_BBs{4,1,object} ===> IoU{BB,object}
        max_IoU_with_idx = IoU.max(/*dim=*/0, /*keepdim=*/false);  // IoU{BB,object} ===> max_IoU_with_idx(IoU{object}, idx{object})
        max_IoU = std::get<0>(max_IoU_with_idx);  // max_IoU_with_idx(IoU{object}, idx{object}) ===> max_IoU{object}
        max_idx = std::get<1>(max_IoU_with_idx);  // max_IoU_with_idx(IoU{object}, idx{object}) ===> max_idx{object}
        response_idx = max_idx.unsqueeze(/*dim=*/0).unsqueeze(/*dim=*/0).expand({5, 1, max_idx.size(0)});  // max_idx{object} ===> response_idx{5,1,object}
        input_response_coord = input_BBs_normalized.gather(/*dim=*/1, /*index=*/response_idx).squeeze(/*dim=*/1);  // input_BBs_normalized{5,BB,object} ===> input_response_coord{5,object}
        target_response_coord = target_BBs_normalized.squeeze(/*dim=*/1);  // target_BBs_normalized{5,1,object} ===> target_response_coord{5,object}
    }

    // -----------------------------------
    // 2. Calculation of Loss
    // -----------------------------------

    // (1) "center coordinate term"
    torch::Tensor input_response_cx, target_response_cx, input_response_cy, target_response_cy, loss_coord_xy;
    /*************************************************************************/
    if (obj_flag){
        input_response_cx = input_response_coord[0];  // input_response_coord{5,object} ===> input_response_cx{object}
        target_response_cx = target_response_coord[0];  // target_response_coord{5,object} ===> target_response_cx{object}
        input_response_cy = input_response_coord[1];  // input_response_coord{5,object} ===> input_response_cy{object}
        target_response_cy = target_response_coord[1];  // target_response_coord{5,object} ===> target_response_cy{object}
        loss_coord_xy = criterion(input_response_cx, target_response_cx) + criterion(input_response_cy, target_response_cy);
    }
    else {
        loss_coord_xy = torch::full({}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    }

    // (2) "range coordinate term"
    torch::Tensor input_response_w, target_response_w, input_response_h, target_response_h, loss_coord_wh;
    /*************************************************************************/
    if (obj_flag){
        input_response_w = input_response_coord[2];  // input_response_coord{5,object} ===> input_response_w{object}
        target_response_w = target_response_coord[2];  // target_response_coord{5,object} ===> target_response_w{object}
        input_response_h = input_response_coord[3];  // input_response_coord{5,object} ===> input_response_h{object}
        target_response_h = target_response_coord[3];  // target_response_coord{5,object} ===> target_response_h{object}
        loss_coord_wh = criterion(input_response_w.sqrt(), target_response_w.sqrt()) + criterion(input_response_h.sqrt(), target_response_h.sqrt());
    }
    else {
        loss_coord_wh = torch::full({}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    }

    // (3) "object confidence term"
    torch::Tensor input_response_conf, loss_obj;
    /*************************************************************************/
    if (obj_flag){
        input_response_conf = input_response_coord[4];  // input_response_coord{5,object} ===> input_response_conf{object}
        loss_obj = criterion(input_response_conf, max_IoU);
    }
    else {
        loss_obj = torch::full({}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    }
    
    // (4) "no object confidence term"
    torch::Tensor target_conf_conf, noobj_conf_mask, input_noobj_conf, target_noobj_conf, loss_noobj;
    /*************************************************************************/
    target_conf_conf = target_conf.unsqueeze(/*dim=*/-1).expand({target_conf.size(0), target_conf.size(1), target_conf.size(2), this->nb});  // target_conf{N,G,G} ===> target_conf_conf{N,G,G,BB}
    noobj_conf_mask = (target_conf_conf == 0.0);  // target_conf_conf{N,G,G,BB} ===> noobj_conf_mask{N,G,G,BB}
    input_noobj_conf = input_conf.masked_select(/*mask=*/noobj_conf_mask);  // input_conf{N,G,G,BB} ===> input_noobj_conf{no object confidence}
    target_noobj_conf = target_conf_conf.masked_select(/*mask=*/noobj_conf_mask);  // target_conf_conf{N,G,G,BB} ===> target_noobj_conf{no object confidence}
    loss_noobj = criterion(input_noobj_conf, target_noobj_conf);
    
    // (5) "class term"
    torch::Tensor target_conf_class, obj_class_mask, input_obj_class, target_obj_class, loss_class;
    /*************************************************************************/
    if (obj_flag){
        target_conf_class = target_conf.unsqueeze(/*dim=*/-1).expand({target_conf.size(0), target_conf.size(1), target_conf.size(2), this->class_num});  // target_conf{N,G,G} ===> target_conf_class{N,G,G,CN}
        obj_class_mask = (target_conf_class > 0.0);  // target_conf_class{N,G,G,CN} ===> obj_class_mask{N,G,G,CN}
        input_obj_class = input_class.masked_select(/*mask=*/obj_class_mask);  // input_class{N,G,G,CN} ===> input_noobj_conf{CN}
        target_obj_class = target_class.masked_select(/*mask=*/obj_class_mask);  // target_class{N,G,G,CN} ===> target_noobj_conf{object class}
        loss_class = criterion(input_obj_class, target_obj_class);
    }
    else{
        loss_class = torch::full({}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    }

    return {loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class};

}
