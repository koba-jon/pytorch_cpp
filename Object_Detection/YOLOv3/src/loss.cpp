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
Loss::Loss(const std::vector<std::vector<std::tuple<float, float>>> anchors_, const long int class_num_, const float ignore_thresh_){

    long int scales = anchors_.size();
    this->na = anchors_.at(0).size();

    std::vector<float> anchors(scales * this->na * 2);
    for (long int i = 0; i < scales; i++){
        for (long int j = 0; j < this->na; j++){
            anchors.at(i * this->na * 2 + j * 2) = std::get<0>(anchors_.at(i).at(j));
            anchors.at(i * this->na * 2 + j * 2 + 1) = std::get<1>(anchors_.at(i).at(j));
        }
    }

    this->class_num = class_num_;
    this->ignore_thresh = ignore_thresh_;
    this->anchors_wh = torch::from_blob(anchors.data(), {scales, 1, 1, 1, this->na, 2}, torch::kFloat).clone();  // anchors_wh{S,1,1,1,A,2}

}


// -------------------------------------
// class{Loss} -> function{build_target}
// -------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Loss::build_target(std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target, const long int ng){

    size_t i, j;
    size_t BB_n;
    size_t mini_batch_size;
    long int id;
    float cx, cy, w, h;
    float cell_size;
    torch::Tensor target_class;
    torch::Tensor target_coord;
    torch::Tensor target_mask;
    torch::Tensor ids, coords;
    std::tuple<torch::Tensor, torch::Tensor> data; 

    mini_batch_size = target.size();  // mini batch size
    cell_size = 1.0 / (float)ng;  // cell_size = 1/G
    target_class = torch::full({(long int)mini_batch_size, ng, ng, this->class_num}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat));  // target_class{N,G,G,CN}
    target_coord = torch::full({(long int)mini_batch_size, ng, ng, 4}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat));  // target_coord{N,G,G,4}
    target_mask = torch::full({(long int)mini_batch_size, ng, ng}, /*value=*/false, torch::TensorOptions().dtype(torch::kBool));  // target_mask{N,G,G}

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
            if (target_mask[n][j][i].item<bool>() == false){
                target_class[n][j][i][id].data() = 1.0;  // class likelihood = 1.0
                target_coord[n][j][i][0].data() = cx;  // center of x[0.0,1.0)
                target_coord[n][j][i][1].data() = cy;  // center of y[0.0,1.0)
                target_coord[n][j][i][2].data() = w;  // width[0.0,1.0)
                target_coord[n][j][i][3].data() = h;  // height[0.0,1.0)
                target_mask[n][j][i].data() = true;  // mask = true
            }
        }
    }
    
    return {target_class.detach().clone(), target_coord.detach().clone(), target_mask.detach().clone()};  // target_class{N,G,G,CN}, target_coord{N,G,G,4}, target_mask{N,G,G}
    
}


// -------------------------------------
// class{Loss} -> function{rescale}
// -------------------------------------
torch::Tensor Loss::rescale(torch::Tensor &BBs){
    torch::Tensor x_min, y_min, x_max, y_max, output;
    x_min = BBs[0] - 0.5 * BBs[2];  // x_min{BB,...}
    y_min = BBs[1] - 0.5 * BBs[3];  // y_min{BB,...}
    x_max = BBs[0] + 0.5 * BBs[2];  // x_max{BB,...}
    y_max = BBs[1] + 0.5 * BBs[3];  // y_max{BB,...}
    output = torch::cat({x_min.unsqueeze(/*dim=*/0), y_min.unsqueeze(/*dim=*/0), x_max.unsqueeze(/*dim=*/0), y_max.unsqueeze(/*dim=*/0)}, /*dim=*/0);  // output{4,BB,...}
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
    area1 = area1.unsqueeze(/*dim=*/1).expand_as(inter);  // area1{BB1,object} ===> {BB1,BB2,object}
    area2 = area2.unsqueeze(/*dim=*/0).expand_as(inter);  // area2{BB2,object} ===> {BB1,BB2,object}

    // (7) Compute IoU from the areas
    unions = area1 + area2 - inter;  // area1{BB1,BB2,object}, area2{BB1,BB2,object}, inter{BB1,BB2,object} ===> unions{BB1,BB2,object}
    IoU = inter / unions;  // inter{BB1,BB2,object}, unions{BB1,BB2,object} ===> IoU{BB1,BB2,object}

    return IoU;

}


// --------------------------------------------------------------
// class{Loss} -> operator
// --------------------------------------------------------------
// <reference>
// GitHub: https://github.com/pjreddie/darknet
// Source: src/yolo_layer.c -> function{forward_yolo_layer}
// --------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Loss::operator()(std::vector<torch::Tensor> &inputs, std::vector<std::tuple<torch::Tensor, torch::Tensor>> &target, const std::tuple<float, float> image_sizes){

    static auto criterion = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kSum));
    torch::Device device = inputs.at(0).device();
    size_t scales = inputs.size();

    // (1) Set image size
    std::vector<float> image_size_vec(2);
    torch::Tensor image_size;
    /*************************************************************************/
    image_size_vec.at(0) = std::get<0>(image_sizes);
    image_size_vec.at(1) = std::get<1>(image_sizes);
    image_size = torch::from_blob(image_size_vec.data(), {1, 1, 1, 1, 2}, torch::kFloat).to(device).clone();  // image_size{1,1,1,1,2}

    // (2) Set object tensor
    torch::Tensor loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class;
    /*************************************************************************/
    loss_coord_xy = torch::full({}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_coord_wh = torch::full({}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_obj = torch::full({}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_noobj = torch::full({}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    loss_class = torch::full({}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

    for (size_t i = 0; i < scales; i++){

        torch::Tensor input = inputs.at(i);
        long int mini_batch_size = input.size(0);
        long int ng = input.size(1);

        // -----------------------------------
        // Preparation
        // -----------------------------------

        // (3.1) Set input tensor
        std::vector<torch::Tensor> input_split;
        torch::Tensor arange, x0, y0, x0y0;
        torch::Tensor input_view, input_class, input_conf;
        torch::Tensor input_xy, input_wh;
        torch::Tensor input_xy_unoffset, input_wh_unoffset, input_coord_unoffset, input_coord_per_unoffset;
        /*************************************************************************/
        arange = torch::arange(/*start=*/0.0, /*end=*/(float)ng, /*step=*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);  // arange{G} = [0,1,2,...,G-1]
        x0 = arange.view({1, 1, ng, 1, 1}).expand({1, ng, ng, 1, 1});  // arange{G} ===> {1,1,G,1,1} ===> x0{1,G,G,1,1}
        y0 = arange.view({1, ng, 1, 1, 1}).expand({1, ng, ng, 1, 1});  // arange{G} ===> {1,G,1,1,1} ===> y0{1,G,G,1,1}
        x0y0 = torch::cat({x0, y0}, /*dim=*/4);  // x0{1,G,G,1,1} + y0{1,G,G,1,1} ===> x0y0{1,G,G,1,2}
        /*************************************************************************/
        input_view = input.view({mini_batch_size, ng, ng, this->na, this->class_num + 5});  // input{N,G,G,A*(CN+5)} ===> input_view{N,G,G,A,CN+5}
        input_split = input_view.split_with_sizes(/*split_sizes=*/{this->class_num, 2, 2, 1}, /*dim=*/4);  // input_view{N,G,G,A,CN+5} ===> input_split({N,G,G,A,CN}, {N,G,G,A,2}, {N,G,G,A,2}, {N,G,G,A,1})
        input_class = torch::sigmoid(input_split.at(0));  // input_class{N,G,G,A,CN}
        input_conf = torch::sigmoid(input_split.at(3)).squeeze(-1);  // input_conf{N,G,G,A}
        /*************************************************************************/
        input_xy = torch::sigmoid(input_split.at(1));  // input_xy{N,G,G,A,2}
        input_wh = input_split.at(2);  // input_wh{N,G,G,A,2}
        /*************************************************************************/
        input_xy_unoffset = (input_xy + x0y0) / (float)ng;  // input_xy{N,G,G,A,2} ===> input_xy_unoffset{N,G,G,A,2}
        input_wh_unoffset = torch::exp(input_wh) * this->anchors_wh[i].to(device) / image_size;  // input_wh{N,G,G,A,2} ===> input_wh_unoffset{N,G,G,A,2}
        input_coord_unoffset = torch::cat({input_xy_unoffset, input_wh_unoffset}, /*dim=*/4);  // input_xy_unoffset{N,G,G,A,2} + input_wh_unoffset{N,G,G,A,2} ===> input_coord_unoffset{N,G,G,A,4}
        input_coord_per_unoffset = input_coord_unoffset.permute({4, 0, 1, 2, 3}).contiguous();  // input_coord_unoffset{N,G,G,A,4} ===> input_coord_per_unoffset{4,N,G,G,A}

        // (3.2) Set target tensor
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> target_new;
        std::vector<torch::Tensor> target_coord_split;
        torch::Tensor target_class;
        torch::Tensor target_coord_unoffset, target_coord_per_unoffset, obj_mask;
        torch::Tensor target_xy_unoffset, target_wh_unoffset, target_xy, target_wh;
        /*************************************************************************/
        target_new = this->build_target(target, ng);  // target{N, ({BB_n}, {BB_n,4}) } ===> target_new({N,G,G,CN}, {N,G,G,4}, {N,G,G})
        target_class = std::get<0>(target_new).to(device).unsqueeze(/*dim=*/3).expand_as(input_class);  // {N,G,G,CN} ===>{N,G,G,1,CN} ===> target_class{N,G,G,A,CN}
        target_coord_unoffset = std::get<1>(target_new).to(device);  // target_coord_unoffset{N,G,G,4}
        target_coord_per_unoffset = target_coord_unoffset.permute({3, 0, 1, 2});  // target_coord_unoffset{N,G,G,4} ===> target_coord_per_unoffset{4,N,G,G}
        obj_mask = std::get<2>(target_new).unsqueeze(/*dim=*/-1).to(device);  // obj_mask{N,G,G,1}
        /*************************************************************************/
        target_coord_split = target_coord_unoffset.split_with_sizes(/*split_sizes=*/{2, 2}, /*dim=*/3);  // target_coord_unoffset{N,G,G,4} ===> target_coord_split({N,G,G,2}, {N,G,G,2})
        target_xy_unoffset = target_coord_split.at(0).unsqueeze(/*dim=*/3).expand_as(input_xy);  // target_coord_split.at(0){N,G,G,2} ===> {N,G,G,1,2} ===> target_xy_unoffset{N,G,G,A,2}
        target_wh_unoffset = target_coord_split.at(1).unsqueeze(/*dim=*/3).expand_as(input_wh);  // target_coord_split.at(1){N,G,G,2} ===> {N,G,G,1,2} ===> target_wh_unoffset{N,G,G,A,2}
        target_xy = target_xy_unoffset * (float)ng - x0y0;  // target_xy_unoffset{N,G,G,A,2} ===> target_xy{N,G,G,A,2}
        target_wh = torch::log(target_wh_unoffset * image_size / this->anchors_wh[i].to(device));  // target_wh_unoffset{N,G,G,A,2} ===> target_wh{N,G,G,A,2}

        // (3.3) Extract response and no response mask
        bool response_flag, no_response_iou_flag;
        torch::Tensor input_coord_rescale, target_coord_rescale;
        torch::Tensor input_coord_grid, target_coord_grid, IoU_grid, grid_max_idx, grid_mask;
        torch::Tensor input_coord_image, target_coord_image, IoU_image, image_max_IoU, image_mask;
        torch::Tensor response_mask, no_response_mask, no_response_iou_mask;
        std::tuple<torch::Tensor, torch::Tensor> max_IoU_grid_with_idx, max_IoU_image_with_idx;
        /*************************************************************************/
        input_coord_rescale = this->rescale(input_coord_per_unoffset);  // input_coord_per_unoffset{4=len[cx,cy,w,h],N,G,G,A} ===> input_coord_rescale{4=len[x_min,y_min,x_max,y_max],N,G,G,A}
        target_coord_rescale = this->rescale(target_coord_per_unoffset);  // target_coord_per_unoffset{4=len[cx,cy,w,h],N,G,G} ===> target_coord_rescale{4=len[x_min,y_min,x_max,y_max],N,G,G}
        /*************************************************************************/
        input_coord_grid = input_coord_rescale.view({4, -1, this->na}).transpose(1, 2).contiguous();  // input_coord_rescale{4,N,G,G,A} ===> {4,N*G*G,A} ===> input_coord_grid{4,A,N*G*G}
        target_coord_grid = target_coord_rescale.view({4, 1, -1}).contiguous();  // target_coord_rescale{4,N,G,G} ===> target_coord_grid{4,1,N*G*G}
        IoU_grid = this->compute_iou(input_coord_grid, target_coord_grid).squeeze(/*dim=*/1).detach().clone();  // input_coord_grid{4,A,N*G*G}, target_coord_grid{4,1,N*G*G} ===> IoU_grid{A,N*G*G}
        max_IoU_grid_with_idx = IoU_grid.max(/*dim=*/0, /*keepdim=*/false);  // IoU_grid{A,N*G*G} ===> max_IoU_grid_with_idx(IoU{N*G*G}, idx{N*G*G})
        grid_max_idx = std::get<1>(max_IoU_grid_with_idx).view({mini_batch_size, ng, ng});  // max_IoU_grid_with_idx(IoU{N*G*G}, idx{N*G*G}) ===> grid_max_idx{N,G,G}
        grid_mask = torch::one_hot(grid_max_idx, this->na).to(torch::kBool).to(device);  // grid_max_idx{N,G,G} ===> grid_mask{N,G,G,A}
        /*************************************************************************/
        input_coord_image = input_coord_rescale.view({4, mini_batch_size, -1}).transpose(1, 2).contiguous();  // input_coord_rescale{4,N,G,G,A} ===> {4,N,G*G*A} ===> input_coord_image{4,G*G*A,N}
        target_coord_image = target_coord_rescale.view({4, mini_batch_size, -1}).transpose(1, 2).contiguous();  // target_coord_rescale{4,N,G,G} ===> {4,N,G*G} ===> target_coord_image{4,G*G,N}
        IoU_image = this->compute_iou(input_coord_image, target_coord_image).detach().clone();  // input_coord_image{4,G*G*A,N}, target_coord_image{4,G*G,N} ===> IoU_image{G*G*A,G*G,N}
        max_IoU_image_with_idx = IoU_image.max(/*dim=*/1, /*keepdim=*/false);  // IoU_image{G*G*A,G*G,N} ===> max_IoU_image_with_idx(IoU{G*G*A,N}, idx{G*G*A,N})
        image_max_IoU = std::get<0>(max_IoU_image_with_idx).view({ng, ng, this->na, mini_batch_size}).permute({3, 0, 1, 2}).contiguous();  // max_IoU_image_with_idx(IoU{G*G*A,N}, idx{G*G*A,N}) ===> {G,G,A,N} ===> image_max_IoU{N,G,G,A}
        image_mask = (image_max_IoU < this->ignore_thresh);  // image_max_IoU{N,G,G,A} ===> image_mask{N,G,G,A}
        /*************************************************************************/
        response_mask = obj_mask * grid_mask;  // obj_mask{N,G,G,1}, grid_mask{N,G,G,A} ===> response_mask{N,G,G,A}
        no_response_mask = (response_mask == false);  // response_mask{N,G,G,A} ===> no_response_mask{N,G,G,A}
        no_response_iou_mask = no_response_mask * image_mask;  // no_response_mask{N,G,G,A}, image_mask{N,G,G,A} ===> no_response_iou_mask{N,G,G,A}
        /*************************************************************************/
        response_flag = (response_mask.nonzero().numel() > 0);
        no_response_iou_flag = (no_response_iou_mask.nonzero().numel() > 0);

        // -----------------------------------
        // Calculation of Loss
        // -----------------------------------

        // (4.0) common process
        torch::Tensor response_coord_mask;
        /*************************************************************************/
        response_coord_mask = response_mask.unsqueeze(/*dim=*/-1).expand_as(input_xy);  // response_mask{N,G,G,A} ===> {N,G,G,A,1} ===> response_coord_mask{N,G,G,A,2}

        // (4.1) "center coordinate term"
        torch::Tensor input_response_xy, target_response_xy;
        /*************************************************************************/
        if (response_flag){
            input_response_xy = input_xy.masked_select(/*mask=*/response_coord_mask);  // input_xy{N,G,G,A,2} ===> input_response_xy{response*2}
            target_response_xy = target_xy.masked_select(/*mask=*/response_coord_mask);  // target_xy{N,G,G,A,2} ===> target_response_xy{response*2}
            loss_coord_xy = loss_coord_xy + criterion(input_response_xy, target_response_xy) * 0.5 / (float)mini_batch_size;
        }

        // (4.2) "range coordinate term"
        torch::Tensor input_response_wh, target_response_wh;
        /*************************************************************************/
        if (response_flag){
            input_response_wh = input_wh.masked_select(/*mask=*/response_coord_mask);  // input_wh{N,G,G,A,2} ===> input_response_wh{response*2}
            target_response_wh = target_wh.masked_select(/*mask=*/response_coord_mask);  // target_wh{N,G,G,A,2} ===> target_response_wh{response*2}
            loss_coord_wh = loss_coord_wh + criterion(input_response_wh, target_response_wh) * 0.5 / (float)mini_batch_size;
        }

        // (4.3) "object confidence term"
        torch::Tensor input_response_conf;
        torch::Tensor target_ones, target_response_ones;
        /*************************************************************************/
        if (response_flag){
            input_response_conf = input_conf.masked_select(/*mask=*/response_mask);  // input_conf{N,G,G,A} ===> input_response_conf{response}
            target_ones = torch::ones_like(input_conf);  // target_ones{N,G,G,A}
            target_response_ones = target_ones.masked_select(/*mask=*/response_mask);  // target_ones{N,G,G,A} ===> target_response_ones{response}
            loss_obj = loss_obj + criterion(input_response_conf, target_response_ones) * 0.5 / (float)mini_batch_size;
        }

        // (4.4) "no object confidence term"
        torch::Tensor input_no_response_conf;
        torch::Tensor target_zeros, target_no_response_zeros;
        /*************************************************************************/
        if (no_response_iou_flag){
            input_no_response_conf = input_conf.masked_select(/*mask=*/no_response_iou_mask);  // input_conf{N,G,G,A} ===> input_no_response_conf{no response iou}
            target_zeros = torch::zeros_like(input_conf);  // target_zeros{N,G,G,A}
            target_no_response_zeros = target_zeros.masked_select(/*mask=*/no_response_iou_mask);  // target_zeros{N,G,G,A} ===> target_no_response_zeros{no response iou}
            loss_noobj = loss_noobj + criterion(input_no_response_conf, target_no_response_zeros) * 0.5 / (float)mini_batch_size;
        }

        // (4.5) "class term"
        torch::Tensor response_class_mask;
        torch::Tensor input_response_class, target_response_class;
        /*************************************************************************/
        if (response_flag){
            response_class_mask = response_mask.unsqueeze(/*dim=*/-1).expand_as(input_class);  // response_mask{N,G,G,A} ===> response_class_mask{N,G,G,A,CN}
            input_response_class = input_class.masked_select(/*mask=*/response_class_mask);  // input_class{N,G,G,A,CN} ===> input_response_class{response*CN}
            target_response_class = target_class.masked_select(/*mask=*/response_class_mask);  // target_class{N,G,G,A,CN} ===> target_response_class{response*CN}
            loss_class = loss_class + criterion(input_response_class, target_response_class) * 0.5 / (float)mini_batch_size;
        }

    }

    return {loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class};

}
