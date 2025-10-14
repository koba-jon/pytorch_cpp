#include <iostream>                    // std::cout, std::cerr
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <chrono>                      // std::chrono
#include <utility>                     // std::pair
#include <iomanip>                     // std::setprecision
#include <cstdlib>                     // std::exit
// For External Library
#include <torch/torch.h>               // torch
#include <opencv2/opencv.hpp>          // cv::Mat
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // YOLOv8
#include "detector.hpp"                // YOLODetector
#include "transforms.hpp"              // transforms_Compose, transforms::apply
#include "visualizer.hpp"              // visualizer

// Define Namespace
namespace po = boost::program_options;


// -------------------
// Demo Function
// -------------------
void demo(po::variables_map &vm, torch::Device &device, YOLOv8 &model, std::vector<transforms_Compose> &transformI, std::vector<transforms_Compose> &transformD, const std::vector<std::string> class_names){

    constexpr double alpha = 0.1;  // current importance of moving average for calculating FPS
    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // (0) Initialization and Declaration
    int key;
    bool flag;
    double seconds, seconds_est, fps;
    std::string path;
    std::stringstream ss1, ss2;
    std::chrono::system_clock::time_point start, end;
    cv::Mat BGR, RGB;
    cv::Mat imageD, imageI, imageO;
    torch::Tensor tensorD, tensorI;
    std::vector<torch::Tensor> tensorO, tensorO_one;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_result;
    cv::VideoCapture cap;

    // (1) Get Model
    path = "checkpoints/" + vm["dataset"].as<std::string>() + "/models/epoch_" + vm["demo_load_epoch"].as<std::string>() + ".pth";
    torch::load(model, path, device);

    // (2) Set Detector
    auto detector = YOLODetector((long int)vm["class_num"].as<size_t>(), vm["prob_thresh"].as<float>(), vm["nms_thresh"].as<float>());
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette = detector.get_label_palette();

    // (3) Set Camera Device
    if (vm["movie"].as<std::string>() == ""){
        cap.open(vm["cam_num"].as<size_t>());
        if (!cap.isOpened()){
            std::cerr << "Error : Couldn't open the camera '" << vm["cam_num"].as<size_t>() << "'." << std::endl;
            std::exit(1);
        }
        else{
            cap.set(cv::CAP_PROP_FRAME_WIDTH, vm["window_w"].as<size_t>());
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, vm["window_h"].as<size_t>());
        }
    }
    else{
        cap.open(vm["movie"].as<std::string>());
        if (!cap.isOpened()){
            std::cerr << "Error : Couldn't open the movie '" << vm["movie"].as<std::string>() << "'." << std::endl;
            std::exit(1);
        }
    }

    // (4) Show Key Information
    std::cout << std::endl;
    std::cout << "<Key Information>" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "| key |        action        |" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "|  q  | Stop the camera.     |" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << std::endl;

    // (5) Demo
    torch::NoGradGuard no_grad;
    model->eval();
    flag = true;
    fps = 0.0;
    start = std::chrono::system_clock::now();
    while (cap.read(BGR)){

        cv::cvtColor(BGR, RGB, cv::COLOR_BGR2RGB);  // {0,1,2} = {B,G,R} ===> {0,1,2} = {R,G,B}

        // (5.1) Set image for input
        RGB.copyTo(imageI);
        tensorI = transforms::apply(transformI, imageI).unsqueeze(/*dim=*/0).to(device);  // imageI{H_D,W_D,C} ==={Resize,ToTensor,etc.}===> tensorI{1,C,H,W}

        // (5.2) Set image for detection
        RGB.copyTo(imageD);
        tensorD = transforms::apply(transformD, imageD);  // imageD{H_D,W_D,3} ==={ToTensor,etc.}===> tensorD{3,H_D,W_D}

        // (5.3) Inference and Detection
        tensorO = model->forward(tensorI);  // {1,C,H,W} ===> {S,{1,G,G,FF}}
        /*************************************************************************/
        tensorO_one = std::vector<torch::Tensor>(tensorO.size());
        for (size_t i = 0; i < tensorO.size(); i++){
            tensorO_one.at(i) = tensorO.at(i)[0];
        }
        detect_result = detector(tensorO_one);  // tensorO_one{S,{G,G,FF}} ===> detect_result{ (ids{BB_n}, coords{BB_n,4}, probs{BB_n}) }
        /*************************************************************************/
        imageO = visualizer::draw_detections_des(tensorD.detach(), {std::get<0>(detect_result), std::get<1>(detect_result)}, std::get<2>(detect_result), class_names, label_palette, /*range=*/output_range);

        // (5.4) Calculate FPS
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 0.001;
        start = std::chrono::system_clock::now();
        switch (flag){
            case true:
                flag = false;
                seconds_est = seconds;
                fps = 1.0 / seconds;
                break;
            default:
                seconds_est = (1.0 - alpha) * seconds_est + alpha * seconds;
                fps = (1.0 - alpha) * fps + alpha * (1.0 / seconds);
        }
        ss1.str(""); ss1.clear(std::stringstream::goodbit);
        ss2.str(""); ss2.clear(std::stringstream::goodbit);
        ss1 << std::setprecision(3) << fps;
        ss2 << std::setprecision(3) << seconds;
        std::cout << "FPS: " << ss1.str() << "[frame/second] (" << ss2.str() << "[second/frame])" << std::endl;
        
        // (5.5) Show the image in which the objects were detected
        cv::imshow("demo", imageO);
        key = cv::waitKey(1);
        if (key == 'q') break;

    }

    // Post Processing
    cap.release();
    cv::destroyAllWindows();

    // End Processing
    return;

}