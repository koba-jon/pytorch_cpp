#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand
// For External Library
#include <torch/torch.h>               // torch
#include <opencv2/opencv.hpp>          // cv::Mat
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // VQVAE2, PixelSnail
#include "transforms.hpp"              // transforms

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void train1(po::variables_map &vm, torch::Device &device, VQVAE2 &model, std::vector<transforms_Compose> &transform);
void train2(po::variables_map &vm, torch::Device &device, VQVAE2 &vqvae2, PixelSnail &model, std::vector<transforms_Compose> &transform);
void train3(po::variables_map &vm, torch::Device &device, VQVAE2 &vqvae2, PixelSnail &model, std::vector<transforms_Compose> &transform);
void test1(po::variables_map &vm, torch::Device &device, VQVAE2 &model, std::vector<transforms_Compose> &transform);
void test2(po::variables_map &vm, torch::Device &device, VQVAE2 &vqvae2, PixelSnail &model, std::vector<transforms_Compose> &transform);
void test3(po::variables_map &vm, torch::Device &device, VQVAE2 &vqvae2, PixelSnail &model, std::vector<transforms_Compose> &transform);
void synth(po::variables_map &vm, torch::Device &device, VQVAE2 &model, PixelSnail &pixelsnail_t, PixelSnail &pixelsnail_b);
void sample(po::variables_map &vm, torch::Device &device, VQVAE2 &model, PixelSnail &pixelsnail_t, PixelSnail &pixelsnail_b);
torch::Device Set_Device(po::variables_map &vm);
template <typename T> void Set_Model_Params(po::variables_map &vm, T &model, const std::string name);
void Set_Options(po::variables_map &vm, int argc, const char *argv[], po::options_description &args, const std::string mode);


// -----------------------------------
// 0. Argument Function
// -----------------------------------
po::options_description parse_arguments(){

    po::options_description args("Options", 200, 30);
    
    args.add_options()

        // (1) Define for General Parameter
        ("help", "produce help message")
        ("dataset", po::value<std::string>(), "dataset name")
        ("size", po::value<size_t>()->default_value(256), "image width and height (x>=8)")
        ("nc", po::value<size_t>()->default_value(3), "input image channel : RGB=3, grayscale=1")
        ("nz", po::value<size_t>()->default_value(64), "dimensions of latent variables/embedding feature")
        ("K", po::value<size_t>()->default_value(512), "the number of embedding feature")
        ("loss", po::value<std::string>()->default_value("l2"), "l1 (mean absolute error), l2 (mean squared error), ssim (structural similarity), etc.")
        ("gpu_id", po::value<int>()->default_value(0), "cuda device : 'x=-1' is cpu device")
        ("seed_random", po::value<bool>()->default_value(false), "whether to make the seed of random number in a random")
        ("seed", po::value<int>()->default_value(0), "seed of random number")

        // (2.1) Define for Training 1
        ("train1", po::value<bool>()->default_value(false), "training 1 mode on/off")
        ("train1_dir", po::value<std::string>()->default_value("train1"), "training 1 image directory : ./datasets/<dataset>/<train1_dir>/<image files>")
        ("train1_epochs", po::value<size_t>()->default_value(200), "training 1 total epoch")
        ("train1_batch_size", po::value<size_t>()->default_value(32), "training 1 batch size")
        ("train1_load_epoch", po::value<std::string>()->default_value(""), "epoch of model to resume learning for training 1")
        ("train1_save_epoch", po::value<size_t>()->default_value(20), "frequency of epoch to save model and optimizer for training 1")

        // (2.2) Define for Validation of Training 1
        ("valid1", po::value<bool>()->default_value(false), "validation 1 mode on/off")
        ("valid1_dir", po::value<std::string>()->default_value("valid1"), "validation 1 image directory : ./datasets/<dataset>/<valid1_dir>/<image files>")
        ("valid1_batch_size", po::value<size_t>()->default_value(1), "validation 1 batch size")
        ("valid1_freq", po::value<size_t>()->default_value(1), "validation frequency to training epoch for validation 1")

        // (2.3) Define for Test for Training 1
        ("test1", po::value<bool>()->default_value(false), "test 1 mode on/off")
        ("test1_in_dir", po::value<std::string>()->default_value("test1I"), "test 1 input image directory : ./datasets/<dataset>/<test1_in_dir>/<image files>")
        ("test1_out_dir", po::value<std::string>()->default_value("test1O"), "test 1 output image directory : ./datasets/<dataset>/<test1_out_dir>/<image files>")
        ("test1_load_epoch", po::value<std::string>()->default_value("latest"), "training epoch used for testing 1")
        ("test1_result_dir", po::value<std::string>()->default_value("test1_result"), "test 1 result directory : ./<test1_result_dir>")

        // (3.1) Define for Training 2
        ("train2", po::value<bool>()->default_value(false), "training 2 mode on/off")
        ("train2_dir", po::value<std::string>()->default_value("train2"), "training 2 image directory : ./datasets/<dataset>/<train2_dir>/<image files>")
        ("train2_epochs", po::value<size_t>()->default_value(200), "training 2 total epoch")
        ("train2_batch_size", po::value<size_t>()->default_value(32), "training 2 batch size")
        ("train2_load_epoch", po::value<std::string>()->default_value(""), "epoch of model to resume learning for training 2")
        ("train2_vqvae2_load_epoch", po::value<std::string>()->default_value("latest"), "training 2 epoch for training 2")
        ("train2_save_epoch", po::value<size_t>()->default_value(20), "frequency of epoch to save model and optimizer for training 2")

        // (3.2) Define for Validation of Training 2
        ("valid2", po::value<bool>()->default_value(false), "validation 2 mode on/off")
        ("valid2_dir", po::value<std::string>()->default_value("valid2"), "validation 2 image directory : ./datasets/<dataset>/<valid2_dir>/<image files>")
        ("valid2_batch_size", po::value<size_t>()->default_value(1), "validation 2 batch size")
        ("valid2_freq", po::value<size_t>()->default_value(1), "validation frequency to training epoch for validation 2")

        // (3.3) Define for Test for Training 2
        ("test2", po::value<bool>()->default_value(false), "test 2 mode on/off")
        ("test2_dir", po::value<std::string>()->default_value("test2"), "test 2 image directory : ./datasets/<dataset>/<test2_in_dir>/<image files>")
        ("test2_vqvae2_load_epoch", po::value<std::string>()->default_value("latest"), "training 1 epoch used for testing 2")
        ("test2_pixelsnail_load_epoch", po::value<std::string>()->default_value("latest"), "training 2 epoch used for testing 2")
        ("test2_result_dir", po::value<std::string>()->default_value("test2_result"), "test 2 result directory : ./<test2_result_dir>")

        // (4.1) Define for Training 3
        ("train3", po::value<bool>()->default_value(false), "training 3 mode on/off")
        ("train3_dir", po::value<std::string>()->default_value("train3"), "training 3 image directory : ./datasets/<dataset>/<train3_dir>/<image files>")
        ("train3_epochs", po::value<size_t>()->default_value(200), "training 3 total epoch")
        ("train3_batch_size", po::value<size_t>()->default_value(32), "training 3 batch size")
        ("train3_load_epoch", po::value<std::string>()->default_value(""), "epoch of model to resume learning for training 3")
        ("train3_vqvae2_load_epoch", po::value<std::string>()->default_value("latest"), "training 3 epoch for training 3")
        ("train3_save_epoch", po::value<size_t>()->default_value(20), "frequency of epoch to save model and optimizer for training 3")

        // (4.2) Define for Validation of Training 3
        ("valid3", po::value<bool>()->default_value(false), "validation 3 mode on/off")
        ("valid3_dir", po::value<std::string>()->default_value("valid3"), "validation 3 image directory : ./datasets/<dataset>/<valid3_dir>/<image files>")
        ("valid3_batch_size", po::value<size_t>()->default_value(1), "validation 3 batch size")
        ("valid3_freq", po::value<size_t>()->default_value(1), "validation frequency to training epoch for validation 3")

        // (4.3) Define for Test for Training 3
        ("test3", po::value<bool>()->default_value(false), "test 3 mode on/off")
        ("test3_dir", po::value<std::string>()->default_value("test3"), "test 3 image directory : ./datasets/<dataset>/<test3_in_dir>/<image files>")
        ("test3_vqvae2_load_epoch", po::value<std::string>()->default_value("latest"), "training 3 epoch used for testing 3")
        ("test3_pixelsnail_load_epoch", po::value<std::string>()->default_value("latest"), "training 3 epoch used for testing 3")
        ("test3_result_dir", po::value<std::string>()->default_value("test3_result"), "test 3 result directory : ./<test3_result_dir>")

        // (5) Define for Synthesis
        ("synth", po::value<bool>()->default_value(false), "synthesis mode on/off")
        ("synth_vqvae2_load_epoch", po::value<std::string>()->default_value("latest"), "training 1 epoch used for synthesis")
        ("synth_pixelsnail_load_epoch", po::value<std::string>()->default_value("latest"), "training 2 epoch used for synthesis")
        ("synth_result_dir", po::value<std::string>()->default_value("synth_result"), "synthesis result directory : ./<synth_result_dir>")
        ("synth_count", po::value<size_t>()->default_value(13), "the number of output images in synthesis")

        // (6) Define for Sampling
        ("sample", po::value<bool>()->default_value(false), "sampling mode on/off")
        ("sample_vqvae2_load_epoch", po::value<std::string>()->default_value("latest"), "training 1 epoch used for sampling")
        ("sample_pixelsnail_load_epoch", po::value<std::string>()->default_value("latest"), "training 2 epoch used for sampling")
        ("sample_result_dir", po::value<std::string>()->default_value("sample_result"), "sampling result directory : ./<sample_result_dir>")
        ("sample_total", po::value<size_t>()->default_value(100), "total number of data obtained by random sampling")

        // (7) Define for Network Parameter
        ("lr", po::value<float>()->default_value(1e-4), "learning rate")
        ("beta1", po::value<float>()->default_value(0.5), "beta 1 in Adam of optimizer method")
        ("beta2", po::value<float>()->default_value(0.999), "beta 2 in Adam of optimizer method")
        ("nf", po::value<size_t>()->default_value(128), "the number of filters in convolution layer closest to image")
        ("res_block", po::value<size_t>()->default_value(2), "the number of blocks in residual layer")
        ("res_nc", po::value<size_t>()->default_value(32), "the number of channel in residual layer")
        ("dim_pix", po::value<size_t>()->default_value(256), "the number of dimensions of PixelSnail")
        ("res_block_pix", po::value<size_t>()->default_value(4), "the number of blocks in residual layer of PixelSnail")
        ("res_nc_pix", po::value<size_t>()->default_value(256), "the number of channel in residual layer of PixelSnail")
        ("droprate", po::value<float>()->default_value(0.1), "the rate of dropout")
        ("out_res_block_pix", po::value<size_t>()->default_value(0), "the number of out residual block")
        ("cond_res_block_pix", po::value<size_t>()->default_value(3), "the number of conditional residual block")
        ("Lambda", po::value<float>()->default_value(0.25), "the multiple of Latent Loss")
        ("dim_pix", po::value<size_t>()->default_value(64), "dimensions of hidden layers of PixelSnail")
        ("n_layers", po::value<size_t>()->default_value(15), "the number of layers of PixelSnail")

    ;
    
    // End Processing
    return args;
}


// -----------------------------------
// 1. Main Function
// -----------------------------------
int main(int argc, const char *argv[]){

    // (1) Extract Arguments
    po::options_description args = parse_arguments();
    po::variables_map vm{};
    po::store(po::parse_command_line(argc, argv, args), vm);
    po::notify(vm);
    if (vm.empty() || vm.count("help")){
        std::cout << args << std::endl;
        return 1;
    }
    
    // (2) Select Device
    torch::Device device = Set_Device(vm);
    std::cout << "using device = " << device << std::endl;

    // (3) Set Seed
    if (vm["seed_random"].as<bool>()){
        std::random_device rd;
        std::srand(rd());
        torch::manual_seed(std::rand());
        torch::globalContext().setDeterministicCuDNN(false);
        torch::globalContext().setBenchmarkCuDNN(true);
    }
    else{
        std::srand(vm["seed"].as<int>());
        torch::manual_seed(std::rand());
        torch::globalContext().setDeterministicCuDNN(true);
        torch::globalContext().setBenchmarkCuDNN(false);
    }

    // (4) Set Transforms
    std::vector<transforms_Compose> transform{
        transforms_Resize(cv::Size(vm["size"].as<size_t>(), vm["size"].as<size_t>()), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                                            // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        transforms_Normalize(0.5, 0.5)                                                                    // [0,1] ===> [-1,1]
    };
    if (vm["nc"].as<size_t>() == 1){
        transform.insert(transform.begin(), transforms_Grayscale(1));
    }
    
    // (5) Define Network
    VQVAE2 vqvae2(vm); vqvae2->to(device);
    PixelSnail pixelsnail_t(vm["K"].as<size_t>(), vm["dim_pix"].as<size_t>(), 5, 4, vm["res_block_pix"].as<size_t>(), vm["res_nc_pix"].as<size_t>(), true, vm["droprate"].as<float>(), 0, 0, 3, vm["out_res_block_pix"].as<size_t>()); pixelsnail_t->to(device);
    PixelSnail pixelsnail_b(vm["K"].as<size_t>(), vm["dim_pix"].as<size_t>(), 5, 4, vm["res_block_pix"].as<size_t>(), vm["res_nc_pix"].as<size_t>(), false, vm["droprate"].as<float>(), vm["cond_res_block_pix"].as<size_t>(), vm["res_nc_pix"].as<size_t>()); pixelsnail_b->to(device);
    
    // (6) Make Directories
    std::string dir = "checkpoints/" + vm["dataset"].as<std::string>();
    fs::create_directories(dir);

    // (7) Save Model Parameters
    Set_Model_Params(vm, vqvae2, "VQ-VAE");
    Set_Model_Params(vm, pixelsnail_t, "PixelSnail Top");
    Set_Model_Params(vm, pixelsnail_b, "PixelSnail Bottom");

    // (8.1.1) Training Phase of VQVAE2
    if (vm["train1"].as<bool>()){
        Set_Options(vm, argc, argv, args, "train1");
        train1(vm, device, vqvae2, transform);
    }

    // (8.1.2) Test Phase of VQVAE2
    if (vm["test1"].as<bool>()){
        Set_Options(vm, argc, argv, args, "test1");
        test1(vm, device, vqvae2, transform);
    }

    // (8.2.1) Training Phase of PixelSnail for top latent space
    if (vm["train2"].as<bool>()){
        Set_Options(vm, argc, argv, args, "train2");
        train2(vm, device, vqvae2, pixelsnail_t, transform);
    }

    // (8.2.2) Test Phase of PixelSnail for top latent space
    if (vm["test2"].as<bool>()){
        Set_Options(vm, argc, argv, args, "test2");
        test2(vm, device, vqvae2, pixelsnail_t, transform);
    }

    // (8.3.1) Training Phase of PixelSnail for bottom latent space
    if (vm["train3"].as<bool>()){
        Set_Options(vm, argc, argv, args, "train3");
        train3(vm, device, vqvae2, pixelsnail_b, transform);
    }

    // (8.3.2) Test Phase of PixelSnail for bottom latent space
    if (vm["test3"].as<bool>()){
        Set_Options(vm, argc, argv, args, "test3");
        test3(vm, device, vqvae2, pixelsnail_b, transform);
    }

    // (8.4) Synthesis Phase
    if (vm["synth"].as<bool>()){
        Set_Options(vm, argc, argv, args, "synth");
        synth(vm, device, vqvae2, pixelsnail_t, pixelsnail_b);
    }

    // (8.5) Sampling Phase
    if (vm["sample"].as<bool>()){
        Set_Options(vm, argc, argv, args, "sample");
        sample(vm, device, vqvae2, pixelsnail_t, pixelsnail_b);
    }

    // End Processing
    return 0;

}


// -----------------------------------
// 2. Device Setting Function
// -----------------------------------
torch::Device Set_Device(po::variables_map &vm){

    // (1) GPU Type
    int gpu_id = vm["gpu_id"].as<int>();
    if (torch::cuda::is_available() && gpu_id>=0){
        torch::Device device(torch::kCUDA, gpu_id);
        return device;
    }

    // (2) CPU Type
    torch::Device device(torch::kCPU);
    return device;

}


// -----------------------------------
// 3. Model Parameters Setting Function
// -----------------------------------
template <typename T>
void Set_Model_Params(po::variables_map &vm, T &model, const std::string name){

    // (1) Make Directory
    std::string dir = "checkpoints/" + vm["dataset"].as<std::string>() + "/model_params/";
    fs::create_directories(dir);

    // (2.1) File Open
    std::string fname = dir + name + ".txt";
    std::ofstream ofs(fname);

    // (2.2) Calculation of Parameters
    size_t num_params = 0;
    for (auto param : model->parameters()){
        num_params += param.numel();
    }
    ofs << "Total number of parameters : " << (float)num_params/1e6f << "M" << std::endl << std::endl;
    ofs << model << std::endl;

    // (2.3) File Close
    ofs.close();

    // End Processing
    return;

}


// -----------------------------------
// 4. Options Setting Function
// -----------------------------------
void Set_Options(po::variables_map &vm, int argc, const char *argv[], po::options_description &args, const std::string mode){

    // (1) Make Directory
    std::string dir = "checkpoints/" + vm["dataset"].as<std::string>() + "/options/";
    fs::create_directories(dir);

    // (2) Terminal Output
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << args << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    // (3.1) File Open
    std::string fname = dir + mode + ".txt";
    std::ofstream ofs(fname, std::ios::app);

    // (3.2) Arguments Output
    ofs << "--------------------------------------------" << std::endl;
    ofs << "Command Line Arguments:" << std::endl;
    for (int i = 1; i < argc; i++){
        if (i % 2 == 1){
            ofs << "  " << argv[i] << '\t' << std::flush;
        }
        else{
            ofs << argv[i] << std::endl;
        }
    }
    ofs << "--------------------------------------------" << std::endl;
    ofs << args << std::endl;
    ofs << "--------------------------------------------" << std::endl << std::endl;

    // (3.3) File Close
    ofs.close();

    // End Processing
    return;

}
