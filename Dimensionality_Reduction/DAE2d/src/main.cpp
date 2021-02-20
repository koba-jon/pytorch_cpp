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
#include "networks.hpp"                // ConvolutionalAutoEncoder
#include "transforms.hpp"              // transforms

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void train(po::variables_map &vm, torch::Device &device, ConvolutionalAutoEncoder &model, std::vector<transforms::Compose*> &transformI, std::vector<transforms::Compose*> &transformO);
void test(po::variables_map &vm, torch::Device &device, ConvolutionalAutoEncoder &model, std::vector<transforms::Compose*> &transformI, std::vector<transforms::Compose*> &transformO);
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
        ("size", po::value<size_t>()->default_value(256), "image width and height (x>=64)")
        ("nc", po::value<size_t>()->default_value(3), "input image channel : RGB=3, grayscale=1")
        ("nz", po::value<size_t>()->default_value(512), "dimensions of latent space")
        ("loss", po::value<std::string>()->default_value("l2"), "l1 (mean absolute error), l2 (mean squared error), ssim (structural similarity), etc.")
        ("gpu_id", po::value<int>()->default_value(0), "cuda device : 'x=-1' is cpu device")
        ("seed_random", po::value<bool>()->default_value(false), "whether to make the seed of random number in a random")
        ("seed", po::value<int>()->default_value(0), "seed of random number")

        // (2) Define for Denoising
        // (2.1) for Random Valued Impulse Noise (RVIN)
        ("RVIN", po::value<bool>()->default_value(false), "Random Valued Impulse Noise (RVIN) addition on/off")
        ("RVIN_prob", po::value<float>()->default_value(0.01), "RVIN probability of occurrence in each pixel")
        // (2.2) for Salt and Pepper Noise (SPN)
        ("SPN", po::value<bool>()->default_value(false), "Salt and Pepper Noise (SPN) addition on/off")
        ("SPN_prob", po::value<float>()->default_value(0.01), "SPN probability of occurrence in each pixel")
        ("SPN_salt_rate", po::value<float>()->default_value(0.5), "salt noise probability of occurrence in total noise")
        // (2.3) for Gaussian Noise (GN)
        ("GN", po::value<bool>()->default_value(false), "Gaussian Noise (GN) addition on/off")
        ("GN_prob", po::value<float>()->default_value(1.0), "GN probability of occurrence in each pixel")
        ("GN_mean", po::value<float>()->default_value(0.0), "the mean of GN")
        ("GN_std", po::value<float>()->default_value(0.01), "the standard deviation of GN")

        // (3) Define for Training
        ("train", po::value<bool>()->default_value(false), "training mode on/off")
        ("train_dir", po::value<std::string>()->default_value("train"), "training image directory : ./datasets/<dataset>/<train_dir>/<image files>")
        ("epochs", po::value<size_t>()->default_value(200), "training total epoch")
        ("batch_size", po::value<size_t>()->default_value(32), "training batch size")
        ("train_load_epoch", po::value<std::string>()->default_value(""), "epoch of model to resume learning")
        ("save_epoch", po::value<size_t>()->default_value(20), "frequency of epoch to save model and optimizer")

        // (4) Define for Validation
        ("valid", po::value<bool>()->default_value(false), "validation mode on/off")
        ("valid_dir", po::value<std::string>()->default_value("valid"), "validation image directory : ./datasets/<dataset>/<valid_dir>/<image files>")
        ("valid_batch_size", po::value<size_t>()->default_value(1), "validation batch size")
        ("valid_freq", po::value<size_t>()->default_value(1), "validation frequency to training epoch")

        // (5) Define for Test
        ("test", po::value<bool>()->default_value(false), "test mode on/off")
        ("test_in_dir", po::value<std::string>()->default_value("testI"), "test input image directory : ./datasets/<dataset>/<test_in_dir>/<image files>")
        ("test_out_dir", po::value<std::string>()->default_value("testO"), "test output image directory : ./datasets/<dataset>/<test_out_dir>/<image files>")
        ("test_load_epoch", po::value<std::string>()->default_value("latest"), "training epoch used for testing")
        ("test_result_dir", po::value<std::string>()->default_value("test_result"), "test result directory : ./<test_result_dir>")

        // (6) Define for Network Parameter
        ("lr", po::value<float>()->default_value(1e-4), "learning rate")
        ("beta1", po::value<float>()->default_value(0.5), "beta 1 in Adam of optimizer method")
        ("beta2", po::value<float>()->default_value(0.999), "beta 2 in Adam of optimizer method")
        ("nf", po::value<size_t>()->default_value(64), "the number of filters in convolution layer closest to image")

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
    // (4.1) for Original Dataset
    std::vector<transforms::Compose*> transformO{
        (transforms::Compose*)new transforms::Resize(cv::Size(vm["size"].as<size_t>(), vm["size"].as<size_t>()), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        (transforms::Compose*)new transforms::ToTensor(),                                                                            // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        (transforms::Compose*)new transforms::Normalize(0.5, 0.5)                                                                    // [0,1] ===> [-1,1]
    };
    if (vm["nc"].as<size_t>() == 1){
        transformO.insert(transformO.begin(), (transforms::Compose*)new transforms::Grayscale(1));
    }
    // (4.2) for Noised Dataset
    std::vector<transforms::Compose*> transformI{
        (transforms::Compose*)new transforms::Resize(cv::Size(vm["size"].as<size_t>(), vm["size"].as<size_t>()), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        (transforms::Compose*)new transforms::ToTensor()                                                                             // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
    };
    if (vm["nc"].as<size_t>() == 1){
        transformI.insert(transformI.begin(), (transforms::Compose*)new transforms::Grayscale(1));
    }
    if (vm["RVIN"].as<bool>()){
        transformI.push_back((transforms::Compose*)new transforms::AddRVINoise(vm["RVIN_prob"].as<float>()));
    }
    if (vm["SPN"].as<bool>()){
        transformI.push_back((transforms::Compose*)new transforms::AddSPNoise(vm["SPN_prob"].as<float>(), vm["SPN_salt_rate"].as<float>()));
    }
    if (vm["GN"].as<bool>()){
        transformI.push_back((transforms::Compose*)new transforms::AddGaussNoise(vm["GN_prob"].as<float>(), vm["GN_mean"].as<float>(), vm["GN_std"].as<float>()));
    }
    transformI.push_back((transforms::Compose*)new transforms::Normalize(0.5, 0.5));                                                  // [0,1] ===> [-1,1]
    
    // (5) Define Network
    ConvolutionalAutoEncoder CAE(vm);
    CAE->to(device);
    
    // (6) Make Directories
    std::string dir = "checkpoints/" + vm["dataset"].as<std::string>();
    fs::create_directories(dir);

    // (7) Save Model Parameters
    Set_Model_Params(vm, CAE, "CAE");

    // (8.1) Training Phase
    if (vm["train"].as<bool>()){
        Set_Options(vm, argc, argv, args, "train");
        train(vm, device, CAE, transformI, transformO);
    }

    // (8.2) Test Phase
    if (vm["test"].as<bool>()){
        Set_Options(vm, argc, argv, args, "test");
        test(vm, device, CAE, transformI, transformO);
    }

    // Post Processing
    for (size_t i = 0; i < transformI.size(); i++){
        delete transformI.at(i);
    }
    for (size_t i = 0; i < transformO.size(); i++){
        delete transformO.at(i);
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
