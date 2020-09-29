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
#include "networks.hpp"                // UNet_Generator, PatchGAN_Discriminator
#include "transforms.hpp"              // transforms

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void train(po::variables_map &vm, torch::Device &device, UNet_Generator &gen, PatchGAN_Discriminator &dis, std::vector<transforms::Compose*> &transformI, std::vector<transforms::Compose*> &transformO);
void test(po::variables_map &vm, torch::Device &device, UNet_Generator &gen, std::vector<transforms::Compose*> &transformI, std::vector<transforms::Compose*> &transformO);
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
        ("input_nc", po::value<size_t>()->default_value(3), "input image channel : RGB=3, grayscale=1")
        ("output_nc", po::value<size_t>()->default_value(3), "output image channel : RGB=3, grayscale=1")
        ("nz", po::value<size_t>()->default_value(512), "dimensions of latent space")
        ("loss", po::value<std::string>()->default_value("vanilla"), "vanilla (cross-entropy), lsgan (mse), etc.")
        ("gpu_id", po::value<int>()->default_value(0), "cuda device : 'x=-1' is cpu device")
        ("seed_random", po::value<bool>()->default_value(false), "whether to make the seed of random number in a random")
        ("seed", po::value<int>()->default_value(0), "seed of random number")

        // (2) Define for Training
        ("train", po::value<bool>()->default_value(false), "training mode on/off")
        ("train_in_dir", po::value<std::string>()->default_value("trainI"), "training input image directory : ./datasets/<dataset>/<train_in_dir>/<image files>")
        ("train_out_dir", po::value<std::string>()->default_value("trainO"), "training output image directory : ./datasets/<dataset>/<train_out_dir>/<image files>")
        ("epochs", po::value<size_t>()->default_value(200), "training total epoch")
        ("batch_size", po::value<size_t>()->default_value(32), "training batch size")
        ("train_load_epoch", po::value<std::string>()->default_value(""), "epoch of model to resume learning")
        ("save_epoch", po::value<size_t>()->default_value(20), "frequency of epoch to save model and optimizer")

        // (3) Define for Validation
        ("valid", po::value<bool>()->default_value(false), "validation mode on/off")
        ("valid_in_dir", po::value<std::string>()->default_value("validI"), "validation input image directory : ./datasets/<dataset>/<valid_in_dir>/<image files>")
        ("valid_out_dir", po::value<std::string>()->default_value("validO"), "validation output image directory : ./datasets/<dataset>/<valid_out_dir>/<image files>")
        ("valid_batch_size", po::value<size_t>()->default_value(1), "validation batch size")
        ("valid_freq", po::value<size_t>()->default_value(1), "validation frequency to training epoch")

        // (4) Define for Test
        ("test", po::value<bool>()->default_value(false), "test mode on/off")
        ("test_in_dir", po::value<std::string>()->default_value("testI"), "test input image directory : ./datasets/<dataset>/<test_in_dir>/<image files>")
        ("test_out_dir", po::value<std::string>()->default_value("testO"), "test output image directory : ./datasets/<dataset>/<test_out_dir>/<image files>")
        ("test_load_epoch", po::value<std::string>()->default_value("latest"), "training epoch used for testing")
        ("test_result_dir", po::value<std::string>()->default_value("test_result"), "test result directory : ./<test_result_dir>")

        // (5) Define for Network Parameter
        ("lr_gen", po::value<float>()->default_value(2e-4), "learning rate for generator")
        ("lr_dis", po::value<float>()->default_value(2e-4), "learning rate for discriminator")
        ("beta1", po::value<float>()->default_value(0.5), "beta 1 in Adam of optimizer method")
        ("beta2", po::value<float>()->default_value(0.999), "beta 2 in Adam of optimizer method")
        ("ngf", po::value<size_t>()->default_value(64), "the number of filters in convolution layer closest to image in generator")
        ("ndf", po::value<size_t>()->default_value(64), "the number of filters in convolution layer closest to image in discriminator")
        ("Lambda", po::value<float>()->default_value(100.0), "the multiple of L1 norm")
        ("n_layers", po::value<size_t>()->default_value(3), "the number of layers in PatchGAN")
        ("no_dropout", po::value<bool>()->default_value(false), "Dropout off/on")

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
    po::store(parse_command_line(argc, argv, args), vm);
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
    }
    else{
        std::srand(vm["seed"].as<int>());
        torch::manual_seed(std::rand());
    }

    // (4) Set Transforms
    std::vector<transforms::Compose*> transformI{
        (transforms::Compose*)new transforms::Resize(cv::Size(vm["size"].as<size_t>(), vm["size"].as<size_t>()), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        (transforms::Compose*)new transforms::ToTensor(),                                                                            // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        (transforms::Compose*)new transforms::Normalize(0.5, 0.5)                                                                    // [0,1] ===> [-1,1]
    };
    if (vm["input_nc"].as<size_t>() == 1){
        transformI.insert(transformI.begin(), (transforms::Compose*)new transforms::Grayscale(1));
    }
    std::vector<transforms::Compose*> transformO{
        (transforms::Compose*)new transforms::Resize(cv::Size(vm["size"].as<size_t>(), vm["size"].as<size_t>()), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        (transforms::Compose*)new transforms::ToTensor(),                                                                            // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        (transforms::Compose*)new transforms::Normalize(0.5, 0.5)                                                                    // [0,1] ===> [-1,1]
    };
    if (vm["output_nc"].as<size_t>() == 1){
        transformO.insert(transformO.begin(), (transforms::Compose*)new transforms::Grayscale(1));
    }
    
    // (5) Define Network
    UNet_Generator gen(vm); gen->to(device);
    PatchGAN_Discriminator dis(vm); dis->to(device);
    
    // (6) Make Directories
    std::string dir = "checkpoints/" + vm["dataset"].as<std::string>();
    fs::create_directories(dir);

    // (7) Save Model Parameters
    Set_Model_Params(vm, gen, "UNet_Generator");
    Set_Model_Params(vm, dis, "PatchGAN_Discriminator");

    // (8.1) Training Phase
    if (vm["train"].as<bool>()){
        Set_Options(vm, argc, argv, args, "train");
        train(vm, device, gen, dis, transformI, transformO);
    }

    // (8.2) Test Phase
    if (vm["test"].as<bool>()){
        Set_Options(vm, argc, argv, args, "test");
        test(vm, device, gen, transformI, transformO);
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
