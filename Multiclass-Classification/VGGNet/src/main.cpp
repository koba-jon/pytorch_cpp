#include <iostream>                    // std::cout, std::cerr
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand, std::exit
// For External Library
#include <torch/torch.h>               // torch
#include <opencv2/opencv.hpp>          // cv::Mat
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "networks.hpp"                // MC_VGGNet
#include "transforms.hpp"              // transforms

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void train(po::variables_map &vm, torch::Device &device, MC_VGGNet &model, std::vector<transforms::Compose*> &transform, const std::vector<std::string> class_names);
void test(po::variables_map &vm, torch::Device &device, MC_VGGNet &model, std::vector<transforms::Compose*> &transform, const std::vector<std::string> class_names);
torch::Device Set_Device(po::variables_map &vm);
template <typename T> void Set_Model_Params(po::variables_map &vm, T &model, const std::string name);
std::vector<std::string> Set_Class_Names(const std::string path, const size_t class_num);
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
        ("class_list", po::value<std::string>()->default_value("list/ImageNet.txt"), "file name in which class names are listed")
        ("class_num", po::value<size_t>()->default_value(1000), "total classes")
        ("size", po::value<size_t>()->default_value(224), "image width and height")
        ("nc", po::value<size_t>()->default_value(3), "input image channel : RGB=3, grayscale=1")
        ("gpu_id", po::value<int>()->default_value(0), "cuda device : 'x=-1' is cpu device")
        ("seed_random", po::value<bool>()->default_value(false), "whether to make the seed of random number in a random")
        ("seed", po::value<int>()->default_value(0), "seed of random number")

        // (2) Define for Training
        ("train", po::value<bool>()->default_value(false), "training mode on/off")
        ("train_dir", po::value<std::string>()->default_value("train"), "training image directory : ./datasets/<dataset>/<train_dir>/<class name>/<image files>")
        ("epochs", po::value<size_t>()->default_value(200), "training total epoch")
        ("batch_size", po::value<size_t>()->default_value(32), "training batch size")
        ("train_load_epoch", po::value<std::string>()->default_value(""), "epoch of model to resume learning")
        ("save_epoch", po::value<size_t>()->default_value(20), "frequency of epoch to save model and optimizer")

        // (3) Define for Validation
        ("valid", po::value<bool>()->default_value(false), "validation mode on/off")
        ("valid_dir", po::value<std::string>()->default_value("valid"), "validation image directory : ./datasets/<dataset>/<valid_dir>/<class name>/<image files>")
        ("valid_batch_size", po::value<size_t>()->default_value(1), "validation batch size")
        ("valid_freq", po::value<size_t>()->default_value(1), "validation frequency to training epoch")

        // (4) Define for Test
        ("test", po::value<bool>()->default_value(false), "test mode on/off")
        ("test_dir", po::value<std::string>()->default_value("test"), "test image directory : ./datasets/<dataset>/<test_dir>/<class name>/<image files>")
        ("test_load_epoch", po::value<std::string>()->default_value("latest"), "training epoch used for testing")
        ("test_result_dir", po::value<std::string>()->default_value("test_result"), "test result directory : ./<test_result_dir>")

        // (5) Define for Network Parameter
        ("lr", po::value<float>()->default_value(1e-4), "learning rate")
        ("beta1", po::value<float>()->default_value(0.5), "beta 1 in Adam of optimizer method")
        ("beta2", po::value<float>()->default_value(0.999), "beta 2 in Adam of optimizer method")
        ("n_layers", po::value<size_t>(), "the number of layer in model")
        ("BN", po::value<bool>(), "whether to use batch normalization")

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
    std::vector<transforms::Compose*> transform{
        (transforms::Compose*)new transforms::Resize(cv::Size(vm["size"].as<size_t>(), vm["size"].as<size_t>()), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        (transforms::Compose*)new transforms::ToTensor(),                                                                            // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        (transforms::Compose*)new transforms::Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})                                // Pixel Value Normalization for ImageNet
    };
    if (vm["nc"].as<size_t>() == 1){
        transform.insert(transform.begin(), (transforms::Compose*)new transforms::Grayscale(1));
    }
    
    // (5) Define Network
    MC_VGGNet model(vm);
    model->to(device);
    
    // (6) Make Directories
    std::string dir = "checkpoints/" + vm["dataset"].as<std::string>();
    fs::create_directories(dir);

    // (7) Save Model Parameters
    Set_Model_Params(vm, model, "VGGNet");

    // (8) Set Class Names
    std::vector<std::string> class_names = Set_Class_Names(vm["class_list"].as<std::string>(), vm["class_num"].as<size_t>());

    // (9.1) Training Phase
    if (vm["train"].as<bool>()){
        Set_Options(vm, argc, argv, args, "train");
        train(vm, device, model, transform, class_names);
    }

    // (9.2) Test Phase
    if (vm["test"].as<bool>()){
        Set_Options(vm, argc, argv, args, "test");
        test(vm, device, model, transform, class_names);
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
// 4. Class Names Setting Function
// -----------------------------------
std::vector<std::string> Set_Class_Names(const std::string path, const size_t class_num){

    // (1) Memory Allocation
    std::vector<std::string> class_names = std::vector<std::string>(class_num);
    
    // (2) Get Class Names
    std::string class_name;
    std::ifstream ifs(path, std::ios::in);
    for (size_t i = 0; i < class_num; i++){
        if (!getline(ifs, class_name)){
            std::cerr << "Error : The number of classes does not match the number of lines in the class name file." << std::endl;
            std::exit(1);
        }
        class_names.at(i) = class_name;
    }
    ifs.close();

    // End Processing
    return class_names;

}


// -----------------------------------
// 5. Options Setting Function
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
