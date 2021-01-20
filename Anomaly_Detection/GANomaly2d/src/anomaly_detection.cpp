#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
// For External Library
#include <boost/program_options.hpp>   // boost::program_options

#define NORMAL 0
#define ANOMALY 1

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
bool judgement(double value, bool class_name, double threshold);


// ----------------------------
// Anomaly Detection Function
// ----------------------------
void anomaly_detection(po::variables_map &vm){

    // (0) Initialization and Declaration
    size_t i, j;
    size_t TP, FP, TN, FN;
    size_t total_data, idx;
    double data_one;
    double min, max, step, thresh;
    double SED, SED_min, cutoff_idx, cutoff_th;
    double TP_rate, FP_rate, TN_rate, FN_rate, pre_TP_rate, pre_FP_rate;
    double precision, recall, specificity;
    double accuracy, F, AUC;
    std::ifstream ifs;
    std::ofstream ofs;
    std::string result_dir, result_path;
    std::vector<double> data[2];

    // (1) Set Directory and Path
    result_dir = vm["AD_result_dir"].as<std::string>();
    result_path = vm["AD_result_dir"].as<std::string>() + "/accuracy.csv";
    fs::create_directories(result_dir);

    // (2.1) Set Anomaly Data
    ifs.open(vm["anomaly_path"].as<std::string>());
    ifs >> data_one;
    min = data_one;
    max = data_one;
    data[ANOMALY].push_back(data_one);
    while (true){
        ifs >> data_one;
        if (ifs.eof()) break;
        if (data_one > max){
            max = data_one;
        }
        else if (data_one < min){
            min = data_one;
        }
        data[ANOMALY].push_back(data_one);
    }
    ifs.close();

    // (2.2) Set Normal Data
    ifs.open(vm["normal_path"].as<std::string>());
    while (true){
        ifs >> data_one;
        if (ifs.eof()) break;
        if (data_one > max){
            max = data_one;
        }
        else if (data_one < min){
            min = data_one;
        }
        data[NORMAL].push_back(data_one);
    }
    ifs.close();

    // (3) Pre-Processing
    ofs.open(result_path);
    ofs << "threshold,TP,FP,TN,FN,TP rate,FP rate,TN rate,FN rate,SED,precision,recall,specificity,accuracy,F" << std::endl;
    total_data = data[NORMAL].size() + data[ANOMALY].size();
    std::cout << "total anomaly detection data : " << total_data << std::endl;

    // (4) Calculation of Step
    step = (double)(max - min) / (double)(vm["n_thresh"].as<size_t>() - 2);
    if (step == 0.0){
        step = 1.0;
    }

    // (5) Anomaly Detection
    AUC = 0.0;
    pre_TP_rate = 1.0;
    pre_FP_rate = 1.0;
    SED_min = 1.0;
    idx = 1;
    cutoff_idx = 1;
    cutoff_th = min;
    thresh = min;
    for (i = 0; i < vm["n_thresh"].as<size_t>(); i++){

        thresh += step;
        idx++;

        // (5.1) Get TP, FN, TN and FP
        TP = 0, FP = 0, TN = 0, FN = 0;
        for (j = 0; j < data[ANOMALY].size(); j++){
            if (judgement(data[ANOMALY][j], ANOMALY, thresh)){
                TP++;
            }
            else{
                FN++;
            }
        }
        for (j = 0; j < data[NORMAL].size(); j++){
            if (judgement(data[NORMAL][j], NORMAL, thresh)){
                TN++;
            }
            else{
                FP++;
            }
        }

        // (5.2) Calculation of Accuracy
        TP_rate = (double)TP / (double)data[ANOMALY].size();
        FP_rate = (double)FP / (double)data[NORMAL].size();
        TN_rate = (double)TN / (double)data[NORMAL].size();
        FN_rate = (double)FN / (double)data[ANOMALY].size();
        precision = (double)TP / (double)(TP + FP);
        recall = (double)TP / (double)(TP + FN);
        specificity = (double)TN / (double)(FP + TN);
        accuracy = (double)(TP + TN) / (double)total_data;
        F = (double)TP / ((double)TP + 0.5 * (double)(FP + FN));
        AUC += (pre_TP_rate + TP_rate) * (pre_FP_rate - FP_rate) * 0.5;
        pre_TP_rate = TP_rate;
        pre_FP_rate = FP_rate;

        // (5.3) Calculation of Cut-Off Value
        SED = (1.0 - TP_rate) * (1.0 - TP_rate) + FP_rate * FP_rate;
        if (SED < SED_min){
            SED_min = SED;
            cutoff_idx = idx;
            cutoff_th = thresh;
        }

        // (5.4) File Output
        ofs << thresh << "," << std::flush;
        ofs << TP << "," << FP << "," << TN << "," << FN << "," << std::flush;
        ofs << TP_rate << "," << FP_rate << "," << TN_rate << "," << FN_rate << "," << std::flush;
        ofs << SED << "," << std::flush;
        ofs << precision << "," << recall << "," << specificity << "," << std::flush;
        ofs << accuracy << "," << F << std::endl;

    }

    // (6) File Output
    ofs << std::endl;
    ofs << "ROC-AUC," << AUC << std::endl;
    ofs << "index(Cut-Off)," << cutoff_idx << std::endl;
    ofs << "threshold(Cut-Off)," << cutoff_th << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}


// ----------------------------
// Anomaly Judgement Function
// ----------------------------
bool judgement(double value, bool class_name, double threshold){

    bool judge;

    if (value < threshold){
        judge = false;
    }
    else{
        judge = true;
    }

    if (class_name == NORMAL){
        return !judge;
    }

    return judge;

}
