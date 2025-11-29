#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <algorithm>                   // std::sort
#include <iomanip>                     // std::fixed, std::setprecision
// For External Library
#include <boost/program_options.hpp>   // boost::program_options

#define NORMAL 0
#define ANOMALY 1

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

struct Sample {
    int label;      // NORMAL or ANOMALY
    double score;   // anomaly score
};


// ----------------------------
// Anomaly Detection Function
// ----------------------------
void anomaly_detection(po::variables_map &vm){

    // (0) Initialization and Declaration
    size_t i;
    size_t TP, FP, TN, FN;
    size_t anomaly_data, normal_data, total_data;
    double score;
    double TP_rate, FP_rate, TN_rate, FN_rate, pre_TP_rate, pre_FP_rate;
    double precision, recall, specificity;
    double accuracy, F, AUC;
    std::ifstream ifs;
    std::ofstream ofs, ofs_auc;
    std::string result_dir, result_path;
    std::vector<Sample> data_all;

    // (1) Set Directory and Path
    result_dir = vm["AD_result_dir"].as<std::string>();
    fs::create_directories(result_dir);
    result_path = result_dir + "/accuracy.csv";

    // (2) Read anomaly scores
    anomaly_data = 0;
    ifs.open(vm["anomaly_path"].as<std::string>());
    while (ifs >> score){
        data_all.push_back(Sample{ANOMALY, score});
        anomaly_data++;
    }
    ifs.close();

    // (3) Read normal scores
    normal_data = 0;
    ifs.open(vm["normal_path"].as<std::string>());
    while (ifs >> score){
        data_all.push_back(Sample{NORMAL, score});
        normal_data++;
    }
    ifs.close();

    // (4) Pre-Processing
    total_data = data_all.size();
    std::cout << "total anomaly detection data: " << total_data << std::endl;

    ofs.open(result_path);
    ofs << "index,TP,FP,TN,FN,TP rate,FP rate,TN rate,FN rate,precision,recall,specificity,accuracy,F" << std::endl;

    // (5) Sort by score ascending
    std::sort(data_all.begin(), data_all.end(), [](const Sample &a, const Sample &b){ return a.score < b.score; });

    // (6) Anomaly Detection (threshold moves along sorted scores)
    AUC = 0.0;
    TP = anomaly_data;
    FP = normal_data;
    TN = 0;
    FN = 0;
    pre_TP_rate = 1.0;
    pre_FP_rate = 1.0;

    for (i = 0; i < data_all.size(); i++){

        // (6.1) Update TP, FN, TN, FP
        if (data_all[i].label == ANOMALY){
            TP -= 1;
            FN += 1;
        }
        else{
            FP -= 1;
            TN += 1;
        }

        // (6.2) Metrics
        TP_rate = (double)TP / (double)anomaly_data;
        FP_rate = (double)FP / (double)normal_data;
        TN_rate = (double)TN / (double)normal_data;
        FN_rate = (double)FN / (double)anomaly_data;

        precision = (TP + FP) ? (double)TP / (double)(TP + FP) : 0.0;
        recall = (double)TP / (double)(TP + FN);              // same as TP_rate-denom
        specificity = (double)TN / (double)(FP + TN);
        accuracy = (double)(TP + TN) / (double)total_data;
        F = (double)TP / ((double)TP + 0.5 * (double)(FP + FN));

        // trapezoid integration in ROC space
        AUC += (pre_TP_rate + TP_rate) * (pre_FP_rate - FP_rate) * 0.5;
        pre_TP_rate = TP_rate;
        pre_FP_rate = FP_rate;

        // (6.3) File Output
        ofs << (i + 1) << "," << std::flush;
        ofs << TP << "," << FP << "," << TN << "," << FN << "," << std::flush;
        ofs << TP_rate << "," << FP_rate << "," << TN_rate << "," << FN_rate << "," << std::flush;
        ofs << precision << "," << recall << "," << specificity << "," << std::flush;
        ofs << accuracy << "," << F << std::endl;
    }

    // (7) AUROC output
    ofs << std::endl;
    ofs << "AUROC," << AUC << std::endl;
    ofs.close();

    ofs_auc.open(result_dir + "/AUROC.txt");
    ofs_auc << std::fixed << std::setprecision(1) << AUC * 100.0 << "  " << std::setprecision(16) << AUC;
    ofs_auc.close();

    return;
}
