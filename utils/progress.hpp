#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <string>
#include <vector>
#include <chrono>


// -------------------------
// class{progress_display}
// -------------------------
class progress_display{
private:
    size_t count;
    size_t count_max;
    size_t header, length;
    std::vector<std::string> loss;
    std::vector<float> loss_sum;
    std::vector<float> loss_ave;
    std::chrono::system_clock::time_point start, end;
public:
    progress_display(){}
    progress_display(const size_t count_max_, const std::pair<size_t, size_t> epoch, const std::vector<std::string> loss_);
    void increment(const std::vector<float> loss_value);
    size_t get_iters();
    std::vector<float> get_ave();
    float get_ave(const int index);
    ~progress_display();
};

#endif