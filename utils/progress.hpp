#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <string>
#include <vector>
#include <chrono>


// -------------------------
// namespace{progress}
// -------------------------
namespace progress{

    // Function Prototype
    std::string separator();
    std::string separator_center(const std::string word);
    std::string current_date();

    // ---------------------------------------
    // namespace{progress} -> class{display}
    // ---------------------------------------
    class display{
    private:
        size_t count;
        size_t count_max;
        size_t header, length;
        std::vector<std::string> loss;
        std::vector<float> loss_sum;
        std::vector<float> loss_ave;
        std::chrono::system_clock::time_point start, end;
    public:
        display(){}
        display(const size_t count_max_, const std::pair<size_t, size_t> epoch, const std::vector<std::string> loss_);
        display(const size_t count_max_, const std::string header1, const std::string header2, const std::vector<std::string> loss_);
        void increment(const std::vector<float> loss_value, std::vector<size_t> hide={});
        size_t get_iters();
        std::vector<float> get_ave();
        float get_ave(const int idx);
        ~display();
    };

    // ---------------------------------------
    // namespace{progress} -> class{irregular}
    // ---------------------------------------
    class irregular{
    private:
        size_t count_start, count_end;
        std::string elap, rem, date, date_fin, sec_per;
        std::chrono::system_clock::time_point start, end;
    public:
        irregular(){}
        void restart(const size_t count_start_, const size_t count_end_);
        void nab(const size_t count);
        std::string get_elap();
        std::string get_rem();
        std::string get_date();
        std::string get_date_fin();
        std::string get_sec_per();
    };

}


#endif