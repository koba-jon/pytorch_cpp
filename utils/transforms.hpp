#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

#include <vector>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#define CV_MAT false
#define TORCH_TENSOR true


// -----------------------
// namespace{transforms}
// -----------------------
namespace transforms{


    // ----------------------------------------
    // namespace{transforms} -> class{Compose}
    // ----------------------------------------
    class Compose{
    public:
        Compose(){}
        virtual bool type() = 0;
        virtual void forward(cv::Mat &data_in, cv::Mat &data_out) = 0;
        virtual void forward(cv::Mat &data_in, torch::Tensor &data_out) = 0;
        virtual void forward(torch::Tensor &data_in, cv::Mat &data_out) = 0;
        virtual void forward(torch::Tensor &data_in, torch::Tensor &data_out) = 0;
        virtual ~Compose(){}
    };

    // Function Prototype
    torch::Tensor apply(std::vector<transforms::Compose*> &transform, cv::Mat &data_in);
    template <typename T_in, typename T_out> void forward(std::vector<transforms::Compose*> &transform_, T_in &data_in, T_out &data_out, const int count);


    // ----------------------------------------------------
    // namespace{transforms} -> class{Grayscale}(Compose)
    // ----------------------------------------------------
    class Grayscale : Compose{
    private:
        int channels;
    public:
        Grayscale(){}
        Grayscale(const int channels_=1);
        bool type() override{return CV_MAT;}
        void forward(cv::Mat &data_in, cv::Mat &data_out) override;
        void forward(cv::Mat &data_in, torch::Tensor &data_out) override{}
        void forward(torch::Tensor &data_in, cv::Mat &data_out) override{}
        void forward(torch::Tensor &data_in, torch::Tensor &data_out) override{}
    };


    // ----------------------------------------------------
    // namespace{transforms} -> class{Resize}(Compose)
    // ----------------------------------------------------
    class Resize : Compose{
    private:
        cv::Size size;
        int interpolation;
    public:
        Resize(){}
        Resize(const cv::Size size_, const int interpolation_=cv::INTER_LINEAR);
        bool type() override{return CV_MAT;}
        void forward(cv::Mat &data_in, cv::Mat &data_out) override;
        void forward(cv::Mat &data_in, torch::Tensor &data_out) override{}
        void forward(torch::Tensor &data_in, cv::Mat &data_out) override{}
        void forward(torch::Tensor &data_in, torch::Tensor &data_out) override{}
    };
    

    // ----------------------------------------------------
    // namespace{transforms} -> class{ConvertIndex}(Compose)
    // ----------------------------------------------------
    class ConvertIndex : Compose{
    private:
        int before, after;
    public:
        ConvertIndex(){}
        ConvertIndex(const int before_, const int after_);
        bool type() override{return CV_MAT;}
        void forward(cv::Mat &data_in, cv::Mat &data_out) override;
        void forward(cv::Mat &data_in, torch::Tensor &data_out) override{}
        void forward(torch::Tensor &data_in, cv::Mat &data_out) override{}
        void forward(torch::Tensor &data_in, torch::Tensor &data_out) override{}
    };


    // ----------------------------------------------------
    // namespace{transforms} -> class{ToTensor}(Compose)
    // ----------------------------------------------------
    class ToTensor : Compose{
    public:
        ToTensor(){}
        bool type() override{return TORCH_TENSOR;}
        void forward(cv::Mat &data_in, cv::Mat &data_out) override{}
        void forward(cv::Mat &data_in, torch::Tensor &data_out) override;
        void forward(torch::Tensor &data_in, cv::Mat &data_out) override{}
        void forward(torch::Tensor &data_in, torch::Tensor &data_out) override{}
    };


    // -------------------------------------------------------
    // namespace{transforms} -> class{ToTensorLabel}(Compose)
    // -------------------------------------------------------
    class ToTensorLabel : Compose{
    public:
        ToTensorLabel(){}
        bool type() override{return TORCH_TENSOR;}
        void forward(cv::Mat &data_in, cv::Mat &data_out) override{}
        void forward(cv::Mat &data_in, torch::Tensor &data_out) override;
        void forward(torch::Tensor &data_in, cv::Mat &data_out) override{}
        void forward(torch::Tensor &data_in, torch::Tensor &data_out) override{}
    };
    

    // ----------------------------------------------------
    // namespace{transforms} -> class{Normalize}(Compose)
    // ----------------------------------------------------
    class Normalize : Compose{
    private:
        float mean, std;
    public:
        Normalize(){}
        Normalize(const float mean_, const float std_);
        bool type() override{return TORCH_TENSOR;}
        void forward(cv::Mat &data_in, cv::Mat &data_out) override{}
        void forward(cv::Mat &data_in, torch::Tensor &data_out) override{}
        void forward(torch::Tensor &data_in, cv::Mat &data_out) override{}
        void forward(torch::Tensor &data_in, torch::Tensor &data_out) override;
    };


}

#endif