

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
namespace heatmap{
    class HeatMaps{
        public:
        HeatMaps (std::string filename, std::string output, int rows, int cols, int x, int w, int y, int h );
        inline cv::Mat getHeatMap(){
            return heatMap;
        }
        inline cv::Mat getBlendedImg(){
            return output;
        }
        private:
        cv::Mat heatMap;
        cv::Mat output;
        inline cv::Mat openMat(std::string filename);
        void create_heatmap(cv::Mat&img, cv::Mat&mask, int rows, int cols, int x, int w, int y, int h );
        inline void resized(cv::Mat&img, cv::Mat&output, int rows, int cols);
        void blend_images(cv::Mat&mask,cv::Mat&image, int x, int w, int y, int h);

    };
}