//g++ heatMap.cpp -o heatmap `pkg-config --libs opencv` -lstdc++fs

#include "heatMap.h"
#include <experimental/filesystem>
#include <string>
#include<vector>
#include <algorithm>
//#include<boost/filesystem.hpp>
namespace fs = std::experimental::filesystem;
namespace heatmap{
    HeatMaps::HeatMaps( std::string filename, std::string output, int rows, int cols, int x, int w, int y, int h ){
        cv::Mat scr = openMat(filename);
        cv::Mat mask = openMat(output);
        //std::cout<<"size mask ->" << mask.rows << " "<<mask.cols<<" "<<mask.channels() <<" size image ->"<<scr.rows<<" " <<scr.cols<<" "<<scr.channels()<<std::endl;
        create_heatmap(scr, mask, rows, cols,x,w,y,h);
        }
    inline cv::Mat  HeatMaps::openMat(std::string filename){
        // open image with 3 channels
        cv::Mat image=cv::imread(filename, CV_LOAD_IMAGE_COLOR);//cv::IMREAD_GRAYSCALE);
        if( !image.data ){
            std::cout<<"Error loadind src n"<< " file ->" <<filename<<std::endl;
            }
        return image;
    }
    void HeatMaps::create_heatmap(cv::Mat&img, cv::Mat&mask, int rows, int cols, int x, int w, int y, int h ){
        cv::Mat im_color;
        cv::applyColorMap(img, im_color, cv::COLORMAP_JET);
        resized(im_color, this->heatMap, rows, cols);
        //cv::imwrite(output, heatMap);
        blend_images(mask, this->heatMap,x,w,y,h); //13,110, 60,187
    }
    inline void HeatMaps::resized(cv::Mat&img, cv::Mat&output, int rows, int cols){
        cv::resize(img, output, cv::Size(cols,rows), cv::INTER_LINEAR);
    }
    void HeatMaps::blend_images(cv::Mat&mask,cv::Mat&image, int x, int w, int y, int h){
        cv::Mat img;
        std::cout<<"size mask ->" << mask.rows << " "<<mask.cols<<" "<<mask.channels() <<" size image ->"<<image.rows<<" " <<image.cols<<" "<<image.channels()<<std::endl;
        cv::addWeighted(mask, 0.5, image, 0.6, 0.0, img );
        this->output = img(cv::Range(x, w), cv::Range(y,h));
    }
}
std::vector<std::string> listdir(const char *directory){
    std::vector<std::string>files;
    int counter =0;
    const fs::path dir{ directory };
    for (auto const& dir_entry : fs::directory_iterator{ dir }){
        const auto filenameStr = dir_entry.path().filename().string();
        //std::cout << dir_entry.path().filename().string()  << "  counter-> "<< counter<< '\n';
        files.push_back(filenameStr);
        counter++;
        }
    return files;
}

std::string extension(std::string file_name, std::string sep, int pos)
{
  //store the position of last '.' in the file name
  int position=file_name.find_last_of(sep);

  //store the characters after the '.' from the file_name string
  std::string result = file_name.substr(position-pos);

  //print the result
  //std::cout<<"The file "<< file_name<<" has <." << result << "> extension."<<std::endl;
  return result;
}
static std::string removeSpaces(std::string str)
{
	str.erase(remove(str.begin(), str.end(), ' '), str.end());
	return str;
}
bool isStringNumber(std::string str){
    for(int i=0;i<str.length();i++){
        if(str[i]<'0'||str[i]>'9'){
            return false;
        }
    }
    return true;
}

int main(){

    const char*m_directory = "/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/images/images/";
    const char *i_directory = "/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/training_data/disparity_maps/dm_4/";

    std::vector<std::string>masks = listdir(m_directory);
    std::vector<std::string>images = listdir(i_directory);
    std::string output = "/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/training_data/finalDM/finaldm4/";



    //int index = ext3.rfind("2", 0);
    //std::cout<<masks.size()<<" "<< images.size()<<" extention -> "<< ext <<" "<< ext3<<" "<< index<<std::endl;
    double scale_down = 1.0; //1.0 is 100%
    int index1 = 0;
    int index2 =0;
    for(int i = 0; i<masks.size();++i){
        std::string ext = masks[i].substr(3,2);//extension(masks[i],".", 2);
        std::string ext1 = masks[i].substr(3,3);
        bool print = isStringNumber(ext);
        bool print1 = isStringNumber(ext1);
        int counter = 0;
        for(int j=0; j<images.size();++j){
            std::string flag = images[j].substr(14,2);
            std::string flag1 = images[j].substr(14,3);
            bool printt = isStringNumber(flag);
            bool printt1 = isStringNumber(flag1);
            std::cout<<masks[i]<<" || "<< images[j]<<" extention -> "<< ext <<" || "<< ext1<<" || "<< flag<<"||"<< flag1<< "||"<<std::endl;
            if((ext.compare(flag)==0)&&(print1==0) &&(printt1==0)){
                std::cout<<masks[i]<<" || "<< images[j]<< " || "<<ext <<" || "<< ext1<<" || "<<" "<< print << " || "<< print1<<" || "<< printt <<" || "<<print1 <<std::endl;
                 heatmap::HeatMaps loader(i_directory+images[j], m_directory+masks[i], 120,160,0,120,0,160);//120,240, 13, 110, 60,187); //100,100, 0, 100, 0,100);
                 cv::Mat heatMap = loader.getBlendedImg();
                 cv::Mat scaled;
                 cv::resize(heatMap,scaled, cv::Size(), scale_down, scale_down, cv::INTER_LINEAR);
                 cv::imwrite(output + images[j], scaled);
            }else if((ext1.compare(flag1)==0)&&(print1==1)&&(printt1==1)){
                //std::cout<<"second condition-> "<<masks[i]<<" || "<< images[j]<<" || "<<ext <<" || "<< ext1<<" || "<<" "<< print << " || "<< print1<<" || "<< printt <<" || "<<print1 <<std::endl;
                 heatmap::HeatMaps loader(i_directory+images[j], m_directory+masks[i], 120,160,0,120,0,160);//120,240, 13, 110, 60,187); //100,100, 0, 100, 0,100);
                 cv::Mat heatMap = loader.getBlendedImg();
                 cv::Mat scaled;
                 cv::resize(heatMap,scaled, cv::Size(), scale_down, scale_down, cv::INTER_LINEAR);
                 cv::imwrite(output + images[j], scaled);
            }
        }
    }/*
    if(masks.size()==images.size()){
        double scale_down = 1.0; //1.0 is 100%
        std::string mask_directory = std::string(m_directory);
        std::string image_directory = std::string(i_directory);
        std::cout<<masks.size()<<" "<< images.size()<<std::endl;
        for(int i =0;i<masks.size();i++){
            std::string ext2 = extension(mask_directory+masks[i],".", 3);
            for(int j =0;j<images.size();j++){
                 std::string ext1 = extension(image_directory+images[j],".", 3);
                 if(ext1==ext2){
                     std::cout << image_directory+images[j]<< "  "<< mask_directory+masks[i]<< " ||" << ext1 << " " << ext2<< '\n';
                     heatmap::HeatMaps loader(image_directory+images[j], mask_directory+masks[i], 120,160,0,120,0,160); //120,240, 13, 110, 60,187); //100,100, 0, 100, 0,100);
                     cv::Mat heatMap = loader.getBlendedImg();
                     cv::Mat scaled;
                     cv::resize(heatMap,scaled, cv::Size(), scale_down, scale_down, cv::INTER_LINEAR);
                     cv::imwrite(output + masks[i], scaled);
                 }
            }
        }
    }
/*
    const char*mask_path = "/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/images/heatmap_30.png";
    const char *path2 = "/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/training_data/binocular_imgs/bino/right/binocular_img_right_30.png";
    std::string path = path2;
    //cv::Mat scr=cv::imread(path2, cv::IMREAD_GRAYSCALE);

    heatmap::HeatMaps loader(path2, mask_path, 120,160,0,120,0,160); //13, 110, 60,187);
    cv::Mat heatMap = loader.getBlendedImg();
    std::cout<< "cols -> "<<heatMap.rows<< "rows -> " <<heatMap.cols<< " "<<heatMap.channels()<<" "<<heatMap.type()<<std::endl;

    cv::imshow("heatmap:", heatMap);
    cv::imwrite("/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/training_data/heatmap_30.png", heatMap);
    cv::waitKey(0);
*/
}