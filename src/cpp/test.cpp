#include "glass_surface_detection.hpp"
#include <boost/filesystem.hpp>
#include <numeric>
#include <iomanip>
using namespace cv;

int main(){
    
    GlassSurfaceDetector detector("config_edge.json");
    Mat img_in = imread("/home/zzq/code/b9/10.png");

    // GlassSurfaceDetector detector("config.json");
    // Mat img_in = imread("/home/zzq/code/b9/images/2.jpg");

    Mat img_out;
    DefectData results;
    //=================================所有代码的执行入口==============================
    results = detector.detect(img_in, img_out);
    imwrite("/home/zzq/code/b9/result.png", img_out);
    return 0;
}