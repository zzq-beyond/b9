#ifndef GLASS_SURFACE_DETECTION_HPP
#define GLASS_SURFACE_DETECTION_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

// 缺陷类型：脏污、裂纹、崩边
enum Dtype {smudge, crack, broken};

// first字段存放缺陷类型，second字段存放缺陷坐标
typedef std::pair<Dtype, cv::Rect> Defect;

// 单片玻璃下的所有缺陷
struct DefectData
{
    // defects 使用vector存放多个 defect
    std::vector<Defect> defects;  // vector< pair<Dtype, cv::Rect> >

    // 缺陷检测结果：由于警示缺陷不需要报NG，但会添加的 defects 字段内，故设置此字段
    bool is_ok;

    // 极性检测结果
    bool is_polarity_ok;

    DefectData() : is_ok(true), is_polarity_ok(true) {}
};

class GlassSurfaceDetector
{
private:
    // 算法配置文件
    std::string m_config_file;
    // 背景图片
    cv::Mat m_img_bg;
    // Gamma 矫正的查询表
    cv::Mat m_look_up_table;
    // OpenVINO model
    ov::CompiledModel m_model;
    
    // 以下参数值，可通过算法配置文件配置：
    
    // 是否是边检相机
    bool m_is_glass_edge;
    // 背景图片路径
    std::string m_img_bg_path;
    // 测试图片与背景图片差值方法
    int m_diff_type;
    // Gamma 矫正的 Gamma 值
    double m_gamma_value;
    // 测试图片与背景图片差异阈值
    double m_diff_thresh;
    // 测试图片与背景图片相减后，膨胀操作迭代次数
    int m_diff_dilate_iters;
    // 玻璃区域二值化阈值
    int m_glass_bin_thresh;
    // 玻璃边缘二值化阈值
    int m_edges_bin_thresh;
    // 霍夫直线检测玻璃边缘
    // 霍夫直线：投票累加器，只有大于该阈值，才会被返回
    int m_hough_thresh;
    // 霍夫直线：最小直线长度，只有大于该阈值，才会被返回
    double m_hough_min_line_len;
    // 霍夫直线：同一直线上的点之间允许的最大间距
    double m_hough_max_line_gap;
    // 水平直线与水平轴 或 竖直直线与垂直轴 的最大倾斜角度
    double m_line_max_included_angle;
    // 图片中最小玻璃区域面积
    double m_min_glass_area;
    
    // 检测缺陷时，自适应二值化，滑动窗大小
    int m_adapt_ksize;
    // 检测缺陷时，自适应二值化，常数C
    double m_adapt_const;
    // 警示缺陷最小外接矩形的像素宽度
    int m_min_warn_defect_rect_width;
    // 警示缺陷最小外接矩形的像素高度
    int m_min_warn_defect_rect_height;
    // 警示缺陷面积过滤时的阈值
    double m_min_warn_defect_area;
    // 致命缺陷最小外接矩形的像素宽度
    int m_min_fatal_defect_rect_width;
    // 致命缺陷最小外接矩形的像素高度
    int m_min_fatal_defect_rect_height;
    // 致命缺陷面积过滤时的阈值
    double m_min_fatal_defect_area;
    // 缺陷最大检测数量
    int m_max_defect_count;

    // 是否需要开启角检
    bool m_polarity_detection;
    // 当前玻璃边是否存在极性角
    bool m_has_polarity;
    // 倒角的最小水平边长
    double m_polarity_min_hlen;
    // 倒角的最大水平边长
    double m_polarity_max_hlen;
    // 倒角的最小竖直边长
    double m_polarity_min_vlen;
    // 倒角的最大竖直边长
    double m_polarity_max_vlen;
    // 倒角的极性角NG比值
    double m_polarity_ng_ratio;

public:
    // 默认构造函数
    GlassSurfaceDetector(const std::string& config_file = "");
    // 析构函数
    ~GlassSurfaceDetector();
    // detect方法：检测缺陷
    DefectData detect(const cv::Mat& img_in, cv::Mat& img_out);

private:
    // 轮廓类型
    typedef std::vector<cv::Point> CONTOUR_TYPE;
    typedef std::vector<CONTOUR_TYPE> CONTOURS_TYPE;

    // detect_glass方法：检测玻璃区域
    bool detect_glass(const cv::Mat& gray_in, cv::Mat& mask_out, 
        cv::Mat& img_out, bool& is_polarity_ok);

    // get_biggest_contour方法：确定面积最大的轮廓
    static bool get_biggest_contour(const CONTOURS_TYPE& contours_in, 
        CONTOUR_TYPE& contour_out, double& max_cnt_area);

    // line_intersection方法：求解两直线交点
    static cv::Point line_intersection(std::vector<cv::Point> line1, 
        std::vector<cv::Point> line2);
};

#endif // GLASS_SURFACE_DETECTION_HPP
