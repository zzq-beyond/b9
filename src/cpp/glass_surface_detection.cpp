#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvdef.h>
#include "glass_surface_detection.hpp"

using std::atan;
using std::abs;
using std::pow;

const double EPSILON = CV_PI - 3.1415926535;
const double Rad2Deg = 180.0 / CV_PI;

// 是否是边检相机
#define IS_GLASS_EDGE_KEY "is_glass_edge"
#define IS_GLASS_EDGE_VAL false
// 背景图片默认路径
#define IMAGE_BG_PATH_KEY "image_bg_path"
#define IMAGE_BG_PATH_VAL "bg.bmp"
// 测试图片与背景图片差值方法
#define DIFF_TYPE_KEY "diff_type"
#define DIFF_TYPE_VAL 0
// 图片二值化的参数
#define GAMMA_VALUE_KEY "gamma_value"
#define GAMMA_VALUE_VAL 3.0
// 测试图片与背景图片差异阈值
#define DIFF_THRESH_KEY "diff_thresh"
#define DIFF_THRESH_VAL 10.0
// 测试图片与背景图片相减后，膨胀操作迭代次数
#define DIFF_DILATE_ITERS_KEY "diff_dilate_iters"
#define DIFF_DILATE_ITERS_VAL 3
// 玻璃区域二值化阈值
#define GLASS_BIN_THRESH_KEY "glass_bin_thresh"
#define GLASS_BIN_THRESH_VAL 100
// 玻璃边缘二值化阈值
#define EDGES_BIN_THRESH_KEY "edges_bin_thresh"
#define EDGES_BIN_THRESH_VAL 200
// 霍夫直线检测玻璃边缘
// 霍夫直线：投票累加器，只有大于该阈值，才会被返回
#define HOUGH_THRESH_KEY "hough_thresh"
#define HOUGH_THRESH_VAL 400
// 霍夫直线：最小直线长度，只有大于该阈值，才会被返回
#define HOUGH_MIN_LINE_LEN_KEY "hough_min_line_len"
#define HOUGH_MIN_LINE_LEN_VAL 200.0
// 霍夫直线：同一直线上的点之间允许的最大间距
#define HOUGH_MAX_LINE_GAP_KEY "hough_max_line_gap"
#define HOUGH_MAX_LINE_GAP_VAL 2.0
// 过滤水平（竖直）直线的参数
#define LINE_MAX_INCLUDED_ANGLE_KEY "line_max_included_angle"
#define LINE_MAX_INCLUDED_ANGLE_VAL 5.0
// 图片中最小玻璃区域面积
#define MIN_GLASS_AREA_KEY "min_glass_area"
#define MIN_GLASS_AREA_VAL 2000000.0    // (4k * 4k) / 8

// 检测缺陷时，自适应二值化，滑动窗大小
#define ADAPT_KSIZE_KEY "adapt_ksize"
#define ADAPT_KSIZE_VAL 39
// 检测缺陷时，自适应二值化，常数C
#define ADAPT_CONST_KEY "adapt_const"
#define ADAPT_CONST_VAL 20.0
// 警示缺陷最小外接矩形的像素宽度
#define MIN_WARN_DEFECT_RECT_WIDTH_KEY "min_warn_defect_rect_width"
#define MIN_WARN_DEFECT_RECT_WIDTH_VAL 2
// 警示缺陷最小外接矩形的像素高度
#define MIN_WARN_DEFECT_RECT_HEIGHT_KEY "min_warn_defect_rect_height"
#define MIN_WARN_DEFECT_RECT_HEIGHT_VAL 2
// 警示缺陷面积过滤时的阈值
#define MIN_WARN_DEFECT_AREA_KEY "min_warn_defect_area"
#define MIN_WARN_DEFECT_AREA_VAL 3.0
// 致命缺陷最小外接矩形的像素宽度
#define MIN_FATAL_DEFECT_RECT_WIDTH_KEY "min_fatal_defect_rect_width"
#define MIN_FATAL_DEFECT_RECT_WIDTH_VAL 10
// 致命缺陷最小外接矩形的像素高度
#define MIN_FATAL_DEFECT_RECT_HEIGHT_KEY "min_fatal_defect_rect_height"
#define MIN_FATAL_DEFECT_RECT_HEIGHT_VAL 10
// 致命缺陷面积过滤时的阈值
#define MIN_FATAL_DEFECT_AREA_KEY "min_fatal_defect_area"
#define MIN_FATAL_DEFECT_AREA_VAL 30.0
// 缺陷最大检测数量
#define MAX_DEFECT_COUNT_KEY "max_defect_count"
#define MAX_DEFECT_COUN_VAL 30

// 是否需要开启角检
#define POLARITY_DETECTION_KEY "polarity_detection"
#define POLARITY_DETECTION_VAL false
// 当前玻璃边是否存在极性角
#define HAS_POLARITY_KEY "has_polarity"
#define HAS_POLARITY_VAL false
// 倒角的最小水平边长
#define POLARITY_MIN_HLEN_KEY "polarity_min_hlen"
#define POLARITY_MIN_HLEN_VAL 15.0
// 倒角的最大水平边长
#define POLARITY_MAX_HLEN_KEY "polarity_max_hlen"
#define POLARITY_MAX_HLEN_VAL 40.0
// 倒角的最小竖直边长
#define POLARITY_MIN_VLEN_KEY "polarity_min_vlen"
#define POLARITY_MIN_VLEN_VAL 15.0
// 倒角的最大竖直边长
#define POLARITY_MAX_VLEN_KEY "polarity_max_vlen"
#define POLARITY_MAX_VLEN_VAL 120.0
// 倒角的极性角NG比值
#define POLARITY_NG_RATIO_KEY "polarity_ng_ratio"
#define POLARITY_NG_RATIO_VAL 2.2


std::vector<float> 
softmax(const float* _arr, size_t _n) 
{
    std::vector<float> probs;
    float sum_exps = 0.0;
    for (size_t i = 0; i < _n; ++i) 
    {
        float exp_value = std::exp(_arr[i]);
        probs.push_back(exp_value);
        sum_exps += exp_value;
    }
    
    for (float& value : probs) {
        value /= sum_exps;
    }
    return probs;
}

ov::Tensor 
prepare_tensor(const cv::Mat& roi)
{
    cv::Mat blob;
    cv::cvtColor(roi, blob, cv::COLOR_BGR2RGB);

    cv::resize(blob, blob, cv::Size(224, 224));
    blob.convertTo(blob, CV_32FC3, 1 / 255.0);
    blob -= cv::Scalar(0.485, 0.456, 0.406);
    blob /= cv::Scalar(0.229, 0.224, 0.225);

    ov::element::Type input_type = ov::element::f32;
    ov::Shape input_shape = {1, 224, 224, 3};
    ov::Tensor tensor = ov::Tensor(input_type, input_shape, blob.data);
    return tensor;
}

ov::CompiledModel 
init_model(std::string& model_path)
{
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    ov::element::Type input_type = ov::element::f32;
    ov::Shape input_shape = {1, 224, 224, 3};
    const ov::Layout tensor_layout{"NHWC"};

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();

    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

    return compiled_model;
}

std::pair<size_t, float> 
infer(ov::CompiledModel& model, const cv::Mat& roi)
{
    ov::InferRequest infer_request = model.create_infer_request();
    ov::Tensor tensor_ov = prepare_tensor(roi);
    infer_request.set_input_tensor(tensor_ov);
    infer_request.infer();

    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    const float* output_arr = output_tensor.data<const float>();
    // std::cout << output_tensor.get_size() << std::endl;
    // std::cout << output_arr[0] << std::endl;
    // std::cout << output_arr[1] << std::endl;
    size_t sz = output_tensor.get_size();
    std::vector<float> output_sm = softmax(output_arr, sz);
    size_t idx = std::distance(output_sm.begin(), std::max_element(output_sm.begin(), output_sm.end()));
    float prob = output_sm[idx];
    std::cout << "class id: "<< idx << ", prob: " << prob << std::endl;
    std::pair<size_t, float> res = std::make_pair(idx, prob);
    return res;
}


GlassSurfaceDetector::GlassSurfaceDetector(const std::string& config_file) 
    : m_config_file(config_file)
{
    namespace fs = boost::filesystem;
    namespace json = boost::json;
    const json::kind t_int64 = json::kind::int64;
    const json::kind t_double = json::kind::double_;
    const json::kind t_string = json::kind::string;
    const json::kind t_bool = json::kind::bool_;

    // 检测缺陷时的参数
    m_is_glass_edge                  = IS_GLASS_EDGE_VAL;
    m_img_bg_path                    = IMAGE_BG_PATH_VAL;
    m_diff_type                      = DIFF_TYPE_VAL;
    m_gamma_value                    = GAMMA_VALUE_VAL;
    m_diff_thresh                    = DIFF_THRESH_VAL;
    m_diff_dilate_iters              = DIFF_DILATE_ITERS_VAL;
    m_glass_bin_thresh               = GLASS_BIN_THRESH_VAL;
    m_edges_bin_thresh               = EDGES_BIN_THRESH_VAL;
    m_hough_thresh                   = HOUGH_THRESH_VAL;
    m_hough_min_line_len             = HOUGH_MIN_LINE_LEN_VAL;
    m_hough_max_line_gap             = HOUGH_MAX_LINE_GAP_VAL;
    m_line_max_included_angle        = LINE_MAX_INCLUDED_ANGLE_VAL;
    m_min_glass_area                 = MIN_GLASS_AREA_VAL;
    
    m_adapt_ksize                    = ADAPT_KSIZE_VAL;
    m_adapt_const                    = ADAPT_CONST_VAL;
    m_min_warn_defect_rect_width     = MIN_WARN_DEFECT_RECT_WIDTH_VAL;
    m_min_warn_defect_rect_height    = MIN_WARN_DEFECT_RECT_HEIGHT_VAL;
    m_min_warn_defect_area           = MIN_WARN_DEFECT_AREA_VAL;
    m_min_fatal_defect_rect_width    = MIN_FATAL_DEFECT_RECT_WIDTH_VAL;
    m_min_fatal_defect_rect_height   = MIN_FATAL_DEFECT_RECT_HEIGHT_VAL;
    m_min_fatal_defect_area          = MIN_FATAL_DEFECT_AREA_VAL;
    m_max_defect_count               = MAX_DEFECT_COUN_VAL;

    m_polarity_detection             = POLARITY_DETECTION_VAL;
    m_has_polarity                   = HAS_POLARITY_VAL;
    m_polarity_min_hlen              = POLARITY_MIN_HLEN_VAL;
    m_polarity_max_hlen              = POLARITY_MAX_HLEN_VAL;
    m_polarity_min_vlen              = POLARITY_MIN_VLEN_VAL;
    m_polarity_max_vlen              = POLARITY_MAX_VLEN_VAL;
    m_polarity_ng_ratio              = POLARITY_NG_RATIO_VAL;

    // 读取配置文件
    if (fs::is_regular_file(fs::path(m_config_file)))
    {
        std::ifstream ifs(m_config_file);
        std::string input(std::istreambuf_iterator<char>(ifs), {});
        const json::value jv = json::parse(input);
        if (jv.kind() == json::kind::object)
        {
            const auto& obj = jv.get_object();
            if (!obj.empty())
            {
                for (const auto& iter : obj)
                {
                    const json::string_view key = iter.key();
                    const json::value jvv = iter.value();
                    const json::kind kind = jvv.kind();

                    // 玻璃区域检测
                    if (key == IS_GLASS_EDGE_KEY && kind == t_bool)
                        m_is_glass_edge = jvv.get_bool();
                    
                    else if (key == IMAGE_BG_PATH_KEY && kind == t_string)
                        m_img_bg_path = jvv.get_string().c_str();
                    
                    else if (key == DIFF_TYPE_KEY && kind == t_int64)
                        m_diff_type = jvv.get_int64();

                    else if (key == GAMMA_VALUE_KEY && kind == t_double)
                        m_gamma_value = jvv.get_double();

                    else if (key == DIFF_THRESH_KEY && kind == t_double)
                        m_diff_thresh = jvv.get_double();

                    else if (key == DIFF_DILATE_ITERS_KEY && kind == t_int64)
                        m_diff_dilate_iters = jvv.get_int64();

                    else if (key == GLASS_BIN_THRESH_KEY && kind == t_int64)
                        m_glass_bin_thresh = jvv.get_int64();

                    else if (key == EDGES_BIN_THRESH_KEY && kind == t_int64)
                        m_edges_bin_thresh = jvv.get_int64();
                    
                    else if (key == HOUGH_THRESH_KEY && kind == t_int64)
                        m_hough_thresh = jvv.get_int64();

                    else if (key == HOUGH_MIN_LINE_LEN_KEY && kind == t_double)
                        m_hough_min_line_len = jvv.get_double();

                    else if (key == HOUGH_MAX_LINE_GAP_KEY && kind == t_double)
                        m_hough_max_line_gap = jvv.get_double();

                    else if (key == LINE_MAX_INCLUDED_ANGLE_KEY && kind == t_double)
                        m_line_max_included_angle = jvv.get_double();

                    else if (key == MIN_GLASS_AREA_KEY && kind == t_double)
                        m_min_glass_area = jvv.get_double();

                    else if (key == MAX_DEFECT_COUNT_KEY && kind == t_int64)
                        m_max_defect_count = jvv.get_int64();

                    // 缺陷检测
                    else if (key == ADAPT_KSIZE_KEY && kind == t_int64)
                        m_adapt_ksize = jvv.get_int64();

                    else if (key == ADAPT_CONST_KEY && kind == t_double)
                        m_adapt_const = jvv.get_double();

                    else if (key == MIN_WARN_DEFECT_RECT_WIDTH_KEY && kind == t_int64)
                        m_min_warn_defect_rect_width = jvv.get_int64();

                    else if (key == MIN_WARN_DEFECT_RECT_HEIGHT_KEY && kind == t_int64)
                        m_min_warn_defect_rect_height = jvv.get_int64();

                    else if (key == MIN_WARN_DEFECT_AREA_KEY && kind == t_double)
                        m_min_warn_defect_area = jvv.get_double();

                    else if (key == MIN_FATAL_DEFECT_RECT_WIDTH_KEY && kind == t_int64)
                        m_min_fatal_defect_rect_width = jvv.get_int64();

                    else if (key == MIN_FATAL_DEFECT_RECT_HEIGHT_KEY && kind == t_int64)
                        m_min_fatal_defect_rect_height = jvv.get_int64();

                    else if (key == MIN_FATAL_DEFECT_AREA_KEY && kind == t_double)
                        m_min_fatal_defect_area = jvv.get_double();

                    // 极性检测
                    else if (key == POLARITY_DETECTION_KEY && kind == t_bool)
                        m_polarity_detection = jvv.get_bool();

                    else if (key == HAS_POLARITY_KEY && kind == t_bool)
                        m_has_polarity = jvv.get_bool();

                    else if (key == POLARITY_MIN_HLEN_KEY && kind == t_double)
                        m_polarity_min_hlen = jvv.get_double();

                    else if (key == POLARITY_MAX_HLEN_KEY && kind == t_double)
                        m_polarity_max_hlen = jvv.get_double();

                    else if (key == POLARITY_MIN_VLEN_KEY && kind == t_double)
                        m_polarity_min_vlen = jvv.get_double();

                    else if (key == POLARITY_MAX_VLEN_KEY && kind == t_double)
                        m_polarity_max_vlen = jvv.get_double();

                    else if (key == POLARITY_NG_RATIO_KEY && kind == t_double)
                        m_polarity_ng_ratio = jvv.get_double();
                    
                } // end for
            } // end if obj
        } // end if kind
    } // end if file

    // 读取背景图片
    m_img_bg = cv::imread(m_img_bg_path, 0);

    // 计算 Gamma 矫正查询表
    m_look_up_table = cv::Mat(1, 256, CV_8U);
    uchar* p = m_look_up_table.ptr();
    for(int i = 0; i < 256; ++i)
        p[i] = uchar(pow(i / 255.0, 1.0 / m_gamma_value) * 255.0);

    // 初始化 OpenVINO 模型
    std::string model_path = "/home/zzq/code/b9/models/model.xml";
    m_model = init_model(model_path);

    // 打印参数，用于确认
    std::cout << "*****************************************\n";
    std::cout << "is_glass_edge = [" << m_is_glass_edge << "]\n";
    std::cout << "image_bg_path = [" << m_img_bg_path << "]\n";
    std::cout << "diff_type = [" << m_diff_type << "]\n";
    std::cout << "gamma_value = [" << m_gamma_value << "]\n";
    std::cout << "diff_thresh = [" << m_diff_thresh << "]\n";
    std::cout << "diff_dilate_iters = [" << m_diff_dilate_iters << "]\n";
    std::cout << "glass_bin_thresh = [" << m_glass_bin_thresh << "]\n";
    std::cout << "edges_bin_thresh = [" << m_edges_bin_thresh << "]\n";
    std::cout << "hough_thresh = [" << m_hough_thresh << "]\n";
    std::cout << "hough_min_line_len = [" << m_hough_min_line_len << "]\n";
    std::cout << "hough_max_line_gap = [" << m_hough_max_line_gap << "]\n";
    std::cout << "line_max_included_angle = [" << m_line_max_included_angle << "]\n";
    std::cout << "min_glass_area = [" << m_min_glass_area << "]\n\n";

    std::cout << "adapt_ksize = [" << m_adapt_ksize << "]\n";
    std::cout << "adapt_const = [" << m_adapt_const << "]\n";
    std::cout << "min_warn_defect_rect_width = [" << m_min_warn_defect_rect_width << "]\n";
    std::cout << "min_warn_defect_rect_height = [" << m_min_warn_defect_rect_height << "]\n";
    std::cout << "min_warn_defect_area = [" << m_min_warn_defect_area << "]\n";
    std::cout << "min_fatal_defect_rect_width = [" << m_min_fatal_defect_rect_width << "]\n";
    std::cout << "min_fatal_defect_rect_height = [" << m_min_fatal_defect_rect_height << "]\n";
    std::cout << "min_fatal_defect_area = [" << m_min_fatal_defect_area << "]\n";
    std::cout << "max_defect_count = [" << m_max_defect_count << "]\n\n";

    std::cout << "polarity_detection = [" << m_polarity_detection << "]\n";
    std::cout << "has_polarity = [" << m_has_polarity << "]\n";
    std::cout << "polarity_min_hlen = [" << m_polarity_min_hlen << "]\n";
    std::cout << "polarity_max_hlen = [" << m_polarity_max_hlen << "]\n";
    std::cout << "polarity_min_vlen = [" << m_polarity_min_vlen << "]\n";
    std::cout << "polarity_max_vlen = [" << m_polarity_max_vlen << "]\n";
    std::cout << "polarity_ng_ratio = [" << m_polarity_ng_ratio << "]\n";
    std::cout << "*****************************************\n";

}

GlassSurfaceDetector::~GlassSurfaceDetector() { /*暂无资源需要释放*/ }

bool
GlassSurfaceDetector::detect_glass(const cv::Mat& gray_in, cv::Mat& mask_out, 
    cv::Mat& img_out, bool& is_polarity_ok)
{
    // 用于定位玻璃区域位置
    mask_out = cv::Mat::zeros(gray_in.size(), CV_8UC1);

    // 移除背景影响
    cv::Mat no_bg, no_bg_lut;
    if (m_diff_type == 0)
    {
        // NOTE：滤除平场校正后带来的亮长条纹不利影响
        cv::subtract(m_img_bg, gray_in, no_bg);
        no_bg = cv::abs(no_bg);
        cv::imwrite("/home/zzq/code/b9/temp/00.jpg", no_bg);
    }
    else
    {
        // NOTE：滤除光源拼接缝处产生的不利影响
        cv::absdiff(gray_in, m_img_bg, no_bg);
    }
    cv::imwrite("/home/zzq/code/b9/temp/01.jpg", no_bg);
    cv::threshold(no_bg, no_bg, m_diff_thresh, 255, cv::THRESH_TOZERO);//注意不是二值化阈值
    cv::imwrite("/home/zzq/code/b9/temp/02.jpg", no_bg);
    cv::normalize(no_bg, no_bg, 0, 255, cv::NORM_MINMAX);
    cv::imwrite("/home/zzq/code/b9/temp/03.jpg", no_bg);
    cv::LUT(no_bg, m_look_up_table, no_bg_lut); //no_bg_lut是比较关键的一个环节
    cv::imwrite("/home/zzq/code/b9/temp/04.jpg", no_bg_lut);
    
    // 找寻玻璃区域：最大亮区域
    cv::dilate(no_bg_lut, no_bg, cv::noArray(), cv::Point(-1, -1), m_diff_dilate_iters);
    cv::imwrite("/home/zzq/code/b9/temp/05.jpg", no_bg);
    cv::threshold(no_bg, no_bg, m_glass_bin_thresh, 255, cv::THRESH_BINARY);
    cv::imwrite("/home/zzq/code/b9/temp/06.jpg", no_bg);
    cv::erode(no_bg, no_bg, cv::noArray(), cv::Point(-1, -1), m_diff_dilate_iters);
    cv::imwrite("/home/zzq/code/b9/temp/07.jpg", no_bg);
    CONTOURS_TYPE contours; //vector< vector<cv::Point> >

    cv::findContours(no_bg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // for(auto i = contours.begin(); i != contours.end(); i++){
    //         std::cout << *i << " ";
    // }

    CONTOUR_TYPE max_cnt; //vector<cv::Point>
    double max_cnt_area;
    bool ret = get_biggest_contour(contours, max_cnt, max_cnt_area);
    std::cout << "max_cnt_area: " << max_cnt_area << std::endl;
     
    if (ret && (max_cnt_area > m_min_glass_area)) //当大于最小玻璃区域时候
    {
        // std::cout << "coming... when area > 2000000" << std::endl;
        CONTOURS_TYPE vec_max_cnt;
        vec_max_cnt.push_back(max_cnt);
        cv::drawContours(mask_out, vec_max_cnt, -1, cv::Scalar(255), -1);
        cv::imwrite("/home/zzq/code/b9/temp/08.jpg", mask_out);

        // 凸显玻璃边缘区域
        cv::Mat edges, edges_v;
        cv::imwrite("/home/zzq/code/b9/temp/09.jpg", no_bg_lut);
        cv::threshold(no_bg_lut, edges, m_edges_bin_thresh, 255, cv::THRESH_BINARY);
        cv::imwrite("/home/zzq/code/b9/temp/10.jpg",edges);
        edges.copyTo(edges_v);
        // 滤除玻璃边缘的干扰，将阈值识别出来的边缘轮廓画成标准的直线
        std::vector<cv::Vec4f> lines;
        cv::HoughLinesP(edges, lines, 1.0, CV_PI / 180, m_hough_thresh,
            m_hough_min_line_len, m_hough_max_line_gap);
        cv::Mat mask_edges_h = cv::Mat::zeros(gray_in.size(), CV_8UC1);
        cv::Mat mask_edges_v = cv::Mat::zeros(gray_in.size(), CV_8UC1);
        cv::imwrite("/home/zzq/code/b9/temp/11.jpg", lines);
    
        for(const auto& line: lines)
        {
            double k = (line[3] - line[1]) / (line[2] - line[0] + EPSILON);
            double b = line[1] - k * line[0];
            double degree = std::atan(k) * Rad2Deg;

            if (std::abs(degree) <= m_line_max_included_angle)  //水平直线
            {
                cv::line(mask_edges_h, cv::Point(line[0], line[1]),
                    cv::Point(line[2], line[3]), cv::Scalar(255), 3);
            }
            else if (std::abs(std::abs(degree) - 90.0) <= m_line_max_included_angle)  // 竖向直线
            {
                cv::line(mask_edges_v, cv::Point(line[0], line[1]),
                cv::Point(line[2], line[3]), cv::Scalar(255), 3);
            }
        }
        cv::imwrite("/home/zzq/code/b9/temp/11-1.jpg", mask_edges_h);
        cv::imwrite("/home/zzq/code/b9/temp/11-2.jpg", mask_edges_v);

        // 水平直线
        cv::Mat nonzeros_h;
        cv::findNonZero(mask_edges_h, nonzeros_h);
        cv::imwrite("/home/zzq/code/b9/temp/12.jpg", mask_edges_h);
        // std::cout << nonzeros_h.size().height << "=====" << nonzeros_h.size().width << std::endl;
        
        std::vector<cv::Point> best_hline;  // 用于角检分析
        cv::Mat mask_best_hline = cv::Mat::zeros(gray_in.size(), CV_8UC1); // 用于角检分析
        if (nonzeros_h.size().height > 1024) // 统计非零点的个数
        {
            cv::Vec4f obj_line;
            cv::fitLine(nonzeros_h, obj_line, cv::DIST_HUBER, 0.0, 0.01, 0.01);
            // std::cout<< obj_line << "===";
            double k = obj_line[1] / (obj_line[0] + EPSILON);
            double b = obj_line[3] - k * obj_line[2];
            double degree = std::atan(k) * Rad2Deg;
            //  std::cout << "角度1: " << std::abs(std::abs(degree)) << std::endl;
            if (std::abs(degree) <= m_line_max_included_angle)
            {
                int x1, y1, x2, y2;
                x1 = 0;
                x2 = gray_in.size().width - 1;
                // std::cout << x2 << "===";
                y1 = int(k * x1 + b);
                y2 = int(k * x2 + b);
                cv::line(mask_out, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0), 32);
                cv::line(img_out, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 3);
                cv::line(mask_best_hline, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), 3);
                cv::imwrite("/home/zzq/code/b9/temp/13.jpg", mask_out);
                cv::imwrite("/home/zzq/code/b9/temp/14.jpg", img_out);
                cv::imwrite("/home/zzq/code/b9/temp/15.jpg", mask_best_hline);

                // 用于角检分析
                best_hline.push_back(cv::Point(x1, y1));
                best_hline.push_back(cv::Point(x2, y2));
                // std::cout << best_hline << "===";

                // 由于竖直方向上，玻璃容易出现左右偏摆，hough直线很可能检测不到直线
                // 过滤掉水平向直线
                cv::line(edges_v, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), 32);
                cv::imwrite("/home/zzq/code/b9/temp/16.jpg", edges_v);

            }
        }
        
        // 边检相机才需要进行下面的操作（竖向直线）
        std::vector<cv::Point> best_vline;  // 用于角检分析
        cv::Mat mask_best_vline = cv::Mat::zeros(gray_in.size(), CV_8UC1); // 用于角检分析
        if (m_is_glass_edge)
        {
            cv::Mat nonzeros_v;
            // mask_edges_v += edges_v;  //没有看懂这一步的操作，什么意思？
            // 滤除小噪点
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::morphologyEx(mask_edges_v, mask_edges_v, cv::MORPH_OPEN, kernel);
            cv::findNonZero(mask_edges_v, nonzeros_v);
            cv::imwrite("/home/zzq/code/b9/temp/16-1.jpg", mask_edges_v);
            // std::cout << "非零点个数：" << nonzeros_v.size().height << std::endl;
            if (nonzeros_v.size().height > 1024)
            {
                cv::Vec4f obj_line;
                cv::fitLine(nonzeros_v, obj_line, cv::DIST_HUBER, 0.0, 0.01, 0.01);
                double k = obj_line[1] / (obj_line[0] + EPSILON);
                double b = obj_line[3] - k * obj_line[2];
                double degree = std::atan(k) * Rad2Deg;
                // std::cout << "角度2: " << std::abs(std::abs(degree) - 90) << std::endl;
                if (std::abs(std::abs(degree) - 90) <= m_line_max_included_angle)
                {
                    int x1, y1, x2, y2;
                    y1 = 0;
                    y2 = gray_in.size().height - 1;
                    x1 = int((y1 - b) / (k + EPSILON));
                    x2 = int((y2 - b) / (k + EPSILON));
                    cv::line(mask_out, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0), 32);
                    cv::imwrite("/home/zzq/code/b9/temp/16-2.jpg", mask_out);
                    cv::line(img_out,  cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 3);
                    cv::imwrite("/home/zzq/code/b9/temp/16-3.jpg", img_out);
                    cv::line(mask_best_vline, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), 3);
                    cv::imwrite("/home/zzq/code/b9/temp/16-4.jpg", mask_best_vline);

                    // 用于角检分析
                    best_vline.push_back(cv::Point(x1, y1));
                    best_vline.push_back(cv::Point(x2, y2));
                }
            }
        }

        if (m_is_glass_edge && !best_hline.empty() && !best_vline.empty())
        {
            // 竖直和水平直线的交点
            cv::Point ipt = line_intersection(best_hline, best_vline);
            int ix = ipt.x;
            int iy = ipt.y;
            cv::circle(img_out, cv::Point(ix, iy), 5, cv::Scalar(0, 0, 255), -1);
            // 验证交点的合法性
            bool valid = true;
            if ((ix < 0) || (ix > gray_in.size().width) ||
                (iy < 0) || (iy > gray_in.size().height))
                valid = false;

            // NOTE：边检相机排除倒角的影响
            cv::circle(mask_out, cv::Point(ix, iy), 150, cv::Scalar(0, 255, 255), -1);
            cv::imwrite("/home/zzq/code/b9/temp/17.jpg", mask_out);
            // 如果需要角检（必定是边检相机）
            if (m_polarity_detection && valid)
            {
                // 求解竖直直线上与交点的最近点
                cv::Point nearest_vpt;
                double min_vdist = std::numeric_limits<double>::infinity();  //double的正无穷大
                cv::Mat filtered_vpoints;
                cv::bitwise_and(edges, mask_best_vline, filtered_vpoints);
                cv::imwrite("/home/zzq/code/b9/temp/17-01.jpg", mask_best_vline);
                cv::imwrite("/home/zzq/code/b9/temp/17-02.jpg", filtered_vpoints);
                std::vector<cv::Point2f> vec_filtered_vpoints;
                cv::findNonZero(filtered_vpoints, vec_filtered_vpoints);
                for (auto& vpt : vec_filtered_vpoints)
                {
                    double dist = cv::norm(cv::Point(vpt) - cv::Point(ix, iy));
                    if (dist < min_vdist)
                    {
                        min_vdist = dist;
                        nearest_vpt = vpt;
                    }
                }
                cv::circle(img_out, nearest_vpt, 5, cv::Scalar(0,255,255), -1);
                cv::imwrite("/home/zzq/code/b9/temp/17-1.jpg", img_out);

                // 求解水平直线上与交点的最近点
                cv::Point nearest_hpt;
                double min_hdist = std::numeric_limits<double>::infinity();
                cv::Mat filtered_hpoints;
                cv::bitwise_and(edges, mask_best_hline, filtered_hpoints);
                cv::imwrite("/home/zzq/code/b9/temp/17-12.jpg", filtered_hpoints);
                std::vector<cv::Point2f> vec_filtered_hpoints;
                cv::findNonZero(filtered_hpoints, vec_filtered_hpoints);
                for (auto& hpt : vec_filtered_hpoints)
                {
                    double dist = cv::norm(cv::Point(hpt) - cv::Point(ix, iy));
                    if (dist < min_hdist)
                    {
                        min_hdist = dist;
                        nearest_hpt = hpt;
                    }
                }
                cv::circle(img_out, nearest_hpt, 5, cv::Scalar(0, 255, 255), -1);
                cv::imwrite("/home/zzq/code/b9/temp/17-2.jpg", img_out);


                cv::line(img_out, cv::Point(ix, iy), nearest_vpt, cv::Scalar(0,255,0), 1);
                cv::line(img_out, cv::Point(ix, iy), nearest_hpt, cv::Scalar(0,255,0), 1);
                cv::line(img_out, nearest_vpt,       nearest_hpt, cv::Scalar(0,255,0), 1);

                double v_len = sqrt(pow(ix - nearest_vpt.x, 2) + pow(iy - nearest_vpt.y, 2));
                double h_len = sqrt(pow(ix - nearest_hpt.x, 2) + pow(iy - nearest_hpt.y, 2));
                // 定义极性角位置：极性角应该出现在玻璃的 左 下 角
                // 定义水平方向：以交点为基准，在交点左边为负方向，否则正方向
                // 定义竖直方向：以交点为基准，在交点上边为负方向，否则正方向
                // 判断逻辑：首先通过判断 高宽比，只有 高宽比 大于一定阈值，才可能是极性角；
                //         再通过判断 极性角方向，需满足 水平方向为正，竖直方向为负
                // 左 上 角方向[H/V]：+/+
                // 右 上 角方向[H/V]：-/+
                // 左 下 角方向[H/V]：+/-
                // 右 下 角方向[H/V]：-/-
                h_len = (nearest_hpt.x > ix ? h_len * 1.0 : h_len * -1.0);
                v_len = (nearest_vpt.y > iy ? v_len * 1.0 : v_len * -1.0);

                // 长度有效性验证
                cv::Scalar green(0,255,0);
                cv::Scalar red(0,0,255);
                std::string text1 = "hlen: null";
                std::string text2 = "vlen: null";
                std::string text3 = "ratio: null";
                std::string text4 = "polarity: null";
                if (abs(h_len) >= m_polarity_min_hlen && abs(h_len) <= m_polarity_max_hlen &&
                    abs(v_len) >= m_polarity_min_vlen && abs(v_len) <= m_polarity_max_vlen)
                {
                    text1 = "hlen: " + std::to_string(h_len);
                    text2 = "vlen: " + std::to_string(v_len);
                    double ratio = abs(v_len / h_len);
                    text3 = "ratio: " + std::to_string(ratio);

                    if (m_has_polarity) // 此部相机下的照片 存在 极性角，即应该是（图中）玻璃左边缘
                    {
                        if (((ratio < m_polarity_ng_ratio) && (h_len > 0) && (v_len < 0)) ||   //左下角
                            ((ratio >= m_polarity_ng_ratio) && (h_len > 0) && (v_len > 0)))    //左上角
                            is_polarity_ok = false;
                    }
                    else // 此部相机下的照片 不存在 极性角，即应该是（图中）玻璃右边缘
                    {
                        if ((ratio >= m_polarity_ng_ratio) && (h_len < 0))
                            is_polarity_ok = false;
                    }
                    text4 = "polarity: " + std::string(is_polarity_ok ? "OK" : "NG");
                }
                else
                    std::cout << "h_len: " << h_len << ", " 
                              << "v_len: " << v_len << ", "
                              << "ratio: " << abs(v_len / h_len) << std::endl;
                cv::putText(img_out, text1, cv::Point(200, 200), cv::FONT_HERSHEY_COMPLEX, 5.0, green, 5.0);
                cv::putText(img_out, text2, cv::Point(200, 350), cv::FONT_HERSHEY_COMPLEX, 5.0, green, 5.0);
                cv::putText(img_out, text3, cv::Point(200, 500), cv::FONT_HERSHEY_COMPLEX, 5.0, green, 5.0);
                cv::putText(img_out, text4, cv::Point(200, 650), cv::FONT_HERSHEY_COMPLEX, 5.0, 
                    (is_polarity_ok ? green : red), 5.0);
            }
        }
    }

    return ret;
}

DefectData
GlassSurfaceDetector::detect(const cv::Mat& img_in, cv::Mat& img_out)
{    
    // 缺陷检测结果
    DefectData results;
    img_in.copyTo(img_out); 
    
    // 转为灰度图
    cv::Mat img_gray;
    cv::cvtColor(img_in, img_gray, cv::COLOR_BGR2GRAY);
    if (m_img_bg.empty())
    {
        std::cout << "[ Warning ] No background image was found. "
                  << "The background will be set zeros.\n";
        m_img_bg = cv::Mat::zeros(img_gray.size(), CV_8UC1);
    }

    // --------------------------- 识别玻璃区域 --------------------------- //
    cv::Mat mask_glass_roi;
    bool is_polarity_ok = true;
    bool ret = detect_glass(img_gray, mask_glass_roi, img_out, is_polarity_ok);//检测玻璃函数
    // img_gray.copyTo(img_out, mask_glass_roi);
    cv::imwrite("/home/zzq/code/b9/temp/18.jpg", mask_glass_roi);
    if (!ret) return results;
    results.is_polarity_ok = is_polarity_ok;
    // -------------------------------------------------------------- //

    // --------------------------- 检测缺陷 -------------------------- //
    results.is_ok = true;
    cv::Mat ada_thresh;
    cv::adaptiveThreshold(img_gray, ada_thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY_INV, m_adapt_ksize, m_adapt_const);
    cv::imwrite("/home/zzq/code/b9/temp/19.jpg", ada_thresh);
    cv::bitwise_and(ada_thresh, mask_glass_roi, ada_thresh);//主要在这里用到了detect_glass的结果，也就是
                                                            //mask_glass_roi
    cv::imwrite("/home/zzq/code/b9/temp/20.jpg", mask_glass_roi);
    cv::imwrite("/home/zzq/code/b9/temp/21.jpg", ada_thresh);

    CONTOURS_TYPE contours;
    cv::findContours(ada_thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // cv::Mat img_gg;                                              //画出缺陷的轮廓
    // img_in.copyTo(img_gg);
    // cv::drawContours(img_gg, contours, -1, cv::Scalar(0,0,255), 1);
    // cv::imwrite("/home/zzq/code/b9/temp/21-1.jpg", img_gg);

    int img_w = img_gray.size().width;
    int img_h = img_gray.size().height;
    int defect_idx = 0;
    for (const auto& cnt : contours)
    {
        cv::Rect rect0 = cv::boundingRect(cnt);
        int x0 = rect0.x;
        int y0 = rect0.y;
        int w0 = rect0.width;
        int h0 = rect0.height;

        // 小缺陷过滤掉
        if (((w0 < m_min_warn_defect_rect_width) && (h0 < m_min_warn_defect_rect_height)) || 
            (cv::contourArea(cnt) < m_min_warn_defect_area)) continue;

        // NOTE：竖向长条纹过滤掉
        if (((w0 < 5) && (h0 > (w0 * 6.0))) || (h0 > (w0 * 12.0))) continue;

        // NOTE：横向长条纹过滤掉
        if (((h0 < 5) && (w0 > (h0 * 6.0))) || (w0 > (h0 * 12.0))) continue;

        // 更新缺陷数量
        defect_idx++;
        
        // 超过最大检测数量退出检测
        if (defect_idx > m_max_defect_count) break;
        
        // 设置缺陷颜色：警示缺陷为黄色，致命缺陷为红色，只有致命缺陷才报NG
        cv::Scalar color(0,255,255); //黄色
        // if (((w0 >= m_min_fatal_defect_rect_width) || (h0 >= m_min_fatal_defect_rect_height)) && 
        //     (cv::contourArea(cnt) >= m_min_fatal_defect_area)) 
        // {
        //     color = cv::Scalar(0,0,255);
        //     // // 设置缺陷检测结果
        //     // results.is_ok = false;
        // }

        // 放大 ROI 与绘制结果
        int MIN_BBOX_HW = 224;
        int MIN_BBOX_OFS = 8;
        const float PROB_THRESH = 0.65;
        int bigger = (w0 > h0 ? w0 : h0);
        cv::Mat roi;
        if (bigger < MIN_BBOX_HW)
        {   //左上角，右下角各扩展8个像素
            int x0_new = (0 > (x0-MIN_BBOX_OFS) ? 0 : (x0-MIN_BBOX_OFS));
            int y0_new = (0 > (y0-MIN_BBOX_OFS) ? 0 : (y0-MIN_BBOX_OFS));
            int x1_new = ((img_w-1) < (x0+w0+MIN_BBOX_OFS) ? (img_w-1) : (x0+w0+MIN_BBOX_OFS));
            int y1_new = ((img_h-1) < (y0+h0+MIN_BBOX_OFS) ? (img_h-1) : (y0+h0+MIN_BBOX_OFS));
            roi = img_in(cv::Rect(cv::Point(x0_new, y0_new), cv::Point(x1_new, y1_new)));
            cv::imwrite("/home/zzq/code/b9/temp/22.jpg", roi);
            cv::Mat roi_resized;
            cv::resize(roi, roi_resized, cv::Size(MIN_BBOX_HW, MIN_BBOX_HW));
            cv::imwrite("/home/zzq/code/b9/temp/23.jpg", roi_resized);
            int cx = (x0_new + x1_new) / 2;
            int cy = (y0_new + y1_new) / 2; //求得中心点
            // ROI 是否需要水平移动
            int x0_test = cx - MIN_BBOX_HW / 2;
            int x1_test = cx + MIN_BBOX_HW / 2;
            if (x0_test < 0)    // 需要右移
            {
                x0_test = 0;
                x1_test = MIN_BBOX_HW;
            }
            else if (x1_test >= img_w)  // 需要左移
            {
                x1_test = img_w-1;
                x0_test = x1_test - MIN_BBOX_HW;
            }
            // ROI 是否需要上下移动
            int y0_test = cy - MIN_BBOX_HW / 2;
            int y1_test = cy + MIN_BBOX_HW / 2;
            if (y0_test < 0)    // 需要下移
            {
                y0_test = 0;
                y1_test = MIN_BBOX_HW;
            }
            else if (y1_test >= img_h)  // 需要上移
            {
                y1_test = img_h-1;
                y0_test = y1_test - MIN_BBOX_HW;
            }
            //计时函数
            auto start = std::chrono::high_resolution_clock::now();
            std::pair<size_t, float> cls = infer(m_model, roi_resized);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dura = end - start;
            std::cout << "OpenVINO inference time: " << dura.count() << "ms\n";
            if (cls.first > 0 && cls.second > PROB_THRESH)
            // if (cls.second > PROB_THRESH)
            {
                if (((w0 >= m_min_fatal_defect_rect_width) || (h0 >= m_min_fatal_defect_rect_height)) && 
                    (cv::contourArea(cnt) >= m_min_fatal_defect_area))
                {
                    color = cv::Scalar(0,0,255);
                    // 设置缺陷检测结果
                    results.is_ok = false;
                }

                // 放大缺陷并填充到目标位置
                cv::Mat roi_tmp(img_out, cv::Rect(x0_test, y0_test, MIN_BBOX_HW, MIN_BBOX_HW));
                roi_resized.copyTo(roi_tmp);
                
                // 绘制缺陷位置
                cv::rectangle(img_out, cv::Point(x0_test, y0_test), cv::Point(x1_test, y1_test), color, 3);
            }
        }
        else    // 大尺寸缺陷
        {
            roi = img_out(rect0);  // retc0是每个轮廓的外界矩形
            std::pair<size_t, float> cls = infer(m_model, roi);
            if (cls.first > 0 && cls.second > PROB_THRESH) 
            {
                color = cv::Scalar(0,0,255);
                results.is_ok = false;
                // 绘制缺陷位置
                cv::rectangle(img_out, rect0, color, 3);
            }
        }
        
        // 添加缺陷类型及坐标
        Defect defect(std::make_pair(Dtype::smudge, rect0));
        results.defects.push_back(defect);
    }
    std::cout << "缺陷数量：" << defect_idx << std::endl;
    // -------------------------------------------------------------- //

    return results;
}

bool
GlassSurfaceDetector::get_biggest_contour(const CONTOURS_TYPE& contours_in,
    CONTOUR_TYPE& contour_out, double& max_cnt_area)
{
    max_cnt_area = 0.0;
    for (const auto& cnt : contours_in)
    {
        double area = cv::contourArea(cnt);
        if (area > max_cnt_area)
        {
            max_cnt_area = area;
            contour_out = cnt;
        }
    }
    if (max_cnt_area > 0.1)
        return true;
    else
        return false;
}

cv::Point 
GlassSurfaceDetector::line_intersection(std::vector<cv::Point> line1, std::vector<cv::Point> line2)
{
    auto det = [](cv::Point2d a, cv::Point2d b) {
        return a.x * b.y - a.y * b.x;
    };

    cv::Point xdiff(line1[0].x - line1[1].x, line2[0].x - line2[1].x);
    cv::Point ydiff(line1[0].y - line1[1].y, line2[0].y - line2[1].y);
    
    int div = det(xdiff, ydiff);
    if (div == 0)
        return cv::Point();

    cv::Point d(det(line1[0], line1[1]), det(line2[0], line2[1]));
    int x = int(det(d, xdiff) / div);
    int y = int(det(d, ydiff) / div);

    return cv::Point(x, y);
}