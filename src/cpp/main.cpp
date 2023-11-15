#include "glass_surface_detection.hpp"
#include <boost/filesystem.hpp>
#include <numeric>
#include <iomanip>

void help_info(const char* exe_name);

int main(int argc, const char* argv[])
{
    help_info(argv[0]);
    
    namespace bfs = boost::filesystem;
    
    // 检查输入
    if (argc < 2)
    {
        std::cout << "The arguments, passed to, are not enough.\n";
        std::exit(1);
    }
    if (!bfs::exists(argv[1]))
    {
        std::cout << "The input path, \"" << argv[1] << "\" does not exist.\n";
        std::exit(1);
    }
    bfs::path test_img_path = argv[1];

    // 创建目录用于保持推理结果
    bfs::path results_ok_path("./results/ok");
    bfs::path results_ng_path("./results/ng");
    bfs::path temp_path("./temp");
    if (!bfs::exists(results_ok_path))
        bfs::create_directories(results_ok_path);
    if (!bfs::exists(results_ng_path))
        bfs::create_directories(results_ng_path);
    if (!bfs::exists(temp_path))
        bfs::create_directories(temp_path);
    // 获取所有图片路径
    std::vector<std::string> all_images;
    if (bfs::is_directory(test_img_path))
    {
        for (auto iter = bfs::directory_iterator(test_img_path);
             iter != bfs::directory_iterator(); ++iter)
        {
            if (bfs::is_regular_file(iter->path()))
                all_images.push_back(iter->path().string());
        }
    }
    else
        all_images.push_back(test_img_path.string());
    
    GlassSurfaceDetector detector("./GK2/11.json");

    // 对所有图片进行检测
    std::vector<size_t> elapsed;
    for (const auto& img_f : all_images)
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        cv::Mat img_in;
        img_in = cv::imread(img_f, 0);

        cv::Mat img_out;
        DefectData results;
        //=================================所有代码的执行入口==============================
        results = detector.detect(img_in, img_out);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dura = end - start;
        elapsed.push_back(dura.count());

        // 打印log信息
        std::cout << "---------------------------------------" << std::endl;
        std::cout << "image: " << img_f << std::endl;
        std::cout << "inference time: " << dura.count() << "ms" << std::endl;
        std::cout << "count of defects: " << results.defects.size() << std::endl;
        std::cout << "coordinate of defects:" << std::endl;
        for (const auto& res : results.defects)
        {
            std::cout << "\tdtype = " << res.first
                      << std::fixed << std::showpoint << std::setprecision(3)
                      << "\t[x0, y0, w0, h0] = " << "["
                      << res.second.x << ", " << res.second.y << ", "
                      << res.second.width << ", " << res.second.height << "]\n";
        }
        std::cout << std::endl;

        std::size_t found = img_f.find_last_of("/\\");
        std::string fname = img_f.substr(found + 1);
        std::string save_img_f;
        if (results.is_ok && results.is_polarity_ok)
            save_img_f = (results_ok_path / fname).c_str();
        else
            save_img_f = (results_ng_path / fname).c_str();
        cv::imwrite(save_img_f, img_out);
    }
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Average time: " 
              << std::accumulate(elapsed.begin(), elapsed.end(), 0) / elapsed.size()
              << "ms\n";

    return 0;
}

void help_info(const char* exe_name)
{
    std::cout
        << "****************************************\n"
        << "Usage: " << exe_name << " </path/to/image/file/or/folder>\n"
        << "****************************************\n";
}