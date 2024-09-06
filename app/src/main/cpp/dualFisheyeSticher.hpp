/**
* @file        :dualFisheyeSticher.hpp
* @brief       :全景拼接,以及用于全景图像的各个视角全景变换
* @details     :This is the detail description.
* @date        :2024/07/08 15:10:46
* @author      :cuixingxing(cuixingxing150@gmail.com)
* @version     :1.0
*
* @copyright Copyright (c) 2024
*
*/
#pragma once

#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cctype>
#include <locale>
#include "opencv2/opencv.hpp"

namespace panorama {

typedef struct cameraParam {
    cv::Mat circleFisheyeImage;  // 单个环形鱼眼图像
    cv::Point2f centerPt;        // 畸变中心像素点,[x,y],单位:像素
    float radius;                //环形鱼眼图像的半径, 单位:像素
    float FOV;                   // 单位:度
    float rotateX;               // 左侧鱼眼图绕x轴旋转角度，单位：度
    float rotateY;               // 左侧鱼眼图绕y轴旋转角度，单位：度
    float rotateZ;               // 左侧鱼眼图绕z轴旋转角度，单位：度
} cameraParam;

template <typename _Tp>
std::vector<_Tp> convertMat2Vector(const cv::Mat& mat) {
    return (std::vector<_Tp>)(mat.reshape(1, 1));  //通道数不变，按行转为一行
}

template <typename _Tp>
cv::Mat convertVector2Mat(std::vector<_Tp> v, int channels, int rows) {
    cv::Mat mat = cv::Mat(v);                            //将vector变成单列的mat
    cv::Mat dest = mat.reshape(channels, rows).clone();  //PS：必须clone()一份，否则返回出错
    return dest;
}

// trim from start (in place),https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring
inline void ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
}

// trim from end (in place)
inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(),
            s.end());
}

// trim from both ends (in place)
inline void trim(std::string& s) {
    rtrim(s);
    ltrim(s);
}

inline cv::Mat rotx(double radian) {
    cv::Mat A = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
                 0.0, std::cos(radian), -std::sin(radian),
                 0.0, std::sin(radian), std::cos(radian));
    return A;
}

inline cv::Mat roty(double radian) {
    cv::Mat A = (cv::Mat_<double>(3, 3) << std::cos(radian), 0, std::sin(radian),
                 0.0, 1, 0.0,
                 -std::sin(radian), 0, std::cos(radian));
    return A;
}

inline cv::Mat rotz(double radian) {
    cv::Mat A = (cv::Mat_<double>(3, 3) << std::cos(radian), -std::sin(radian), 0,
                 std::sin(radian), std::cos(radian), 0.0,
                 0.0, 0.0, 1.0);
    return A;
}

// Function to normalize yaw to the range [270, -90]
inline double normalizeYaw(double inputAngle) {
    return std::fmod(inputAngle + 90.0, 360.0) - 90.0;
}
class DualFisheyeSticher {
   public:
    /**
   * @brief       构造函数
   * @details     给定前后双鱼眼图像的参数和指定全景图输出大小配置构造全景图
   * @param[in]   frontCam 前环形鱼眼拼接参数.
   * @param[out]  backCam 后环形鱼眼拼接参数.
   * @param[in]   outputSize 全景图输出大小，[width,height]，单位：像素.
   * @return      无
   * @retval      void
   * @par 标识符
   *     保留
   * @par 其它
   *
   * @par 修改日志
   *      cuixingxing于2024/07/11创建
   */
    DualFisheyeSticher(cameraParam& frontCam, cameraParam& backCam, cv::Size outputSize = cv::Size(5760, 2880));

    /**
    * @brief       构造函数
    * @details     直接给定映射pgm文件，mask文件和指定全景图输出大小配置构造全景图
    * @param[in]   mapX_front_file 前鱼眼映射x坐标表.
    * @param[in]   mapX_back_file 前鱼眼映射y坐标表.
    * @param[in]   mapY_back_file 后鱼眼映射x坐标表.
    * @param[in]   mapY_back_file 后鱼眼映射y坐标表.
    * @param[in]   frontmask_file 前鱼眼mask文件.
    * @param[in]   backmask_file 后鱼眼mask文件.
    * @param[in]   outputSize 全景图输出大小，[width,height]，单位：像素.
    * @return      无
    * @retval      void
    * @par 标识符
    *     保留
    * @par 其它
    *
    * @par 修改日志
    *      cuixingxing于2024/07/11创建
    */
    DualFisheyeSticher(std::string mapX_front_file, std::string mapY_front_file, std::string mapX_back_file, std::string mapY_back_file, std::string frontmask_file, std::string backmask_file, cv::Size outputSize = cv::Size(5760, 2880));

    ~DualFisheyeSticher();

    /**
    * @brief       对双鱼眼图像进行全景拼接
    * @details     This is the detail description.
    * @param[in]   frontFisheye 单个环形前视鱼眼图像.
    * @param[out]  backFisheye 单个环形前后鱼眼图像.
    * @return      等距全景图
    * @retval      cv::Mat
    * @par 标识符
    *     保留
    * @par 其它
    *
    * @par 修改日志
    *      cuixingxing于2024/07/09创建
    */
    cv::Mat stich(cv::Mat& frontFisheye, cv::Mat& backFisheye);

   private:
    cv::Mat m_mapX_front;
    cv::Mat m_mapY_front;
    cv::Mat m_mapX_back;
    cv::Mat m_mapY_back;
    cv::Mat m_front_mask;         // CV32FC1,范围在[0,1]之间
    cv::Mat m_back_mask;          // CV32FC1,范围在[0,1]之间
    std::vector<int> m_edgesIdx;  // 过渡融合带区域横坐标，size为4

    /**
    * @brief       读取pgm映射文件
    * @details     按照固定格式读取里面映射数组
    * @param[in]   pgmFile 后缀为pgm文件，同样支持ffmpeg使用
    * @return      返回映射坐标二维矩阵
    * @retval      cv::Mat
    * @par 标识符
    *     保留
    * @par 其它
    *
    * @par 修改日志
    *      cuixingxing于2024/07/10创建
    */
    cv::Mat readPGMFile(std::string pgmFile);

    /**
    * @brief       逐个元素相乘
    * @details     用于mask图像与bgr图像逐个相乘，替代opencv mul内置函数，提高效率
    * @param[in]   bgr 三通道图像,CV_8UC3类型
    * @param[out]  mask 单通道CV_32FC1,[0,1]范围图像
    * @return      返回值，与bgr同大小，类型的输出
    * @retval      返回值类型
    * @par 标识符
    *     保留
    * @par 其它
    *
    * @par 修改日志
    *      cuixingxing于2024/07/10创建
    */
    cv::Mat elementMultiply(cv::Mat& bgr, cv::Mat& mask);

    /**
    * @brief       单个环形鱼眼图像转360度equirectangular panorama
    * @details     通过指定该环形鱼眼镜头的水平，竖直视场角hFOV,vFOV,畸变中心像素坐标(cx,cy)，畸变半径radius，输出具有指定的宽高，不同角度的全景图像
    * @param[in]   fisheyeImage 输入单个环形鱼眼图像.
    * @param[out]  panoImage 输出全景图像，equirectangular panorama.
    * @param[out]  mapX 输出x映射坐标表
    * @param[out]  mapY 输出y映射坐标表
    * @param[in]   hFOV  输入的水平视场角，单位：度
    * @param[in]   vFOV 输入的竖直视场角，单位：度
    * @param[in]   cx 畸变中心x坐标,单位：像素
    * @param[in]   cy 畸变中心y坐标,单位：像素
    * @param[in]   radius 畸变半径，单位：像素
    * @param[in]   width 输出图像panoImage的宽度，单位：像素
    * @param[in]   height 输出图像panoImage的高度，单位：像素
    * @param[in]   x 俯仰角，单位：度
    * @param[in]   y 翻滚角，单位：度
    * @param[in]   z 方位角，单位：度,相机光轴默认朝向y轴正方向
    * @return      无
    * @retval      void
    * @par 标识符
    *     保留
    * @par 其它
    *
    * @par 修改日志
    *      cuixingxing于2024/07/11创建
    */
    void fish2sphere(cv::Mat& fisheyeImage, cv::Mat& panoImage, cv::Mat& mapX, cv::Mat& mapY, float hFOV, float vFOV, float cx = -1.0, float cy = -1.0, float radius = -1.0, int width = 5760, int height = 2880, float xDeg = 0.0, float yDeg = 0.0, float zDeg = 0.0);

    double interp1(const std::vector<double>& x, const std::vector<double>& v, double xi);

    // 从m_front_mask图像中推理blend区域的横坐标
    void getEdgeIdxs(cv::Size outputSize);
};

}  // namespace panorama
