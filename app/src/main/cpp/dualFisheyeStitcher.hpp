/**
 * @file        :dualFisheyeStitcher.hpp
 * @brief       :用户调用接口类
 * @details     :This is the detail description.
 * @date        :2025/03/13 15:29:41
 * @author      :cuixingxing(cuixingxing150@gmail.com)
 * @version     :1.0
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

namespace panorama {

// 定义陀螺仪数据结构体
typedef struct SensorData {
    double timestamp; // 时间戳，单位：秒
    double gyro[3]; // 陀螺仪三轴角速度，单位：rad/s
    double accl[3]; // 加速度计三轴加速度，单位：m/s^2
} SennsorData;

// 前向声明实现类
class DualFisheyeStitcherImpl;

// 接口类
class DualFisheyeStitcher {
public:
    DualFisheyeStitcher(cv::Size inputSize, cv::Size outputSize = cv::Size(5760, 2880));

    ~DualFisheyeStitcher();

    // 实时拼接，每次有新的图像时调用
    cv::Mat stitch(cv::Mat& frontFisheye, cv::Mat& backFisheye);

    // 设置 AHRS 算法参数，初始化只执行一次即可
    void setAHRSConfig(float sampleRate, float gyroRange, float accelRange);

    // 添加陀螺仪数据,timestamp单位为秒，每次有新的陀螺仪数据时调用
    void addSensorData(double timestamp, const SensorData& data);

    // 获取时间戳对应的旋转矩阵，图像拼接锁定水平视角时调用
    cv::Mat getRotationMatrixAt(double imageTimestamp);

    // 带姿态校正的图像拼接，暂用于测试使用，不推荐实时应用
    cv::Mat stitchWithOrientation(cv::Mat& frontFisheye, cv::Mat& backFisheye, cv::Mat& R);

private:
    std::unique_ptr<DualFisheyeStitcherImpl> pImpl; 
};

} // namespace panorama