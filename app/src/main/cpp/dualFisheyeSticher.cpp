#include "dualFisheyeSticher.hpp"

namespace panorama {
DualFisheyeSticher::DualFisheyeSticher(cameraParam& frontCam, cameraParam& backCam, cv::Size outputSize) {
    if (outputSize.width % 2 != 0 || outputSize.height % 2 != 0 || outputSize.width * 1.0 / outputSize.height != 2) {
        CV_Error_(cv::Error::StsBadArg,
                  ("Please specify the value of outputSize correctly, the width and height must be even and the width is twice the height!"));
    }

    // Load your images
    cv::Mat frontFisheyeImage = frontCam.circleFisheyeImage;
    cv::Mat backFisheyeImage = backCam.circleFisheyeImage;
    cv::Mat frontImage360, backImage360;

#ifdef PROFILING
    double t1 = static_cast<double>(cv::getTickCount());
#endif
    // 初始化m_mapX_front, m_mapY_front, m_mapX_back, m_mapY_back
    fish2sphere(frontFisheyeImage, frontImage360, m_mapX_front, m_mapY_front,
                frontCam.FOV, frontCam.FOV,
                frontCam.centerPt.x, frontCam.centerPt.y, frontCam.radius,
                outputSize.width, outputSize.height, frontCam.rotateX, frontCam.rotateY, frontCam.rotateZ);
    fish2sphere(backFisheyeImage, backImage360, m_mapX_back, m_mapY_back,
                backCam.FOV, backCam.FOV,
                backCam.centerPt.x, backCam.centerPt.y, backCam.radius,
                outputSize.width, outputSize.height, backCam.rotateX, backCam.rotateY, backCam.rotateZ);
    cv::convertMaps(m_mapX_front, m_mapY_front, m_mapX_front, m_mapY_front, CV_16SC2);
    cv::convertMaps(m_mapX_back, m_mapY_back, m_mapX_back, m_mapY_back, CV_16SC2);

#ifdef PROFILING
    double t2 = static_cast<double>(cv::getTickCount());
    double t = (t2 - t1) / cv::getTickFrequency();
    std::cout << "fish2sphere elapsed time " << t << " seconds." << std::endl;
#endif

#ifdef MY_DEBUG
    cv::Mat showfront, showback;
    cv::Size shows = cv::Size(800, 400);
    cv::resize(frontImage360, showfront, shows);
    cv::resize(backImage360, showback, shows);

    cv::imshow("frontImage360", showfront);
    cv::imshow("backImage360", showback);
#endif

#ifdef PROFILING
    t1 = static_cast<double>(cv::getTickCount());
#endif

    double camSt1_rotateZ = frontCam.rotateZ;
    double camSt2_rotateZ = backCam.rotateZ;
    double camSt1_FOV = frontCam.FOV;
    double camSt2_FOV = backCam.FOV;

    double frontImage360Base = normalizeYaw(90 + camSt1_rotateZ);
    double backImage360Base = normalizeYaw(90 + camSt2_rotateZ);

    double epsilon = 30;
    assert(frontImage360Base > 90 - epsilon && frontImage360Base < 90 + epsilon);
    assert(backImage360Base > 270 - epsilon || backImage360Base < -90 + epsilon);

    // Mask generation
    // 以下lonNodes每个数值应当分别保证在[270],(270,180),(180,90),(90,0),(0,-90),[-90]范围
    std::vector<double> lonNodes = {
        270,
        frontImage360Base + camSt1_FOV / 2,
        backImage360Base - camSt2_FOV / 2,
        backImage360Base + camSt2_FOV / 2,
        frontImage360Base - camSt1_FOV / 2,
        -90};
    for (size_t i = 1; i < lonNodes.size() - 1; i++) {
        lonNodes[i] = normalizeYaw(lonNodes[i]);
    }

    if (lonNodes[1] < lonNodes[2] || lonNodes[3] < lonNodes[4]) {
        CV_Error_(cv::Error::StsBadArg,
                  ("Please adjust the 2 fisheye FOV and rotateZ correctly to make sure there is an intersection area."));
    }

#ifdef PROFILING
    t2 = static_cast<double>(cv::getTickCount());
    t = (t2 - t1) / cv::getTickFrequency();
    std::cout << "lonNodes generation elapsed time " << t << " seconds." << std::endl;
#endif

#ifdef PROFILING
    t1 = static_cast<double>(cv::getTickCount());
#endif
    int w = outputSize.width;
    int h = outputSize.height;

    std::vector<double> x = {
        (lonNodes[0] - lonNodes[0]) / 360.0,
        (lonNodes[0] - lonNodes[1]) / 360.0,
        (lonNodes[0] - lonNodes[2]) / 360.0,
        (lonNodes[0] - lonNodes[3]) / 360.0,
        (lonNodes[0] - lonNodes[4]) / 360.0,
        (lonNodes[0] - lonNodes[5]) / 360.0};

    std::vector<double> xq(w);
    for (int i = 0; i < w; ++i) {
        xq[i] = static_cast<double>(i) / w;
    }

    std::vector<double> v = {0, 0, 1, 1, 0, 0};
    std::vector<double> v2 = {1, 1, 0, 0, 1, 1};

    cv::Mat frontMask(h, w, CV_32FC1);
    cv::Mat backMask(h, w, CV_32FC1);

    for (int i = 0; i < w; ++i) {
        double xi = xq[i];
        frontMask.col(i) = cv::Mat::ones(h, 1, CV_32FC1) * interp1(x, v, xi);
        backMask.col(i) = cv::Mat::ones(h, 1, CV_32FC1) * interp1(x, v2, xi);
    }

    // 初始化m_front_mask，m_back_mask
    m_front_mask = frontMask.clone();
    m_back_mask = backMask.clone();

#ifdef PROFILING
    t2 = static_cast<double>(cv::getTickCount());
    t = (t2 - t1) / cv::getTickFrequency();
    std::cout << "mask generation elapsed time " << t << " seconds." << std::endl;
#endif

    // 初始化m_edgeIdxs
    for (size_t i = 1; i < 5; i++) {
        m_edgesIdx.push_back(static_cast<int>(x[i] * w));
    }

#ifdef MY_DEBUG
    cv::Mat panorama = elementMultiply(frontImage360, m_front_mask) + elementMultiply(backImage360, m_back_mask);
    cv::Mat showpano;
    cv::resize(panorama, showpano, shows);
    cv::imshow("Panorama", showpano);
    cv::waitKey(0);
#endif
}

DualFisheyeSticher::DualFisheyeSticher(std::string mapX_front_file, std::string mapY_front_file, std::string mapX_back_file, std::string mapY_back_file, std::string frontmask_file, std::string backmask_file, cv::Size outputSize) {
    if (outputSize.width % 2 != 0 || outputSize.height % 2 != 0 || outputSize.width * 1.0 / outputSize.height != 2) {
        CV_Error_(cv::Error::StsBadArg,
                  ("Please specify the value of outputSize correctly, the width and height must be even and the width is twice the height!"));
    }

    m_mapX_front = readPGMFile(mapX_front_file);
    m_mapY_front = readPGMFile(mapY_front_file);
    m_mapX_back = readPGMFile(mapX_back_file);
    m_mapY_back = readPGMFile(mapY_back_file);
    m_front_mask = cv::imread(frontmask_file, cv::IMREAD_UNCHANGED);  // 单通道 uint8图像
    m_back_mask = cv::imread(backmask_file, cv::IMREAD_UNCHANGED);    // 单通道 uint8图像
    getEdgeIdxs(outputSize);

    m_front_mask.convertTo(m_front_mask, CV_32FC1, 1.0 / 255);
    m_back_mask.convertTo(m_back_mask, CV_32FC1, 1.0 / 255);

    cv::resize(m_mapX_front, m_mapX_front, outputSize);
    cv::resize(m_mapY_front, m_mapY_front, outputSize);
    cv::resize(m_mapX_back, m_mapX_back, outputSize);
    cv::resize(m_mapY_back, m_mapY_back, outputSize);
    cv::convertMaps(m_mapX_front, m_mapY_front, m_mapX_front, m_mapY_front, CV_16SC2);
    cv::convertMaps(m_mapX_back, m_mapY_back, m_mapX_back, m_mapY_back, CV_16SC2);

    cv::resize(m_front_mask, m_front_mask, outputSize);
    cv::resize(m_back_mask, m_back_mask, outputSize);
}

void DualFisheyeSticher::getEdgeIdxs(cv::Size outputSize) {
    cv::Mat rowM = m_front_mask.row(0) < 255 & m_front_mask.row(0) > 0;
    uchar* p = rowM.ptr<uchar>(0);
    for (int i = 0; i < rowM.cols - 1; i++) {
        if (cv::abs(p[i + 1] - p[i]) > 0) {
            float r = static_cast<float>(outputSize.width) / m_front_mask.cols;
            m_edgesIdx.push_back(static_cast<int>(i * r));
        }
    }

#ifdef MY_DEBUG
    assert(m_edgesIdx.size() == 4);
    std::cout << "rowM.size:" << rowM.size() << ",m_degesIdx:" << m_edgesIdx[0] << "," << m_edgesIdx[1] << "," << m_edgesIdx[2] << "," << m_edgesIdx[3] << std::endl;
#endif
}

cv::Mat DualFisheyeSticher::stich(cv::Mat& frontFisheye, cv::Mat& backFisheye) {
    assert(frontFisheye.type() == CV_8UC3 && backFisheye.type() == CV_8UC3);
#ifdef PROFILING
    double t1 = static_cast<double>(cv::getTickCount());
#endif

    cv::Mat front360;
    cv::Mat back360;
    cv::remap(frontFisheye, front360, m_mapX_front, m_mapY_front, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(100));
    cv::remap(backFisheye, back360, m_mapX_back, m_mapY_back, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(100));

#ifdef PROFILING
    double t2 = static_cast<double>(cv::getTickCount());
    double t = (t2 - t1) / cv::getTickFrequency() * 1000;
    std::cout << "remap take time " << t << " ms per frame." << std::endl;
#endif

#ifdef PROFILING
    t1 = cv::getTickCount();
#endif
    cv::Rect leftS = cv::Rect(m_edgesIdx[0], 0, m_edgesIdx[1] - m_edgesIdx[0], front360.rows);
    cv::Rect rightS = cv::Rect(m_edgesIdx[2], 0, m_edgesIdx[3] - m_edgesIdx[2], front360.rows);

    cv::Mat frontCrop = front360(leftS);
    cv::Mat frontMaskCrop = m_front_mask(leftS);
    cv::Mat backCrop = back360(leftS);
    cv::Mat backMaskCrop = m_back_mask(leftS);
    cv::Mat leftBlend = elementMultiply(frontCrop, frontMaskCrop) + elementMultiply(backCrop, backMaskCrop);

#ifdef MY_DEBUG
    // cv::namedWindow("fm", cv::WINDOW_NORMAL);

    cv::imshow("fc", frontCrop);
    // cv::imwrite("frontCrop.png", frontCrop);

    cv::imshow("fm", frontMaskCrop);
    // cv::imwrite("frontMaskCrop.png", frontMaskCrop);

    cv::imshow("bc", backCrop);
    // cv::imwrite("backCrop.png", backCrop);
    cv::imshow("bm", backMaskCrop);

    cv::imshow("ml", elementMultiply(frontCrop, frontMaskCrop));
    // cv::imwrite("backMaskCrop.png", backMaskCrop);
    // cv::Mat frontMaskTemp;
    // cv::cvtColor(m_front_mask, frontMaskTemp, cv::COLOR_GRAY2BGR);
    // cv::rectangle(frontMaskTemp, leftS, cv::Scalar(0, 0, 255), 2);
    // cv::rectangle(frontMaskTemp, rightS, cv::Scalar(0, 255, 0), 2);
    // cv::imshow("frontMaskTemp", frontMaskTemp);

    // cv::Mat frontmasked = elementMultiply(front360, m_front_mask);  //m_front_mask.mul(front360);
    // cv::Mat backmasked = elementMultiply(back360, m_back_mask);     //m_back_mask.mul(back360);
    //cv::Mat panorama = frontmasked + backmasked;
#endif
    frontCrop = front360(rightS);
    frontMaskCrop = m_front_mask(rightS);
    backCrop = back360(rightS);
    backMaskCrop = m_back_mask(rightS);
    cv::Mat rightBlend = elementMultiply(frontCrop, frontMaskCrop) + elementMultiply(backCrop, backMaskCrop);

    cv::Mat left = back360(cv::Rect(0, 0, m_edgesIdx[0], front360.rows));
    cv::Mat middle = front360(cv::Rect(m_edgesIdx[1], 0, m_edgesIdx[2] - m_edgesIdx[1], front360.rows));
    cv::Mat right = back360(cv::Rect(m_edgesIdx[3], 0, front360.cols - m_edgesIdx[3], front360.rows));
    cv::Mat dst[5] = {left, leftBlend, middle, rightBlend, right};
    cv::Mat panorama;
    cv::hconcat(dst, 5, panorama);

#ifdef PROFILING
    t2 = cv::getTickCount();
    t = (t2 - t1) / cv::getTickFrequency() * 1000;
    std::cout << "multipy take time " << t << " ms per frame." << std::endl;
#endif
    return panorama;
}

double DualFisheyeSticher::interp1(const std::vector<double>& x, const std::vector<double>& v, double xi) {
    // Linear interpolation
    for (size_t i = 1; i < x.size(); ++i) {
        if (xi <= x[i]) {
            double t = (xi - x[i - 1]) / (x[i] - x[i - 1]);
            return v[i - 1] + t * (v[i] - v[i - 1]);
        }
    }
    return v.back();
}

cv::Mat DualFisheyeSticher::elementMultiply(cv::Mat& bgr, cv::Mat& mask) {
    //参考我的回答 https://stackoverflow.com/a/78733373/9939634
    assert(bgr.type() == CV_8UC3 && mask.type() == CV_32FC1 && bgr.size() == mask.size());
    int H = bgr.rows;
    int W = bgr.cols;
    cv::Mat dst(bgr.size(), bgr.type());

    for (int i = 0; i < H; ++i) {
        uchar* pdst = dst.data + i * dst.step;
        const uchar* pbgr = bgr.data + i * bgr.step;
        const float* pmask = (float*)(mask.data + i * mask.step);
        for (int j = 0; j < W; ++j) {
            pdst[0] = static_cast<uchar>(static_cast<float>(pbgr[0]) * pmask[j]);
            pdst[1] = static_cast<uchar>(static_cast<float>(pbgr[1]) * pmask[j]);
            pdst[2] = static_cast<uchar>(static_cast<float>(pbgr[2]) * pmask[j]);

            pdst += 3;
            pbgr += 3;
        }
    }
    return dst;
}

DualFisheyeSticher::~DualFisheyeSticher() {
}

void DualFisheyeSticher::fish2sphere(cv::Mat& fisheyeImage, cv::Mat& panoImage, cv::Mat& mapX, cv::Mat& mapY, float hFOV, float vFOV, float cx, float cy, float radius, int width, int height, float xDeg, float yDeg, float zDeg) {
    assert(hFOV >= 180 && vFOV >= 180);
    if (cx == -1) cx = fisheyeImage.cols / 2.0;
    if (cy == -1) cy = fisheyeImage.rows / 2.0;
    if (radius == -1) radius = std::min(fisheyeImage.rows / 2.0, fisheyeImage.cols / 2.0);

    mapX = cv::Mat(height, width, CV_32FC1);
    mapY = cv::Mat(height, width, CV_32FC1);

    float xRad = xDeg * CV_PI / 180.0;
    float yRad = yDeg * CV_PI / 180.0;
    float zRad = zDeg * CV_PI / 180.0;

    float thetaStep = 360.0 / width;
    float phiStep = 180.0 / height;

    for (int i = 0; i < height; ++i) {
        float* px = (float*)(mapX.data + i * mapX.step);
        float* py = (float*)(mapY.data + i * mapY.step);
        for (int j = 0; j < width; ++j) {
            float theta = (270 - j * thetaStep) * CV_PI / 180.0;
            float phi = (-90 + i * phiStep) * CV_PI / 180.0;

            float x = static_cast<float>(cos(phi) * cos(theta));
            float y = static_cast<float>(cos(phi) * sin(theta));
            float z = static_cast<float>(sin(phi));

            // Apply rotation matrices
            // cv::Mat xyz = (cv::Mat_<double>(3, 1) << x, y, z);
            // cv::Mat A = rotz(zDeg * CV_PI / 180.0) * roty(yDeg * CV_PI / 180.0) * rotx(xDeg * CV_PI / 180.0) * xyz;  // 'ZYX' order
            // x = A.at<double>(0, 0);
            // y = A.at<double>(1, 0);
            // z = A.at<double>(2, 0);
            //-----------------------------------------或者直接使用下列演算结果代替--------------------------------
            // 推理公式过程如下
            // syms xRad yRad zRad x y z
            // rx = @(angle)[1,0,0;
            //     0,cos(angle),-sin(angle);
            //     0,sin(angle),cos(angle)];
            // ry = @(angle)[cos(angle),0,sin(angle);
            //     0,1,0;
            //     -sin(angle),0,cos(angle)];
            // rz = @(angle)[cos(angle),-sin(angle),0;
            //     sin(angle),cos(angle),0;
            //     0,0,1];
            // A = rz(zRad)*ry(yRad)*rx(xRad)*[x;y;z];
            // x_ = A(1)
            // y_ = A(2)
            // z_ = A(3)
            float x_ = z * (sin(xRad) * sin(zRad) + cos(xRad) * cos(zRad) * sin(yRad)) - y * (cos(xRad) * sin(zRad) - cos(zRad) * sin(xRad) * sin(yRad)) + x * cos(yRad) * cos(zRad);
            float y_ = y * (cos(xRad) * cos(zRad) + sin(xRad) * sin(yRad) * sin(zRad)) - z * (cos(zRad) * sin(xRad) - cos(xRad) * sin(yRad) * sin(zRad)) + x * cos(yRad) * sin(zRad);
            float z_ = z * cos(xRad) * cos(yRad) - x * sin(yRad) + y * cos(yRad) * sin(xRad);
            x = x_;
            y = y_;
            z = z_;

            // Map to fisheye coordinates
            float longit = atan2(z, x) * 180 / CV_PI;
            float latit = atan2(sqrt(x * x + z * z), y) * 180 / CV_PI;
            float rx = 2 * radius * latit / hFOV;
            float ry = 2 * radius * latit / vFOV;

            float map_x = static_cast<float>(cx + rx * cos(longit * CV_PI / 180));
            float map_y = static_cast<float>(cy + ry * sin(longit * CV_PI / 180));

            px[j] = map_x;
            py[j] = map_y;
        }
    }

    cv::remap(fisheyeImage, panoImage, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(100));
}

cv::Mat DualFisheyeSticher::readPGMFile(std::string pgmFile) {
    // 读取pgm/csv数据为二维矩阵
    std::ifstream inFile(pgmFile, std::ios::in);
    if (!inFile) {
        CV_Error_(cv::Error::StsBadArg,
                  ("Cannot open pgm file: %s", pgmFile.c_str()));
    }

    std::string lineStr;
    while (std::getline(inFile, lineStr)) {
        trim(lineStr);
        if ((lineStr.compare(0, 1, "#") == 0) || (lineStr.compare(0, 1, "P2") == 0)) {
            continue;
        }
        // 计算包含以空格为分隔符的项数
        std::istringstream iss(lineStr);
        int itemCount = std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
        if (itemCount < 5) {
            continue;
        }
        break;
    }

    std::vector<float> values;
    std::stringstream temp1(lineStr);
    std::string single_value;
    std::vector<std::vector<float> > all_data;
    while (std::getline(temp1, single_value, ' ')) {
        if (single_value == "") {
            continue;
        }
        values.push_back(atof(single_value.c_str()));
    }
    all_data.push_back(values);
    while (std::getline(inFile, lineStr)) {
        values.clear();
        std::stringstream temp2(lineStr);
        while (std::getline(temp2, single_value, ' ')) {
            if (single_value == "") {
                continue;
            }
            values.push_back(atof(single_value.c_str()));
        }
        all_data.push_back(values);
    }

    cv::Mat vect = cv::Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_32FC1);
    for (int row = 0; row < (int)all_data.size(); row++) {
        float* p = vect.ptr<float>(row);
        for (int col = 0; col < (int)all_data[0].size(); col++) {
            p[col] = all_data[row][col];
        }
    }
    // std::cout << vect.rows << "," << vect.cols << std::endl;
    // std::cout << vect.rowRange(0, 4).colRange(0, 5) << std::endl;
    return vect;
}
}  // namespace panorama
