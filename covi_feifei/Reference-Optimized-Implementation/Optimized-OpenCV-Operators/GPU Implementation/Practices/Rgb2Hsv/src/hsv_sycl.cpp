//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iomanip>

#include <opencv2/imgproc.hpp>

#include <CL/sycl.hpp>

#include <chrono>

using namespace std;
using namespace cv;
// using namespace sycl;

const int c_HsvDivTable[256] = {1, 1044481, 522241, 348161, 261121, 208897, 174081, 149212, 130561, 116054, 104449, 94954, 87041, 80346, 74607, 69633, 65281, 61441, 58028, 54974, 52225, 49738, 47477, 45413, 43521, 41780, 40173, 38685, 37304, 36018, 34817, 33694, 32641, 31652, 30721, 29843, 29014, 28230, 27487, 26783, 26113, 25476, 24870, 24291, 23739, 23212, 22707, 22224, 21761, 21317, 20891, 20481, 20087, 19708, 19343, 18992, 18652, 18325, 18009, 17704, 17409, 17124, 16847, 16580, 16321, 16070, 15826, 15590, 15361, 15138, 14922, 14712, 14508, 14309, 14116, 13927, 13744, 13566, 13392, 13222, 13057, 12896, 12739, 12585, 12435, 12289, 12146, 12007, 11870, 11737, 11606, 11479, 11354, 11232, 11112, 10996, 10881, 10769, 10659, 10551, 10446, 10342, 10241, 10142, 10044, 9948, 9855, 9762, 9672, 9583, 9496, 9411, 9327, 9244, 9163, 9083, 9005, 8928, 8853, 8778, 8705, 8633, 8562, 8493, 8424, 8357, 8291, 8225, 8161, 8098, 8035, 7974, 7914, 7854, 7796, 7738, 7681, 7625, 7570, 7515, 7462, 7409, 7356, 7305, 7254, 7204, 7155, 7106, 7058, 7011, 6964, 6918, 6873, 6828, 6783, 6740, 6696, 6654, 6612, 6570, 6529, 6488, 6448, 6409, 6370, 6331, 6293, 6255, 6218, 6181, 6145, 6109, 6074, 6038, 6004, 5969, 5936, 5902, 5869, 5836, 5804, 5772, 5740, 5709, 5678, 5647, 5616, 5586, 5557, 5527, 5498, 5469, 5441, 5413, 5385, 5357, 5330, 5303, 5276, 5250, 5223, 5197, 5172, 5146, 5121, 5096, 5071, 5047, 5023, 4999, 4975, 4951, 4928, 4905, 4882, 4859, 4837, 4814, 4792, 4770, 4749, 4727, 4706, 4685, 4664, 4643, 4623, 4602, 4582, 4562, 4542, 4523, 4503, 4484, 4465, 4446, 4427, 4408, 4390, 4371, 4353, 4335, 4317, 4299, 4282, 4264, 4247, 4230, 4213, 4196, 4179, 4162, 4146, 4129, 4113, 4097};
const int c_HsvDivTable180[256] = {0, 122880, 61440, 40960, 30720, 24576, 20480, 17554, 15360, 13653, 12288, 11171, 10240, 9452, 8777, 8192, 7680, 7228, 6827, 6467, 6144, 5851, 5585, 5343, 5120, 4915, 4726, 4551, 4389, 4237, 4096, 3964, 3840, 3724, 3614, 3511, 3413, 3321, 3234, 3151, 3072, 2997, 2926, 2858, 2793, 2731, 2671, 2614, 2560, 2508, 2458, 2409, 2363, 2318, 2276, 2234, 2194, 2156, 2119, 2083, 2048, 2014, 1982, 1950, 1920, 1890, 1862, 1834, 1807, 1781, 1755, 1731, 1707, 1683, 1661, 1638, 1617, 1596, 1575, 1555, 1536, 1517, 1499, 1480, 1463, 1446, 1429, 1412, 1396, 1381, 1365, 1350, 1336, 1321, 1307, 1293, 1280, 1267, 1254, 1241, 1229, 1217, 1205, 1193, 1182, 1170, 1159, 1148, 1138, 1127, 1117, 1107, 1097, 1087, 1078, 1069, 1059, 1050, 1041, 1033, 1024, 1016, 1007, 999, 991, 983, 975, 968, 960, 953, 945, 938, 931, 924, 917, 910, 904, 897, 890, 884, 878, 871, 865, 859, 853, 847, 842, 836, 830, 825, 819, 814, 808, 803, 798, 793, 788, 783, 778, 773, 768, 763, 759, 754, 749, 745, 740, 736, 731, 727, 723, 719, 714, 710, 706, 702, 698, 694, 690, 686, 683, 679, 675, 671, 668, 664, 661, 657, 654, 650, 647, 643, 640, 637, 633, 630, 627, 624, 621, 617, 614, 611, 608, 605, 602, 599, 597, 594, 591, 588, 585, 582, 580, 577, 574, 572, 569, 566, 564, 561, 559, 556, 554, 551, 549, 546, 544, 541, 539, 537, 534, 532, 530, 527, 525, 523, 521, 518, 516, 514, 512, 510, 508, 506, 504, 502, 500, 497, 495, 493, 492, 490, 488, 486, 484, 482};

void ShowDevice(sycl::queue &q)
{
    // Output platform and device information.
    auto device = q.get_device();
    auto p_name = device.get_platform().get_info<sycl::info::platform::name>();
    cout << std::setw(20) << "Platform Name: " << p_name << "\n";
    auto p_version = device.get_platform().get_info<sycl::info::platform::version>();
    cout << std::setw(20) << "Platform Version: " << p_version << "\n";
    auto d_name = device.get_info<sycl::info::device::name>();
    cout << std::setw(20) << "Device Name: " << d_name << "\n";
    auto max_work_group = device.get_info<sycl::info::device::max_work_group_size>();
    cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
    auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
    cout << std::setw(20) << "Max Compute Units: " << max_compute_units << "\n\n";
}

void display_array(uchar *array, int size)
{
    printf("array is: ");
    for (int i = 0; i < size; i++)
    {

        printf("%3d, ", array[i]);
    }
    printf("\n");
}

void assign_BGR(const Mat &cvmat, uchar *inB, uchar *inG, uchar *inR)
{
    int total = cvmat.rows * cvmat.cols;
    for (int i = 0; i < total; i++)
    {
        inB[i] = cvmat.data[i * 3];     // 绿色分量
        inG[i] = cvmat.data[i * 3 + 1]; // 蓝色分量
        inR[i] = cvmat.data[i * 3 + 2]; // 红色分量
    }
}

void display_array(const Mat &cvmat)
{

    if (cvmat.isContinuous())
    {
        unsigned char *array = cvmat.data;

        int total = cvmat.rows * cvmat.cols * 3;

        // for (int i = 0; i < total; i += 3)
        for (int i = total - 30; i < total; i += 3)
        {
            printf("[%3d, %3d, %3d], ", array[i], array[i + 1], array[i + 2]);
            if ((i / 3 + 1) % cvmat.cols == 0)
            {
                cout << std::endl;
            }
        }
    }

    cout << std::endl;
}
/**
 * 从 opencv cuda 版本拷贝过来的 RGB2HSV 实现
 */
void cvt(uchar blue, uchar green, uchar red)
{
    const int hsv_shift = 12;
    const int hr = 180;
    const int *hdiv_table = c_HsvDivTable180;

    const int b = blue;
    const int g = green;
    const int r = red;

    int h, s, v = b;
    int vmin = b, diff;
    int vr, vg;

    v = std::max(v, g);
    v = std::max(v, r);
    vmin = std::min(vmin, g);
    vmin = std::min(vmin, r);

    diff = v - vmin;
    vr = (v == r) * -1;
    vg = (v == g) * -1;

    s = (diff * c_HsvDivTable[v] - diff + (1 << (hsv_shift - 1))) >> hsv_shift;
    h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h = (h * hdiv_table[diff] + (1 << (hsv_shift - 1))) >> hsv_shift;
    h += (h < 0) * hr;

    printf("BGR [%3d,%3d,%3d] ==> HSV [%3d,%3d,%3d] \n",
           blue, green, red,
           h, s, v);
}

int main()
{

    std::string filename = "./data/color_4288.jpg";

    // std::string filename = "./data/starry_night.png";

    Mat img = imread(filename);
    if (img.empty())
    {
        std::cout << "Could not read the image: " << filename << std::endl;
        return 1;
    }

    // cout << "img (python) = \n"
    //      << format(img, Formatter::FMT_PYTHON) << ";" << std::endl
    //      << std::endl;

    cout << "img rows is " << img.rows << std::endl;
    cout << "img cols is " << img.cols << std::endl;
    cout << "img step is " << img.step << std::endl;

    int total_pixel = img.rows * img.cols;

    // total_pixel = 3686400;
    // total_pixel = 2000000;

    printf("total pixel is %d \n", total_pixel);

    // uchar inB[total_pixel];
    // uchar inG[total_pixel];
    // uchar inR[total_pixel];

    uchar *inB = new uchar[total_pixel]; // 将 opencv 读取到的图片分别读入到 RGB 三个分量数组中
    uchar *inG = new uchar[total_pixel];
    uchar *inR = new uchar[total_pixel];

    // uchar outH[total_pixel];
    // uchar outS[total_pixel];
    // uchar outV[total_pixel];

    uchar *outH = new uchar[total_pixel]; // sycl 每一个核函数分别将结果写入 HSV 三个分量数组中
    uchar *outS = new uchar[total_pixel];
    uchar *outV = new uchar[total_pixel];

    for (int i = 0; i < total_pixel; i++)
    {
        inB[i] = 0;
        inG[i] = 0;
        inR[i] = 0;
        outH[i] = 0;
        outS[i] = 0;
        outV[i] = 0;
    }

    assign_BGR(img, inB, inG, inR);

    // display_array(inG, total_pixel);

    // display_array(img);

    cout << std::endl;

    cv::Mat hsv;
    // const int64 start = cv::getTickCount();

    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    cv::cvtColor(img, hsv, COLOR_BGR2HSV);
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "cvtColor() took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " milliseconds\n";

    // cv::cvtColor(img, hsv, COLOR_BGR2RGB);

    // display_array(hsv);

    // imshow("My Window", hsv);
    // waitKey();

    cvt(142, 226, 214);
    cvt(140, 225, 217);
    cvt(136, 222, 216);
    cvt(125, 211, 203);
    cvt(131, 221, 208);

    sycl::queue Q;

    ShowDevice(Q);

    // 在这个定义域内创建的sycl buffer 会在程序走出定义域外以后，同步设备内存，所以加了定义域就无需 wait 等待
    {
        sycl::buffer<uchar, 1> inG_buf(inG, sycl::range<1>(total_pixel));
        sycl::buffer<uchar, 1> inB_buf(inB, sycl::range<1>(total_pixel));
        sycl::buffer<uchar, 1> inR_buf(inR, sycl::range<1>(total_pixel));

        sycl::buffer<uchar, 1> outH_buf(outH, sycl::range<1>(total_pixel));
        sycl::buffer<uchar, 1> outS_buf(outS, sycl::range<1>(total_pixel));
        sycl::buffer<uchar, 1> outV_buf(outV, sycl::range<1>(total_pixel));

        sycl::buffer<int, 1> c_HsvDivTable_buf(c_HsvDivTable, sycl::range<1>(256));
        sycl::buffer<int, 1> c_HsvDivTable180_buf(c_HsvDivTable180, sycl::range<1>(256));

        Q.submit([&](sycl::handler &h)
                 {
                     sycl::accessor accG(inG_buf, h, sycl::read_only);
                     sycl::accessor accB(inB_buf, h, sycl::read_only);
                     sycl::accessor accR(inR_buf, h, sycl::read_only);
                     sycl::accessor acc_HsvDivTable(c_HsvDivTable_buf, h, sycl::read_only);
                     sycl::accessor acc_HsvDivTable180(c_HsvDivTable180_buf, h, sycl::read_only);

                     sycl::accessor accH(outH_buf, h, sycl::write_only);
                     sycl::accessor accS(outS_buf, h, sycl::write_only);
                     sycl::accessor accV(outV_buf, h, sycl::write_only);

                     //  h.parallel_for(250, [=](sycl::id<1> idx) {
                     //      accH[idx] = 0;
                     //  });

                     sycl::stream out(1024 * 20, 256, h);

                     h.parallel_for(sycl::range<1>(total_pixel), [=](sycl::id<1> i)
                                    {
                                        const int hsv_shift = 12;
                                        const int hr = 180;
                                        // const int *hdiv_table = c_HsvDivTable180;

                                        const int b = accB[i];
                                        const int g = accG[i];
                                        const int r = accR[i];

                                        int h, s, v = b;
                                        int vmin = b, diff;
                                        int vr, vg;

                                        v = std::max(v, g);
                                        v = std::max(v, r);
                                        vmin = std::min(vmin, g);
                                        vmin = std::min(vmin, r);

                                        diff = v - vmin;
                                        vr = (v == r) * -1;
                                        vg = (v == g) * -1;

                                        s = (diff * acc_HsvDivTable[v] - diff + (1 << (hsv_shift - 1))) >> hsv_shift;
                                        h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
                                        h = (h * acc_HsvDivTable180[diff] + (1 << (hsv_shift - 1))) >> hsv_shift;
                                        h += (h < 0) * hr;

                                        // if (i.get(0) > 600)
                                        // {
                                        //     out << "idx: " << i.get(0) << " ";
                                        //     out << "BGR : " << b << " " << g << " " << r;
                                        //     out << "==> HSV: " << h << " " << s << " " << v << sycl::endl;
                                        // }

                                        accH[i] = h;
                                        accS[i] = s;
                                        accV[i] = v;
                                    }); });
    }

    // display_array(outV, total_pixel);

    // uchar allHSV[total_pixel * 3];

    uchar *allHSV = new uchar[total_pixel * 3];

    for (int i = 0; i < total_pixel; i++)
    {
        allHSV[i * 3] = outH[i];
        allHSV[i * 3 + 1] = outS[i];
        allHSV[i * 3 + 2] = outV[i];
    }

    bool isSame = true;

    for (int i = 0; i < total_pixel * 3; i++)
    {
        if (hsv.data[i] != allHSV[i])
        {
            isSame = false;
            break;
        }
    }

    if (isSame)
    {
        printf("Pass\n");
    }
    else
    {
        printf("Fail\n");
    }

    delete[] inB;
    delete[] inG;
    delete[] inR;

    delete[] outH;
    delete[] outS;
    delete[] outV;

    delete[] allHSV;
    return 0;
}
