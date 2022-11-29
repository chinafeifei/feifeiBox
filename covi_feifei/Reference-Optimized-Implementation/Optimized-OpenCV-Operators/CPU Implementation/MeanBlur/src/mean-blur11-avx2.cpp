//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
// #include <windows.h>
#include <memory.h>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>
#include <ittnotify.h>
#include "iw++/iw.hpp"
#include "ippi.h"
#if !(defined(_MSC_VER))
#include <immintrin.h>
#include <avxintrin.h>
#endif
#pragma warning(disable : 4700)

using timeunit = std::chrono::microseconds;
#ifdef _WIN32
// Create a domain that is visible globally: we will use it in our example.
__itt_domain *domain = __itt_domain_create((const wchar_t *)("Example.Domain.Global"));
// Create string handles which associates with the "main" task.
__itt_string_handle *handle_cv = __itt_string_handle_create((const wchar_t *)"cv");
__itt_string_handle *handle_ipp = __itt_string_handle_create((const wchar_t *)"ipp");
__itt_string_handle *handle_isd = __itt_string_handle_create((const wchar_t *)"isd");
__itt_string_handle *handle_for_loop = __itt_string_handle_create((const wchar_t *)"for_loop");
#else
// Create a domain that is visible globally: we will use it in our example.
__itt_domain *domain = __itt_domain_create((const char *)("Example.Domain.Global"));
// Create string handles which associates with the "main" task.
__itt_string_handle *handle_cv = __itt_string_handle_create((const char *)"cv");
__itt_string_handle *handle_ipp = __itt_string_handle_create((const char *)"ipp");
__itt_string_handle *handle_isd = __itt_string_handle_create((const char *)"isd");
__itt_string_handle *handle_for_loop = __itt_string_handle_create((const char *)"for_loop");
#endif
static IW_INLINE IwSize owniAlignStep(IwSize step, int align)
{
    return (step + (align - 1)) & -align;
}

void compare(Ipp8u pSrc[], cv::Mat &smoothed, cv::Mat &img)
{
    Ipp8u *pDst = smoothed.data;
    Ipp8u *pImg = img.data;
    int cv_step = smoothed.cols * smoothed.channels();
    int src_step = owniAlignStep(cv_step, 64);

    int same = 0;
    int diff = 0;
    for (int x = 0; x < smoothed.rows - 10; x++)
    {
        for (int y = 0; y < smoothed.cols * smoothed.channels(); y++)
        {
            if (abs(pSrc[x * src_step + y] - pDst[x * cv_step + y]) <= 1)
                same++;
            else
            {
                diff++;
            }
        }
    }

    float fsame = same;
    float fall = same + diff;

    float correctness = fsame / fall * 100;

    printf("pixels correctness is %2.2f %% \n", correctness);
}

static void x11CommonLoop_8u_C1R(Ipp32f *__restrict pBuffer, Ipp8u *__restrict dst, int width, Ipp32f scl)
{
#if 0 // OpenCV
    Ipp32f s = 0;
    Ipp32f* pbuf = (Ipp32f*)pBuffer;
    Ipp8u dst_tmp[32];
    for (int i = 0; i < xMaskSize; i++)
        s += (Ipp32f)pbuf[i];
    dst_tmp[0] = (round)(s / 121);
    // dst[0] = s / 121;
    //for (int i = 0; i < width; i++)
    for (int i = 0; i < 63; i++)
    {
        s += (Ipp32f)pbuf[i + xMaskSize] - (Ipp32f)pbuf[i];
        // dst[i + 1] = s / 121;
        dst_tmp[i + 1] = (round)(s / 121);
    }
#endif

    Ipp32f *pbuf = pBuffer;
    int xMaskSize = 11;
    __m256 sclS = _mm256_set1_ps(scl); // Ipp32f scl = 1.f / (Ipp32f)(maskSize.height * maskSize.width);
    Ipp32f sum = 0;
    Ipp32f e[8];
    int width32 = (width >> 5) << 5;
    for (int i = 0; i < xMaskSize; i++)
        sum += pbuf[i];
    int w = 0;
    int N = 0;
    for (w = 0; w < width32; w += 32)
    {
        __m256i ti;
        __m256i tj;
        __m256 sum256;
        e[0] = sum;
        N = w;
        for (int k = 1; k < 8; k++)
        {
            sum += pbuf[xMaskSize + N] - pbuf[N];
            e[k] = sum;
            N += 1;
        }
        sum256 = _mm256_set_ps(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
        sum256 = _mm256_mul_ps(sum256, sclS);
        ti = _mm256_cvtps_epi32(sum256);
        sum += pbuf[xMaskSize + N] - pbuf[N];
        N += 1;

        e[0] = sum;
        for (int k = 1; k < 8; k++)
        {
            sum += pbuf[xMaskSize + N] - pbuf[N];
            N += 1;
            e[k] = sum;
        }
        sum256 = _mm256_set_ps(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
        sum256 = _mm256_mul_ps(sum256, sclS);
        tj = _mm256_cvtps_epi32(sum256);
        ti = _mm256_packs_epi32(ti, tj);
        sum += pbuf[xMaskSize + N] - pbuf[N];
        N += 1;

        e[0] = sum;
        for (int k = 1; k < 8; k++)
        {
            sum += pbuf[xMaskSize + N] - pbuf[N];
            N += 1;
            e[k] = sum;
        }
        sum256 = _mm256_set_ps(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
        sum256 = _mm256_mul_ps(sum256, sclS);
        tj = _mm256_cvtps_epi32(sum256);
        sum += pbuf[xMaskSize + N] - pbuf[N];
        N += 1;

        e[0] = sum;
        for (int k = 1; k < 8; k++)
        {
            sum += pbuf[xMaskSize + N] - pbuf[N];
            N += 1;
            e[k] = sum;
        }

        sum256 = _mm256_set_ps(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
        sum256 = _mm256_mul_ps(sum256, sclS);
        tj = _mm256_packs_epi32(tj, _mm256_cvtps_epi32(sum256));
        tj = _mm256_packus_epi16(ti, tj);
        tj = _mm256_permutevar8x32_epi32(tj, _mm256_set_epi32(
                                                 7, 3, 6, 2,
                                                 5, 1, 4, 0));
        //_mm256_storeu_epi8(dst, tj);
        _mm256_storeu_si256((__m256i *)dst, tj);
        sum += pbuf[xMaskSize + N] - pbuf[N];
        dst += 32;
    }
}

static void x11GetLastRow_8u_C1R(const Ipp8u *__restrict pSrc1, const Ipp8u *__restrict pSrc2, Ipp32f *__restrict pDst, int size)
{
    int widthS = size;
    int widthS16 = (widthS >> 4) << 4;
    const Ipp8u *rPtr1 = pSrc1;
    const Ipp8u *rPtr2 = pSrc2;
    Ipp32f *pbuf = pDst;
    int i = 0;
    for (i = 0; i < widthS16; i += 16)
    {
        // int sumC;
        __m128i mmx0;
        __m128i mPtr1, mPtr0;
        __m128 mse, t;
        // mse   = *(__m128*)pbuf;
        mse = _mm_loadu_ps(pbuf);
        mPtr0 = _mm_loadu_si128((__m128i *)rPtr1); // 0 - 15
        mPtr1 = _mm_loadu_si128((__m128i *)rPtr2);
        rPtr1 += 16, rPtr2 += 16;
        mmx0 = _mm_sub_epi32(_mm_cvtepu8_epi32(mPtr1), _mm_cvtepu8_epi32(mPtr0));
        t = _mm_cvtepi32_ps(mmx0);
        mse = _mm_add_ps(mse, t);
        //*(__m128*)pbuf = mse;
        _mm_storeu_ps(pbuf, mse);
        pbuf += 4;
        // mse   = *(__m128*)pbuf;
        mse = _mm_loadu_ps(pbuf);
        mmx0 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(mPtr1, 4)), _mm_cvtepu8_epi32(_mm_srli_si128(mPtr0, 4)));
        t = _mm_cvtepi32_ps(mmx0);
        mse = _mm_add_ps(mse, t);
        //*(__m128*)pbuf = mse;
        _mm_storeu_ps(pbuf, mse);
        pbuf += 4;
        // mse   = *(__m128*)pbuf;
        mse = _mm_loadu_ps(pbuf);
        mmx0 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(mPtr1, 8)), _mm_cvtepu8_epi32(_mm_srli_si128(mPtr0, 8)));
        t = _mm_cvtepi32_ps(mmx0);
        mse = _mm_add_ps(mse, t);
        //*(__m128*)pbuf = mse;
        _mm_storeu_ps(pbuf, mse);
        pbuf += 4;
        // mse   = *(__m128*)pbuf;
        mse = _mm_loadu_ps(pbuf);
        mmx0 = _mm_sub_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(mPtr1, 12)), _mm_cvtepu8_epi32(_mm_srli_si128(mPtr0, 12)));
        t = _mm_cvtepi32_ps(mmx0);
        mse = _mm_add_ps(mse, t);
        //*(__m128*)pbuf = mse;
        _mm_storeu_ps(pbuf, mse);
        pbuf += 4;
    }
    for (i = i; i < widthS; i++, rPtr1++, rPtr2++)
    {
        *pbuf++ += *rPtr2 - *rPtr1;
    }
}

static void x11GetFirstSum_8u_C1R(const Ipp8u *pSrc, Ipp64s srcStep, IppiSize dstRoiSize, Ipp32f *pBuffer)
{
    int h, w;
    const Ipp8u *rPtr1, *rPtr2;
    int width = dstRoiSize.width;
    int xMaskSize = 11;
    int yMaskSize = 11;
    int width8 = ((width + xMaskSize - 1) >> 3) << 3;
    rPtr1 = pSrc;

    for (h = 0; h < width8; h += 8)
    {
        __m256 a;
        __m256i ai;
        __m128i aii;
        rPtr2 = rPtr1;
        _mm256_storeu_ps(pBuffer, _mm256_setzero_ps());
        for (w = yMaskSize; w--;)
        {
            aii = _mm_loadl_epi64((__m128i const *)rPtr2); //? replace ??
            ai = _mm256_cvtepu8_epi32(aii);                // zero packet
            a = _mm256_cvtepi32_ps(ai);                    // convert 32 bit int
            _mm256_storeu_ps(pBuffer, _mm256_add_ps(a, _mm256_loadu_ps(pBuffer)));
            rPtr2 = (Ipp8u *)((Ipp8u *)rPtr2 + srcStep);
        }
        rPtr1 += 8;
        pBuffer += 8;
    }
#if 0	
    for (h = h; h < (width + xMaskSize - 1 * nCh); h++, rPtr1++)
    {
        rPtr2 = rPtr1;
        *pBuffer = 0.f;
        for (w = yMaskSize; w--;) {
            *pBuffer += *rPtr2;
            rPtr2 = (Ipp8u*)((Ipp8u*)rPtr2 + srcStep);
        }
        pBuffer++;
    }
    *pBuffer = 0.f;
#endif
}

void x11FilterBoxBorder_8u_C1R(const Ipp8u *pSrc, Ipp64s srcStep,
                               Ipp8u *pDst, Ipp64s dstStep, IppiSize dstRoiSize, Ipp32f *pBuffer)
{
    Ipp64s h;
    Ipp64s height = (Ipp64s)dstRoiSize.height;
    int width = dstRoiSize.width;
    const Ipp8u *rPtr2;
    int xMaskSize = 11;
    int yMaskSize = 11;
    Ipp32f scl = 1.f / (Ipp32f)(xMaskSize * yMaskSize);
    x11GetFirstSum_8u_C1R(pSrc, srcStep, dstRoiSize, pBuffer);
    for (h = 0; h < height - 11; h++)
    {
        int lasrow = (h == (height - 1)) ? 0 : 1;
        const Ipp8u *rPtr1 = (Ipp8u *)((Ipp8u *)pSrc + (Ipp64s)h * srcStep);
        Ipp8u *dst = (Ipp8u *)((Ipp8u *)pDst + (Ipp64s)(h + 5) * dstStep + 5);
        rPtr2 = (Ipp8u *)((Ipp8u *)rPtr1 + yMaskSize * srcStep);
        x11CommonLoop_8u_C1R(pBuffer, dst, width, scl);
        if (lasrow)
        {
            x11GetLastRow_8u_C1R(rPtr1, rPtr2, pBuffer, (width + xMaskSize - 1));
        }
    }
}

int main(int argc, char **argv)
{
    // Create images
#ifdef _WIN32
    HANDLE mHandle = GetCurrentProcess();
    BOOL result = SetPriorityClass(mHandle, REALTIME_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x00000001); // 0xFF，Core0
#endif
    std::string filename = "./data/color_4288.jpg";
    // LARGE_INTEGER st, en, c;
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::Mat smoothed;
    printf("Size %d %d\n", img.cols, img.rows);
    int ksize = 11;
    {
        float time_cv = 0;
        for (int i = 0; i < 10; i++)
        {
            __itt_task_begin(domain, __itt_null, __itt_null, handle_cv);
            auto cv_start = std::chrono::high_resolution_clock::now();
            cv::blur(img, smoothed, cv::Size(ksize, ksize), cv::Point(-1, -1));
            auto cv_stop = std::chrono::high_resolution_clock::now();
            auto cv_time = std::chrono::duration_cast<timeunit>(cv_stop - cv_start).count() / 1000.0;
            time_cv += cv_time;
            __itt_task_end(domain);
        }
        printf("\nCV Time consuming(kernel size %d): %f\n", ksize, time_cv / 10);

        ipp::IwiImage srcImage, cvtImage;
        srcImage.Init(ipp::IwiSize(img.cols, img.rows), ipp8u, img.channels(), NULL, &img.data[0], img.cols * img.channels());
        cvtImage.Alloc(srcImage.m_size, ipp8u, img.channels());
        float time_ipp = 0;
        for (int i = 0; i < 10; i++)
        {
            __itt_task_begin(domain, __itt_null, __itt_null, handle_ipp);
            auto ipp_start = std::chrono::high_resolution_clock::now();
            ipp::iwiFilterBox(srcImage, cvtImage, ksize);
            auto ipp_stop = std::chrono::high_resolution_clock::now();
            auto ipp_time = std::chrono::duration_cast<timeunit>(ipp_stop - ipp_start).count() / 1000.0;
            time_ipp += ipp_time;
            __itt_task_end(domain);
        }
        printf("\nIPP Time consuming(kernel size %d): %f\n", ksize, time_ipp / 10);

        ipp::IwiImage cvtImage2;
        cvtImage2.Alloc(srcImage.m_size, ipp8u, img.channels());
        IppiSize roisize = {img.cols, img.rows};
        float time_ipp2 = 0;
        IppiSize maskSize = {11, 11};
        int tmpBufferSize = 0;
        IppStatus status = ippiFilterBoxBorderGetBufferSize(roisize, maskSize, ipp8u, img.channels(), &tmpBufferSize);
        Ipp8u *pTmpBuffer = (Ipp8u *)ippMalloc_L(tmpBufferSize);

        for (int i = 0; i < 10; i++)
        {
            __itt_task_begin(domain, __itt_null, __itt_null, handle_isd);
            auto ipp_start2 = std::chrono::high_resolution_clock::now();
            x11FilterBoxBorder_8u_C1R((const Ipp8u *)(srcImage.m_ptr), img.cols * img.channels(), (Ipp8u *)(cvtImage2.m_ptr), owniAlignStep(img.cols * img.channels(), 64), roisize, (Ipp32f *)pTmpBuffer);
            auto ipp_stop2 = std::chrono::high_resolution_clock::now();
            auto ipp_time2 = std::chrono::duration_cast<timeunit>(ipp_stop2 - ipp_start2).count() / 1000.0;
            time_ipp2 += ipp_time2;
            __itt_task_end(domain);
        }
        printf("\nOwn IPP Time consuming(kernel size %d): %f\n", ksize, time_ipp2 / 10);
        compare((Ipp8u *)cvtImage2.m_ptr, smoothed, img);
    }
    return 0;
}