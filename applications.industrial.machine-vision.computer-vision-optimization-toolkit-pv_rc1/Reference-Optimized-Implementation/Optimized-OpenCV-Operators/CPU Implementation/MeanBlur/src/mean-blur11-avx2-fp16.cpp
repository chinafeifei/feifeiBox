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

static void x11CommonLoop_8u_C1R(Ipp16f *__restrict pBuf, Ipp8u *__restrict dst, int width, Ipp32f scl)
{
#if 0 // OpenCV
    Ipp16f s = 0;
    Ipp16f* pbuf = (Ipp16f*)pBuffer;
    Ipp8u dst_tmp[128];
    for (int i = 0; i < xMaskSize; i++)
        s += (Ipp32f)pbuf[i];
    dst_tmp[0] = (round)(s / 121.0);
    // dst[0] = s / 121;
    //for (int i = 0; i < width; i++)
    for (int i = 0; i < 126; i++)
    {
        s += (Ipp32f)pbuf[i + xMaskSize] - (Ipp32f)pbuf[i];
//        dst[i + 1] = (round)(s / 121);
        dst_tmp[i + 1] = (round)(s / 121.0);
    }
#endif
    __m256i t0, m0, e0, m1, e1;
    int xMaskSize = 11;
    Ipp16s sum = 0;
    Ipp16s e[16];
    int width32 = (width >> 5) << 5;
    for (int i = 0; i < xMaskSize; i++)
        sum += pBuf[i];
    int w = 0;
    // int N = 0;
    __m256i yDiv = _mm256_set1_epi32(0x10ED);
    __m256i yRound = _mm256_set1_epi32(0x00040000);
    for (w = 0; w < width32; w += 32)
    {
        __m256i mSrc0, mSrc0_0;
        mSrc0 = (*(__m256i *)(pBuf + 0));
        mSrc0_0 = _mm256_loadu_si256((__m256i *)(pBuf + xMaskSize));
        // mSrc0_0 = (*(__m256i*)(pBuf + xMaskSize));
        t0 = _mm256_sub_epi16(mSrc0_0, mSrc0);
        Ipp16s *p = (Ipp16s *)(&t0);
        Ipp16s *e = (Ipp16s *)(&e0);
        e[0] = sum;
        for (int k = 0; k < 15; k++)
        {
            e[k + 1] = e[k] + p[k];
        }
        sum = e[15] + p[15];

        m0 = _mm256_shuffle_epi8(e0, _mm256_set_epi8(0xFF, 0xFF, 7, 6, 0xFF, 0xFF, 5, 4, 0xFF, 0xFF, 3, 2, 0xFF, 0xFF, 1, 0,
                                                     0xFF, 0xFF, 7, 6, 0xFF, 0xFF, 5, 4, 0xFF, 0xFF, 3, 2, 0xFF, 0xFF, 1, 0));
        e0 = _mm256_shuffle_epi8(e0, _mm256_set_epi8(0xFF, 0xFF, 15, 14, 0xFF, 0xFF, 13, 12, 0xFF, 0xFF, 11, 10, 0xFF, 0xFF, 9, 8,
                                                     0xFF, 0xFF, 15, 14, 0xFF, 0xFF, 13, 12, 0xFF, 0xFF, 11, 10, 0xFF, 0xFF, 9, 8));
        m0 = _mm256_mullo_epi32(m0, yDiv);
        e0 = _mm256_mullo_epi32(e0, yDiv);
        m0 = _mm256_add_epi32(m0, yRound);
        e0 = _mm256_add_epi32(e0, yRound);
        m0 = _mm256_srli_epi32(m0, 19);
        e0 = _mm256_srli_epi32(e0, 19);
        e0 = _mm256_packs_epi32(m0, e0);
        mSrc0 = (*(__m256i *)(pBuf + 16));
        // mSrc0_0 = (*(__m256i*)(pBuf + 16 + xMaskSize));
        mSrc0_0 = _mm256_loadu_si256((__m256i *)(pBuf + 16 + xMaskSize));
        t0 = _mm256_sub_epi16(mSrc0_0, mSrc0);
        e = (Ipp16s *)(&e1);
        e[0] = sum;
        for (int k = 0; k < 15; k++)
        {
            e[k + 1] = e[k] + p[k];
        }
        sum = e[15] + p[15];

        m1 = _mm256_shuffle_epi8(e1, _mm256_set_epi8(0xFF, 0xFF, 7, 6, 0xFF, 0xFF, 5, 4, 0xFF, 0xFF, 3, 2, 0xFF, 0xFF, 1, 0,
                                                     0xFF, 0xFF, 7, 6, 0xFF, 0xFF, 5, 4, 0xFF, 0xFF, 3, 2, 0xFF, 0xFF, 1, 0));
        e1 = _mm256_shuffle_epi8(e1, _mm256_set_epi8(0xFF, 0xFF, 15, 14, 0xFF, 0xFF, 13, 12, 0xFF, 0xFF, 11, 10, 0xFF, 0xFF, 9, 8,
                                                     0xFF, 0xFF, 15, 14, 0xFF, 0xFF, 13, 12, 0xFF, 0xFF, 11, 10, 0xFF, 0xFF, 9, 8));
        m1 = _mm256_mullo_epi32(m1, yDiv);
        e1 = _mm256_mullo_epi32(e1, yDiv);
        m1 = _mm256_add_epi32(m1, yRound);
        e1 = _mm256_add_epi32(e1, yRound);
        m1 = _mm256_srli_epi32(m1, 19);
        e1 = _mm256_srli_epi32(e1, 19);
        e1 = _mm256_packs_epi32(m1, e1);

        e0 = _mm256_packus_epi16(e0, e1);
        e0 = _mm256_permutevar8x32_epi32(e0, _mm256_set_epi32(
                                                 7, 6, 3, 2,
                                                 5, 4, 1, 0));
        //_mm256_storeu_epi8(dst, e0);
        _mm256_storeu_si256((__m256i *)dst, e0);
        dst += 32;
        pBuf += 32;
    }
}

static void x11GetLastRow_8u_C1R(const Ipp8u *__restrict pSrc1, const Ipp8u *__restrict pSrc2, Ipp16f *__restrict pBuf, int size)
{
    int widthS = size;
    int widthS16 = (widthS >> 4) << 4;
    const Ipp8u *rPtr1 = pSrc1;
    const Ipp8u *rPtr2 = pSrc2;
    int i = 0;
    for (i = 0; i < widthS16; i += 16)
    {
        __m256i mmx0, mse;
        __m256i mPtr1, mPtr0;
        mPtr1 = _mm256_cvtepu8_epi16(*((__m128i *)rPtr2));
        mPtr0 = _mm256_cvtepu8_epi16(*((__m128i *)rPtr1));
        mmx0 = _mm256_sub_epi16(mPtr1, mPtr0);
        mse = _mm256_add_epi16(*(__m256i *)pBuf, mmx0);
        _mm256_storeu_si256((__m256i *)pBuf, mse);
        pBuf += 16;
        rPtr1 += 16, rPtr2 += 16;
    }
    for (i = i; i < widthS; i++, rPtr1++, rPtr2++)
    {
        *pBuf++ += *rPtr2 - *rPtr1;
    }
}

static void x11GetFirstSum_8u_C1R(const Ipp8u *pSrc, Ipp64s srcStep, IppiSize dstRoiSize, Ipp16f *pBuffer)
{
    int h, w;
    const Ipp8u *rPtr1, *rPtr2;
    int width = dstRoiSize.width;
    int xMaskSize = 11;
    int yMaskSize = 11;
    int width16 = ((width + xMaskSize - 1) >> 4) << 4;
    rPtr1 = pSrc;
    Ipp16f *pBufferT = (Ipp16f *)pBuffer;
    for (h = 0; h < width16; h += 16)
    {
        __m256 a;
        __m256i ai = _mm256_setzero_si256();
        __m128i aii;
        rPtr2 = rPtr1;
        for (w = yMaskSize; w--;)
        {
            __m256i mSrc0;
            mSrc0 = _mm256_cvtepu8_epi16(*(__m128i *)(rPtr2 + 0));
            ai = _mm256_add_epi16(ai, mSrc0);
            rPtr2 = (Ipp8u *)((Ipp8u *)rPtr2 + srcStep);
        }
        _mm256_storeu_si256((__m256i *)pBufferT, ai);
        rPtr1 += 16;
        pBufferT += 16;
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
                               Ipp8u *pDst, Ipp64s dstStep, IppiSize dstRoiSize, Ipp16f *pBuffer)
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
        Ipp16f *pBuffer = (Ipp16f *)ippMalloc_L(owniAlignStep(img.cols * img.channels(), 64) * sizeof(Ipp16f));
        for (int i = 0; i < 10; i++)
        {
            __itt_task_begin(domain, __itt_null, __itt_null, handle_isd);
            auto ipp_start2 = std::chrono::high_resolution_clock::now();
            x11FilterBoxBorder_8u_C1R((const Ipp8u *)(srcImage.m_ptr), img.cols * img.channels(), (Ipp8u *)(cvtImage2.m_ptr), owniAlignStep(img.cols * img.channels(), 64), roisize, pBuffer);
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