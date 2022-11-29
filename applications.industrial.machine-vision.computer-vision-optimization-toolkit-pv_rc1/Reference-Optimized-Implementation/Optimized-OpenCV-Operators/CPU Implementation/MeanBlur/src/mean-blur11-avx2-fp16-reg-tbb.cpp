//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <memory.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ittnotify.h>
#include "iw++/iw.hpp"
#include "ippi.h"
#include <tbb/tbb.h>
#include <iostream>
#if !(defined(_MSC_VER))
#include <immintrin.h>
#include <avxintrin.h>
#endif
#pragma warning(disable : 4700)

using namespace tbb;
using timeunit = std::chrono::microseconds;

// Create a domain that is visible globally: we will use it in our example.
__itt_domain *domain = __itt_domain_create((const wchar_t *)("Example.Domain.Global"));
// Create string handles which associates with the "main" task.
__itt_string_handle *handle_cv = __itt_string_handle_create((const wchar_t *)"cv");
__itt_string_handle *handle_ipp = __itt_string_handle_create((const wchar_t *)"ipp");
__itt_string_handle *handle_isd = __itt_string_handle_create((const wchar_t *)"isd");
__itt_string_handle *handle_for_loop = __itt_string_handle_create((const wchar_t *)"for_loop");

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

static void x11CommonLoop_8u_C1R(Ipp16f *__restrict pBuf, Ipp8u *__restrict dst, int width)
{
    __m256i t0, m0, e0, m1, e1;
    int xMaskSize = 11;
    int width32 = (width >> 5) << 5;
    int width4 = (width >> 2) << 2;
    int width8 = (width >> 3) << 3;
    int w = 0;
    __m256i yDiv = _mm256_set1_epi32(0x10ED);
    __m256i yRound = _mm256_set1_epi32(0x00040000);
    __m256i mask = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0xFFFF, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i mSrc0, mSrc0_0;
    __m256i mSum0, mSum1;
    Ipp16f sum = 0;
    for (int i = 0; i < xMaskSize; i++)
        sum += pBuf[i];

    __m256i eee;
    __m256i ppp;
    Ipp16f *ee = (Ipp16f *)&eee;
    Ipp16f *pp = (Ipp16f *)&t0;
    for (w = 0; w < width32; w += 32)
    {
        mSrc0 = (*(__m256i *)(pBuf));
        mSrc0_0 = _mm256_loadu_si256((__m256i *)(pBuf + xMaskSize));
        // mSrc0_0 = (*(__m256i*)(pBuf + xMaskSize));
        t0 = _mm256_sub_epi16(mSrc0_0, mSrc0);

        ppp = _mm256_and_si256(_mm256_set1_epi16(pp[7]), mask);
        ppp = _mm256_or_si256(ppp, _mm256_slli_si256(t0, 2));
        eee = _mm256_add_epi16(ppp, _mm256_slli_si256(ppp, 2));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 4));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 6));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 8));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 10));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 12));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 14));

        mSum0 = _mm256_set1_epi32(sum);
        sum += ee[7];
        mSum1 = _mm256_set1_epi32(sum);
        sum += pp[15] + ee[15];

        m0 = _mm256_cvtepi16_epi32(*(__m128i *)ee);
        e0 = _mm256_cvtepi16_epi32(*(__m128i *)(ee + 8));
        m0 = _mm256_add_epi32(m0, mSum0);
        e0 = _mm256_add_epi32(e0, mSum1);
        m0 = _mm256_mullo_epi32(m0, yDiv);
        e0 = _mm256_mullo_epi32(e0, yDiv);
        m0 = _mm256_add_epi32(m0, yRound);
        e0 = _mm256_add_epi32(e0, yRound);
        m0 = _mm256_srli_epi32(m0, 19);
        e0 = _mm256_srli_epi32(e0, 19);
        e0 = _mm256_packs_epi32(m0, e0);
        pBuf += 16;

        mSrc0 = (*(__m256i *)(pBuf));
        mSrc0_0 = (*(__m256i *)(pBuf + xMaskSize));
        mSrc0_0 = _mm256_loadu_si256((__m256i *)(pBuf + xMaskSize));
        t0 = _mm256_sub_epi16(mSrc0_0, mSrc0);

        ppp = _mm256_and_si256(_mm256_set1_epi16(pp[7]), mask);
        ppp = _mm256_or_si256(ppp, _mm256_slli_si256(t0, 2));
        eee = _mm256_add_epi16(ppp, _mm256_slli_si256(ppp, 2));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 4));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 6));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 8));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 10));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 12));
        eee = _mm256_add_epi16(eee, _mm256_slli_si256(ppp, 14));

        mSum0 = _mm256_set1_epi32(sum);
        sum += ee[7];
        mSum1 = _mm256_set1_epi32(sum);
        sum += pp[15] + ee[15];

        m1 = _mm256_cvtepi16_epi32(*(__m128i *)ee);
        e1 = _mm256_cvtepi16_epi32(*(__m128i *)(ee + 8));
        m1 = _mm256_add_epi32(m1, mSum0);
        e1 = _mm256_add_epi32(e1, mSum1);
        m1 = _mm256_mullo_epi32(m1, yDiv);
        e1 = _mm256_mullo_epi32(e1, yDiv);
        m1 = _mm256_add_epi32(m1, yRound);
        e1 = _mm256_add_epi32(e1, yRound);
        m1 = _mm256_srli_epi32(m1, 19);
        e1 = _mm256_srli_epi32(e1, 19);
        e1 = _mm256_packs_epi32(m1, e1);
        e0 = _mm256_packus_epi16(e0, e1);
        e0 = _mm256_permutevar8x32_epi32(e0, _mm256_set_epi32(
                                                 7, 3, 6, 2,
                                                 5, 1, 4, 0));
        //_mm256_storeu_epi8(dst, e0);
        _mm256_storeu_si256((__m256i *)dst, e0);
        dst += 32;
        pBuf += 16;
    }
    for (w = w; w < width; w++)
    {
        dst[0] = (round)(sum / 121.0);
        sum += pBuf[xMaskSize] - pBuf[0];
        dst++;
        pBuf++;
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

    for (h = h; h < (width + xMaskSize - 1); h++, rPtr1++)
    {
        rPtr2 = rPtr1;
        *pBufferT = 0.f;
        for (w = yMaskSize; w--;)
        {
            *pBufferT += *rPtr2;
            rPtr2 = (Ipp8u *)((Ipp8u *)rPtr2 + srcStep);
        }
        pBufferT++;
    }
    *pBufferT = 0.f;
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
    x11GetFirstSum_8u_C1R(pSrc, srcStep, dstRoiSize, pBuffer);
    for (h = 0; h < height; h++)
    {
        int lasrow = (h == (height - 1)) ? 0 : 1;
        const Ipp8u *rPtr1 = (Ipp8u *)((Ipp8u *)pSrc + (Ipp64s)h * srcStep);
        Ipp8u *dst = (Ipp8u *)((Ipp8u *)pDst + (Ipp64s)h * dstStep);
        rPtr2 = (Ipp8u *)((Ipp8u *)rPtr1 + yMaskSize * srcStep);
        x11CommonLoop_8u_C1R(pBuffer, dst, width);
        if (lasrow)
        {
            x11GetLastRow_8u_C1R(rPtr1, rPtr2, pBuffer, (width + xMaskSize - 1));
        }
    }
}

typedef struct
{
    Ipp64s lenBorder;
    Ipp64s lenTemp;
} boxBufferSize;

IppStatus x11FilterBoxBorderGetBufferSize(IppiSize roiSize, boxBufferSize *pBufferSize)
{
    IppiSize maskSize = {11, 11};
    Ipp64s lenUp = 0;
    Ipp64s lenBottom = 0;
    Ipp64s lenLeft = 0;
    Ipp64s lenRight = 0;
    Ipp64s lenTemp = 0;

    lenUp = ((Ipp64s)roiSize.width + (maskSize.width - 1)) * ((Ipp64s)maskSize.height - 1 + 5);
    lenLeft = ((Ipp64s)roiSize.height + (maskSize.height - 1)) * ((Ipp64s)maskSize.width - 1 + 5);

    lenBottom = (maskSize.height != 2) ? lenUp : lenUp + ((Ipp64s)roiSize.width + (maskSize.width - 1));
    lenRight = (maskSize.width != 2) ? lenLeft : lenLeft + ((Ipp64s)roiSize.height + (maskSize.height - 1));

    lenUp = lenUp + lenBottom + lenLeft + lenRight;
    pBufferSize->lenBorder = lenUp;

    lenTemp = (Ipp64s)roiSize.width + maskSize.width; // +3;

    pBufferSize->lenTemp = lenTemp * sizeof(Ipp16f);
    return ippStsNoErr;
}

static IppStatus x11FilterBoxBorderInMemP_8u_C1R(const Ipp8u *pSrc, int srcStep32,
                                                 Ipp8u *pDst, int dstStep32, IppiSize dstRoiSize, Ipp8u *pBuffer, int invBorder = 0x0f)
{
    boxBufferSize pAllBufferSize;
    IppStatus status = ippStsNoErr;
    IppiSize maskSize = {11, 11};
    IppiBorderType Inborder = ippBorderRepl;
    // Inborder = IppiBorderType(ippBorderFirstStageInMemTop| ippBorderRepl);
    IppiPoint anchor = {5, 5};
    IppiPoint invanchor = {5, 5};
    Ipp64s dstStep = (Ipp64s)dstStep32, srcStep = (Ipp64s)srcStep32;

    int topBorderHeight, leftBorderWidth;
    Ipp8u *pBuf = pBuffer;
    Ipp32f *pTempbuf = (Ipp32f *)pBuffer;
    Ipp8u *pDstBdr;
    const Ipp8u *pSrcBdr;
    IppiSize srcRoiSize;
    IppiSize brdRoiSize;
    int topBrdH = 0;
    int botBrdH = 0;
    const Ipp8u *src = pSrc;
    Ipp8u *dst = pDst;
    int topOffsetSrc = 0;

    x11FilterBoxBorderGetBufferSize(dstRoiSize, &pAllBufferSize);
    pBuf = (Ipp8u *)(pBuffer + pAllBufferSize.lenTemp);

    /* Inside */
    if (invBorder == ippBorderInMem)
    {
        src = (Ipp8u *)((Ipp8u *)(pSrc - anchor.x) - anchor.y * srcStep);
        x11FilterBoxBorder_8u_C1R(src, srcStep, dst, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
        return status;
    }
    else
    {
        srcRoiSize.width = dstRoiSize.width - (invanchor.x + anchor.x);
        srcRoiSize.height = dstRoiSize.height - (invanchor.y + anchor.y);
        dst = (Ipp8u *)((Ipp8u *)(pDst + anchor.x) + anchor.y * dstStep);
        x11FilterBoxBorder_8u_C1R(src, srcStep, dst, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
    }

    /* Top */
    if (invBorder & 1)
    {
        srcRoiSize.width = dstRoiSize.width;
        srcRoiSize.height = anchor.y + invanchor.y;
        brdRoiSize.width = dstRoiSize.width + (maskSize.width - 1);
        brdRoiSize.height = srcRoiSize.height + anchor.y;

        leftBorderWidth = anchor.x * ((invBorder & 4) >> 2);
        topBorderHeight = anchor.y;
        pSrcBdr = pSrc;
        ippiCopyReplicateBorder_8u_C1R(pSrcBdr, (int)srcStep, srcRoiSize, (Ipp8u *)pBuf, brdRoiSize.width * sizeof(Ipp8u), brdRoiSize, topBorderHeight, leftBorderWidth);
        srcRoiSize.height = anchor.y;
        srcRoiSize.width = dstRoiSize.width;
        topBrdH = srcRoiSize.height;
        x11FilterBoxBorder_8u_C1R((Ipp8u *)pBuf, (Ipp64s)(brdRoiSize.width * sizeof(Ipp8u)), pDst, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
    }
    else
    {
        IppiSize srcRoiSize = {dstRoiSize.width - (invanchor.x + anchor.x), anchor.y};
        topBrdH = 0;
        topOffsetSrc -= (anchor.y * srcStep);
        src = (Ipp8u *)((Ipp8u *)(pSrc + topOffsetSrc));
        dst = (Ipp8u *)((Ipp8u *)(pDst + anchor.x));
        x11FilterBoxBorder_8u_C1R(src, srcStep, dst, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
    }

    /* Buttom */
    if (invBorder & 2)
    {
        srcRoiSize.width = dstRoiSize.width;
        srcRoiSize.height = anchor.y + invanchor.y;
        brdRoiSize.width = dstRoiSize.width + maskSize.width - 1;
        brdRoiSize.height = srcRoiSize.height + invanchor.y;
        leftBorderWidth = anchor.x * ((invBorder & 4) >> 2);
        topBorderHeight = 0;
        pSrcBdr = (Ipp8u *)((Ipp8u *)pSrc + (dstRoiSize.height - invanchor.y - anchor.y) * srcStep);
        ippiCopyReplicateBorder_8u_C1R(pSrcBdr, (int)srcStep, srcRoiSize, (Ipp8u *)pBuf, brdRoiSize.width * sizeof(Ipp8u), brdRoiSize, topBorderHeight, leftBorderWidth);
        srcRoiSize.height = invanchor.y;
        srcRoiSize.width = dstRoiSize.width;
        botBrdH = srcRoiSize.height;
        pDstBdr = (Ipp8u *)((Ipp8u *)(pDst + 0) + (dstRoiSize.height - srcRoiSize.height) * dstStep);
        x11FilterBoxBorder_8u_C1R((Ipp8u *)pBuf, (Ipp64s)(brdRoiSize.width * sizeof(Ipp8u)), pDstBdr, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
    }
    else
    {
        IppiSize srcRoiSize = {dstRoiSize.width - (invanchor.x + anchor.x), invanchor.y};
        botBrdH = 0;
        src = (Ipp8u *)((Ipp8u *)pSrc + (dstRoiSize.height - anchor.y) * srcStep - invanchor.y * srcStep);
        dst = (Ipp8u *)((Ipp8u *)(pDst + anchor.x) + (dstRoiSize.height - invanchor.y) * dstStep);
        x11FilterBoxBorder_8u_C1R(src, srcStep, dst, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
    }

    /* Left */
    if (invBorder & 4)
    {
        srcRoiSize.width = anchor.x + invanchor.x;
        srcRoiSize.height = (dstRoiSize.height - botBrdH - topBrdH) + (maskSize.height - 1);

        brdRoiSize.width = srcRoiSize.width + anchor.x;
        brdRoiSize.height = srcRoiSize.height;

        leftBorderWidth = anchor.x;
        topBorderHeight = 0;
        pSrcBdr = (Ipp8u *)((Ipp8u *)pSrc + topOffsetSrc);
        ippiCopyReplicateBorder_8u_C1R(pSrcBdr, (int)srcStep, srcRoiSize, (Ipp8u *)pBuf, brdRoiSize.width * sizeof(Ipp8u), brdRoiSize, topBorderHeight, leftBorderWidth);
        srcRoiSize.width = anchor.x;
        srcRoiSize.height = dstRoiSize.height - (topBrdH + botBrdH);
        pDstBdr = (Ipp8u *)((Ipp8u *)(pDst + 0) + topBrdH * dstStep);
        x11FilterBoxBorder_8u_C1R((Ipp8u *)pBuf, (Ipp64s)(brdRoiSize.width * sizeof(Ipp8u)), pDstBdr, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
    }
    else
    {
        IppiSize srcRoiSize = {anchor.x, dstRoiSize.height - (topBrdH + botBrdH)};
        src = (Ipp8u *)(pSrc - anchor.x);
        dst = (Ipp8u *)((Ipp8u *)pDst + topBrdH * dstStep);
        x11FilterBoxBorder_8u_C1R(src, srcStep, dst, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
    }

    /* Right */
    if (invBorder & 8)
    {
        srcRoiSize.width = invanchor.x + anchor.x;
        srcRoiSize.height = (dstRoiSize.height - botBrdH - topBrdH) + (maskSize.height - 1);
        brdRoiSize.width = srcRoiSize.width + invanchor.x;
        brdRoiSize.height = srcRoiSize.height;
        leftBorderWidth = 0;
        topBorderHeight = 0;
        pSrcBdr = (Ipp8u *)((Ipp8u *)(pSrc + dstRoiSize.width - srcRoiSize.width) + topOffsetSrc);
        ippiCopyReplicateBorder_8u_C1R(pSrcBdr, (int)srcStep, srcRoiSize, (Ipp8u *)pBuf, brdRoiSize.width * sizeof(Ipp8u), brdRoiSize, topBorderHeight, leftBorderWidth);
        srcRoiSize.width = invanchor.x;
        srcRoiSize.height = dstRoiSize.height - (topBrdH + botBrdH);
        pDstBdr = (Ipp8u *)((Ipp8u *)(pDst + dstRoiSize.width - srcRoiSize.width) + topBrdH * dstStep);
        x11FilterBoxBorder_8u_C1R((Ipp8u *)pBuf, (Ipp64s)(brdRoiSize.width * sizeof(Ipp8u)), pDstBdr, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
    }
    else
    {
        IppiSize srcRoiSize = {invanchor.x, dstRoiSize.height - (topBrdH + botBrdH)};
        src = (Ipp8u *)((Ipp8u *)(pSrc + dstRoiSize.width - anchor.x - invanchor.x) + topOffsetSrc);
        dst = (Ipp8u *)((Ipp8u *)(pDst + dstRoiSize.width - invanchor.x) + topBrdH * dstStep);
        x11FilterBoxBorder_8u_C1R(src, srcStep, dst, dstStep, srcRoiSize, (Ipp16f *)pTempbuf);
    }

    return status;
}

void tbb_ipp_meanfilter(cv::Mat img, cv::Mat smoothed)

{
    ipp::IwiImage srcImage, cvtImage2;
    srcImage.Init(ipp::IwiSize(img.cols, img.rows), ipp8u, img.channels(), NULL, &img.data[0], img.cols * img.channels());
    cvtImage2.Alloc(srcImage.m_size, ipp8u, img.channels());
    IppiSize maskSize = {11, 11};

    const int THREAD_NUM = 4; // tbb 线程 slot 数量，在 4 核 8 线程的 cpu 上，该值最多为8
                              // 如果设置为 8 ，而 parallel_for 中的 blocked_range 为 0 ，WORKLOAD_NUM = 4
                              // 那么将会有 4 个 tbb 线程在工作，另外 4 个 tbb 线程空闲

    const int WORKLOAD_NUM = 4;              // 将计算负载分割为多少份
    int chunksize = img.rows / WORKLOAD_NUM; // 每个线程计算区域的行数

    __itt_task_begin(domain, __itt_null, __itt_null, handle_isd);

    boxBufferSize pAllBufferSize;
    IppiSize roiSize = {img.cols, chunksize}; // 每一个线程计算的区域大小
    x11FilterBoxBorderGetBufferSize(roiSize, &pAllBufferSize);
    int tmpBufferSize = pAllBufferSize.lenBorder + pAllBufferSize.lenTemp;
    Ipp8u *pBufferArray[WORKLOAD_NUM];
    for (int i = 0; i < WORKLOAD_NUM; i++)
    {
        pBufferArray[i] = (Ipp8u *)ippMalloc_L(tmpBufferSize);
    }
    tbb::task_arena no_hyper_thread_arena(tbb::task_arena::constraints{}.set_max_threads_per_core(1).set_max_concurrency(THREAD_NUM));
    // 限制每个物理核只能运行一个 tbb 线程，避免超线程影响
    // 设置 tbb 线程 slot 数量
    float time_tbb = 0;
    const int LOOP_NUM = 20;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        auto tbb_start = std::chrono::high_resolution_clock::now();
        no_hyper_thread_arena.execute([&]
                                      { parallel_for(blocked_range<size_t>(0, WORKLOAD_NUM, 1),
                                                     [&](const blocked_range<size_t> &r)
                                                     {
                                                         // 这里一共有 WORKLOAD_NUM 个循环，由 tbb 自动分配给相应的线程执行
                                                         for (size_t i = r.begin(); i != r.end(); ++i)
                                                         {

                                                             Ipp8u *pSrcT; // 分别指向原矩阵的不同起始地址
                                                             Ipp8u *pDstT; // 分别指向目标矩阵的不同起始地址
                                                             IppStatus tStatus;
                                                             pSrcT = (Ipp8u *)srcImage.m_ptr + img.cols * img.channels() * chunksize * i;
                                                             pDstT = (Ipp8u *)(cvtImage2.m_ptr) + owniAlignStep(img.cols * img.channels(), 64) * chunksize * i;
                                                             int tBorder_top = 13;    // 1101 Bottom in memory, other replicated
                                                             int tBorder_middle = 12; // 1100  Top & Bottom in memory, other replicated
                                                             int tBorder_bottom = 14; // 1110  Top in memory, other replicated
                                                             if (i == 0)
                                                             {
                                                                 x11FilterBoxBorderInMemP_8u_C1R(pSrcT, img.cols * img.channels(), pDstT, owniAlignStep(img.cols * img.channels(), 64), roiSize, pBufferArray[i], tBorder_top);
                                                             }
                                                             else if (i == 3)
                                                             {
                                                                 x11FilterBoxBorderInMemP_8u_C1R(pSrcT, img.cols * img.channels(), pDstT, owniAlignStep(img.cols * img.channels(), 64), roiSize, pBufferArray[i], tBorder_bottom);
                                                             }
                                                             else
                                                                 x11FilterBoxBorderInMemP_8u_C1R(pSrcT, img.cols * img.channels(), pDstT, owniAlignStep(img.cols * img.channels(), 64), roiSize, pBufferArray[i], tBorder_middle);
                                                         }
                                                     }); });
        auto tbb_stop = std::chrono::high_resolution_clock::now();
        auto tbb_time = std::chrono::duration_cast<timeunit>(tbb_stop - tbb_start).count() / 1000.0;
        time_tbb += tbb_time;
        __itt_task_end(domain);
    }

    printf("\nOwn IPP Time consuming(kernel size %d): %f\n", 11, time_tbb / (LOOP_NUM));
    compare((Ipp8u *)cvtImage2.m_ptr, smoothed, img);
}

int main(int argc, char **argv)
{
    // Create images
#ifdef _WIN32
    HANDLE mHandle = GetCurrentProcess();
    BOOL result = SetPriorityClass(mHandle, REALTIME_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x00000001); // 0x01，Core0
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
            // cv::blur(img, smoothed, cv::Size(ksize, ksize), cv::Point(-1, -1));
            cv::boxFilter(img, smoothed, -1, cv::Size(ksize, ksize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
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
        IppiSize roiSize = {img.cols, img.rows};
        IppiSize maskSize = {11, 11};
        float time_ipp2 = 0;
        Ipp16f *pBuffer = (Ipp16f *)ippMalloc_L(owniAlignStep(img.cols * img.channels(), 64) * sizeof(Ipp16f));

        boxBufferSize pAllBufferSize;
        IppiSize roiSizeBuf;
        roiSizeBuf.height = roiSize.height + maskSize.height - 1;
        roiSizeBuf.width = roiSize.width + maskSize.width - 1;
        int topBorderHeight;
        int leftBorderWidth, srcStepBuf;
        srcStepBuf = roiSizeBuf.width * sizeof(Ipp8u);
        topBorderHeight = 5;
        leftBorderWidth = 5;
        for (int i = 0; i < 10; i++)
        {
            __itt_task_begin(domain, __itt_null, __itt_null, handle_isd);
            auto ipp_start2 = std::chrono::high_resolution_clock::now();
            IppiSize roiSize = {img.cols, img.rows};
            x11FilterBoxBorderGetBufferSize(roiSize, &pAllBufferSize);
            // int tmpBufferSize = pAllBufferSize.lenBorder + pAllBufferSize.lenTemp + pAllBufferSize.sizeInPlsBuf + pAllBufferSize.sizeInPlsPtr;
            int tmpBufferSize = pAllBufferSize.lenBorder + pAllBufferSize.lenTemp;
            Ipp8u *pTmpBuffer = (Ipp8u *)ippMalloc_L(tmpBufferSize);
            IwiBorderType border = ippBorderRepl;
            Ipp64f borderVal[4] = {0};
            const Ipp64f *pBorderVal = borderVal;
            x11FilterBoxBorderInMemP_8u_C1R((Ipp8u *)((Ipp8u *)srcImage.m_ptr), img.cols * img.channels(), (Ipp8u *)(cvtImage2.m_ptr), owniAlignStep(img.cols * img.channels(), 64), roiSize, pTmpBuffer);
            // x11FilterBoxBorder_8u_C1R((const Ipp8u*)(srcImage.m_ptr), img.cols * img.channels(), (Ipp8u*)(cvtImage2.m_ptr), owniAlignStep(img.cols * img.channels(), 64), roiSize, pBuffer);
            auto ipp_stop2 = std::chrono::high_resolution_clock::now();
            auto ipp_time2 = std::chrono::duration_cast<timeunit>(ipp_stop2 - ipp_start2).count() / 1000.0;
            time_ipp2 += ipp_time2;
            __itt_task_end(domain);
        }
        printf("\nOwn IPP Time consuming(kernel size %d): %f\n", ksize, time_ipp2 / 10);
        compare((Ipp8u *)cvtImage2.m_ptr, smoothed, img);
    }

    // tbb_test
    tbb_ipp_meanfilter(img, smoothed);
    return 0;
}