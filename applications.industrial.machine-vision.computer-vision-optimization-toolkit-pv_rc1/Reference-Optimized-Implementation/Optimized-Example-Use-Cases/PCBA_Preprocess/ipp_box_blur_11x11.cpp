//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include "ippi.h"

#include "ipp_header.h"

#include "ippcore.h"

#include <tbb/tbb.h>

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

static void x11CommonLoop_8u_C1R(Ipp16f *__restrict pBuf, Ipp8u *__restrict dst, int width)
{

    // Common accumulation loop for col sum
    const int divScale = 69327;
    const int divDelta = 61;
    const int SHIFT = 23;

    Ipp16s sumBuf[5000];
    for (int i = 0; i < width; i++)
    {
        sumBuf[i] = pBuf[i] + pBuf[i + 1] + pBuf[i + 2] + pBuf[i + 3] + pBuf[i + 4] + pBuf[i + 5] + pBuf[i + 6] + pBuf[i + 7] + pBuf[i + 8] + pBuf[i + 9] + pBuf[i + 10];

        dst[i] = (uchar)((sumBuf[i] + divDelta) * divScale >> SHIFT);
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

void tbb_ipp_boxfilter_11x11(cv::Mat &image, cv::Mat &output)

{
    // 创建输出图片
    output.create(image.size(), image.type());

    // ipp::IwiImage srcImage, cvtImage2;
    // srcImage.Init(ipp::IwiSize(image.cols, image.rows), ipp8u, image.channels(), NULL, &image.data[0], image.cols * image.channels());
    // cvtImage2.Alloc(srcImage.m_size, ipp8u, image.channels());
    IppiSize maskSize = {11, 11};

    const int THREAD_NUM = 4; // tbb 线程 slot 数量，在 4 核 8 线程的 cpu 上，该值最多为8
                              // 如果设置为 8 ，而 parallel_for 中的 blocked_range 为 0 ，WORKLOAD_NUM = 4
                              // 那么将会有 4 个 tbb 线程在工作，另外 4 个 tbb 线程空闲

    const int WORKLOAD_NUM = 4;                // 将计算负载分割为多少份
    int chunksize = image.rows / WORKLOAD_NUM; // 每个线程计算区域的行数

    //__itt_task_begin(domain, __itt_null, __itt_null, handle_isd);

    boxBufferSize pAllBufferSize;
    IppiSize roiSize = {image.cols, chunksize}; // 每一个线程计算的区域大小
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

    no_hyper_thread_arena.execute([&]
                                  { tbb::parallel_for(tbb::blocked_range<size_t>(0, WORKLOAD_NUM, 1),
                                                      [&](const tbb::blocked_range<size_t> &r)
                                                      {
                                                          // 这里一共有 WORKLOAD_NUM 个循环，由 tbb 自动分配给相应的线程执行
                                                          for (size_t i = r.begin(); i != r.end(); ++i)
                                                          {

                                                              Ipp8u *pSrcT; // 分别指向原矩阵的不同起始地址
                                                              Ipp8u *pDstT; // 分别指向目标矩阵的不同起始地址
                                                              IppStatus tStatus;
                                                              pSrcT = (Ipp8u *)image.data + image.cols * image.channels() * chunksize * i;
                                                              pDstT = (Ipp8u *)(output.data) + image.cols * image.channels() * chunksize * i;
                                                              ;
                                                              int tBorder_top = 13;    // 1101 Bottom in memory, other replicated
                                                              int tBorder_middle = 12; // 1100  Top & Bottom in memory, other replicated
                                                              int tBorder_bottom = 14; // 1110  Top in memory, other replicated
                                                              if (i == 0)
                                                              {
                                                                  x11FilterBoxBorderInMemP_8u_C1R(pSrcT, image.cols * image.channels(), pDstT, image.cols * image.channels(), roiSize, pBufferArray[i], tBorder_top);
                                                              }
                                                              else if (i == 3)
                                                              {
                                                                  x11FilterBoxBorderInMemP_8u_C1R(pSrcT, image.cols * image.channels(), pDstT, image.cols * image.channels(), roiSize, pBufferArray[i], tBorder_bottom);
                                                              }
                                                              else
                                                                  x11FilterBoxBorderInMemP_8u_C1R(pSrcT, image.cols * image.channels(), pDstT, image.cols * image.channels(), roiSize, pBufferArray[i], tBorder_middle);
                                                          }
                                                      }); }); // END no_hyper_thread_arena.execute

    // compare((Ipp8u*)output.data, smoothed, img);
}
