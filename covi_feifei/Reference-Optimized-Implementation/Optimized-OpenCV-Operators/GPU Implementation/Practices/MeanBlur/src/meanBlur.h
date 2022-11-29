//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
void waitForKernelCompletion();
void meanBlurEsimd3x3(const uint8_t *src, uint8_t *dst, int width, int height, int radio);
void meanBlurEsimd5x5(const uint8_t *src, uint8_t *dst, int width, int height, int radio);
void meanBlurEsimd11x11(const uint8_t *src, uint8_t *dst, int width, int height, int radio);
void meanBlurEsimd(const uint8_t *src, uint8_t *dst, int width, int height, int radio);