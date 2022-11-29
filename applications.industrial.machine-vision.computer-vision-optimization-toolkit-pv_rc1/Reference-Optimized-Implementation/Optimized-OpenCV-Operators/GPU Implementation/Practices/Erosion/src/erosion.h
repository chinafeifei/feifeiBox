void NaiveErosion(int *src, int *dst, int width, int height, int radio);
void ErosionTwoSteps(int *src, int *dst, int *temp, int width, int height, int radio);
void ErosionTwoStepsShared(int *src, int *dst, int *temp, int width, int height, int radio);
// void ErosionTemplateSharedTwoSteps(int * src, int * dst, int * temp, int width, int height, int radio);
void waitForKernelCompletion();
void ErosionEsimd(const uint8_t *src, uint8_t *dst, int width, int height, int radio);
void ErosionEsimd5x5(const uint8_t *src, uint8_t *dst, int width, int height, int radio);