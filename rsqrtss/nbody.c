#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <xmmintrin.h>

static inline void print__m256_vector(__m256 v) {
    float* ptr_init = (float*)&v;
    for (unsigned i = 0; i < 8; i++) printf("%f ", *(ptr_init + i));
    printf("\n");
}

static inline void print__m128_vector(__m128 v) {
    float* ptr_init = (float*)&v;
    for (unsigned i = 0; i < 4; i++) printf("%f ", *(ptr_init + i));
    printf("\n");
}

__m128 square(__m128 num) {
    printf("> square(__m128)\n");

    float* ptr_init = (float*)&num;
    for (unsigned i = 0; i < 4; i++) printf("%f ", *(ptr_init + i));
    printf("\n");

    __m128 a = _mm_rsqrt_ss(num);

    ptr_init = (float*)&a;
    for (unsigned i = 0; i < 4; i++) printf("%f ", *(ptr_init + i));
    printf("\n");

    printf("< square(__m128)\n");
    return a;
}

static inline __m128 create__m128(float* values) {
    return _mm_loadu_ps(values);  //    _mm_setr_ps(16.0f, 1.0f, 4.0f, 9.0f);
}

float reduce_add(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    print__m128_vector(vlow);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);  // high 128
    print__m128_vector(vhigh);
    vlow = _mm_add_ps(vlow, vhigh);  // add the low 128
    print__m128_vector(vlow);
    vlow = _mm_hadd_ps(vlow, vlow);
    print__m128_vector(vlow);
    return ((float*)&vlow)[0] + ((float*)&vlow)[1];  // hsum_ps_sse3(vlow);
    // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}

int main() {
    /*float* vector = malloc(4 * sizeof(float));
    vector[0] = 16.0f;
    vector[1] = 1.0f;
    vector[2] = 4.0f;
    vector[3] = 9.0f;

    __m128 init = create__m128(vector);

    __m128 result = square(init);

    float results[4];
    _mm_store1_ps(results, result);

    for (unsigned i = 0; i < 4; i++) printf("%f ", results[i]);
    printf("\n");
    */

    __m256 vector =
        _mm256_set1_ps(1.0f);  // _mm256_set_ps(1, 2, 3, 4, 5, 6, 7, 8);
    print__m256_vector(vector);
    float result = reduce_add(vector);
    printf("Result is: %f\n", result);

    return 0;
}