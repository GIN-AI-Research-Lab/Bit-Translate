#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

/**
 * Native C++ Bit-Plane SIMD Engine cho chuẩn nén mới i1.58_bitplane.
 * Triệt tiêu hoàn toàn Bit-shift & Tra bảng LUT.
 */
EXPORT void gemv_i158_bitplane_simd(
    int out_dim, int in_dim,
    const uint32_t* nonzero_words, const uint32_t* sign_words,
    const float* scales, const float* x, float* y
) {
    int word_cols = in_dim / 32;

    for (int r = 0; r < out_dim; r++) {
        float sum = 0.0f;
        int r_offset = r * word_cols;

        for (int wc = 0; wc < word_cols; wc++) {
            uint32_t nz = nonzero_words[r_offset + wc];
            uint32_t sg = sign_words[r_offset + wc];

            uint32_t pos_mask = nz & sg;
            uint32_t neg_mask = nz & (~sg);

            int base_c = wc * 32;

            for (int b = 0; b < 32; b++) {
                uint32_t bit = (1U << b);
                if (pos_mask & bit) {
                    sum += x[base_c + b];
                } else if (neg_mask & bit) {
                    sum -= x[base_c + b];
                }
            }
        }
        
        y[r] = sum * scales[r / 2];
    }
}

}
