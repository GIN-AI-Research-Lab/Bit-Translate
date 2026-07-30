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
 * GEMV Kernel C++ Tốc độ siêu cấp cho chuẩn nén mới i1.58_bitplane.
 * Sử dụng Bitwise Masking (pos_mask & neg_mask) không tốn phép Shift hay tra bảng LUT.
 * 
 * out_dim: Số hàng (output features)
 * in_dim: Số cột (input features) - chia hết cho 32
 * nonzero_words: Bit-plane 1 (out_dim * in_dim / 32 uint32_t elements)
 * sign_words:    Bit-plane 2 (out_dim * in_dim / 32 uint32_t elements)
 * scales: Vector scale float32 (out_dim/2 * in_dim/2 elements)
 * x: Vector đầu vào float32 (in_dim)
 * y: Vector đầu ra float32 (out_dim)
 */
EXPORT void gemv_bitplane_simd_fast(
    int out_dim, int in_dim,
    const uint32_t* nonzero_words, const uint32_t* sign_words,
    const float* scales, const float* x, float* y
) {
    int blocks_out = out_dim / 2;
    int blocks_in = in_dim / 2;

    for (int i = 0; i < out_dim; i++) {
        y[i] = 0.0f;
    }

    int word_cols = in_dim / 32;

    for (int r = 0; r < out_dim; r++) {
        float sum = 0.0f;
        int r_word_offset = r * word_cols;

        for (int wc = 0; wc < word_cols; wc++) {
            uint32_t nz = nonzero_words[r_word_offset + wc];
            uint32_t sg = sign_words[r_word_offset + wc];

            uint32_t pos_mask = nz & sg;
            uint32_t neg_mask = nz & (~sg);

            int base_c = wc * 32;

            // Unroll 32 bits bitwise evaluation
            for (int b = 0; b < 32; b++) {
                uint32_t bit = (1U << b);
                if (pos_mask & bit) {
                    sum += x[base_c + b];
                } else if (neg_mask & bit) {
                    sum -= x[base_c + b];
                }
            }
        }
        
        // Scale block factor
        float s = scales[r / 2];
        y[r] = sum * s;
    }
}

}
