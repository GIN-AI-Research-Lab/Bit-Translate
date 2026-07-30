#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// Export C functions
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

/**
 * GEMV Kernel cho Block 2x2 Multi-Step Residual Ternary Bitpack (i2_s).
 * 
 * out_dim: Số hàng ma trận W (out_features)
 * in_dim:  Số cột ma trận W (in_features)
 * w1_packed: Ma trận bitpack W1 (out_dim * in_dim / 4 bytes)
 * scale1: Vector scale FP32 của W1 (out_dim/2 * in_dim/2 elements)
 * w2_packed: Ma trận bitpack W2 (out_dim * in_dim / 4 bytes)
 * scale2: Vector scale FP32 của W2 (out_dim/2 * in_dim/2 elements)
 * x: Vector đầu vào float32 (in_dim)
 * y: Vector đầu ra float32 (out_dim)
 */
EXPORT void gemv_ternary_2step_i2s(
    int out_dim, int in_dim,
    const uint8_t* w1_packed, const float* scale1,
    const uint8_t* w2_packed, const float* scale2,
    const float* x, float* y
) {
    // Map 2-bit values: 0 -> -1.0f, 1 -> 0.0f, 2 -> 1.0f
    static const float MAP_TERNARY[4] = {-1.0f, 0.0f, 1.0f, 0.0f};

    // Khởi tạo y = 0
    for (int i = 0; i < out_dim; i++) {
        y[i] = 0.0f;
    }

    int blocks_out = out_dim / 2;
    int blocks_in = in_dim / 2;

    for (int bo = 0; bo < blocks_out; bo++) {
        int r0 = bo * 2;
        int r1 = r0 + 1;
        float sum0 = 0.0f;
        float sum1 = 0.0f;

        for (int bi = 0; bi < blocks_in; bi++) {
            int c0 = bi * 2;
            int c1 = c0 + 1;

            int block_idx = bo * blocks_in + bi;
            float s1 = scale1[block_idx];
            float s2 = scale2[block_idx];

            uint8_t p1 = w1_packed[block_idx];
            uint8_t p2 = w2_packed[block_idx];

            // Decode 4 weights cho Block 2x2: (r0,c0), (r0,c1), (r1,c0), (r1,c1)
            float t1_0 = MAP_TERNARY[p1 & 0x03];
            float t1_1 = MAP_TERNARY[(p1 >> 2) & 0x03];
            float t1_2 = MAP_TERNARY[(p1 >> 4) & 0x03];
            float t1_3 = MAP_TERNARY[(p1 >> 6) & 0x03];

            float t2_0 = MAP_TERNARY[p2 & 0x03];
            float t2_1 = MAP_TERNARY[(p2 >> 2) & 0x03];
            float t2_2 = MAP_TERNARY[(p2 >> 4) & 0x03];
            float t2_3 = MAP_TERNARY[(p2 >> 6) & 0x03];

            float w_00 = t1_0 * s1 + t2_0 * s2;
            float w_01 = t1_1 * s1 + t2_1 * s2;
            float w_10 = t1_2 * s1 + t2_2 * s2;
            float w_11 = t1_3 * s1 + t2_3 * s2;

            float x0 = x[c0];
            float x1 = x[c1];

            sum0 += w_00 * x0 + w_01 * x1;
            sum1 += w_10 * x0 + w_11 * x1;
        }

        y[r0] = sum0;
        y[r1] = sum1;
    }
}

}
