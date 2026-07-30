// -*- coding: utf-8 -*-
/*
NATIVE C++ SIMD BIT-PLANE EXECUTION ENGINE (`i1.58_bitplane`)
Mục tiêu: Đạt tốc độ suy luận xé gió > 180 - 220 tok/s bằng cổng logic bitwise (pos_mask & neg_mask)
triệt tiêu 100% bit-shift và tra bảng LUT.
*/
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cmath>

#if defined(_WIN32)
#define EXPORT_API extern "C" __declspec(dllexport)
#else
#define EXPORT_API extern "C"
#endif

EXPORT_API void gemv_i158_bitplane_simd(
    int64_t out_dim,
    int64_t in_dim,
    const uint32_t* nz_words,
    const uint32_t* sg_words,
    const float* scales,
    const float* x,
    float* y
) {
    int64_t word_cols = in_dim / 32;

    #pragma omp parallel for schedule(static) if(out_dim > 64)
    for (int64_t r = 0; r < out_dim; r++) {
        float sum = 0.0f;
        int64_t r_offset = r * word_cols;

        for (int64_t wc = 0; wc < word_cols; wc++) {
            uint32_t nz = nz_words[r_offset + wc];
            uint32_t sg = sg_words[r_offset + wc];

            uint32_t pos_mask = nz & sg;
            uint32_t neg_mask = nz & (~sg);

            int64_t base_c = wc * 32;

            #pragma unroll(8)
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
