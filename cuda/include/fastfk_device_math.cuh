#pragma once

#include <cmath>

namespace cs179::gpu {

__device__ inline float fastfk_wrap_pi(float x) {
    const float pi = 3.14159265358979323846f;
    const float two_pi = 6.28318530717958647692f;
    x = fmodf(x + pi, two_pi);
    if (x < 0.0f) {
        x += two_pi;
    }
    return x - pi;
}

__device__ inline float fastfk_device_sin(float x) {
    x = fastfk_wrap_pi(x);
    const float x2 = x * x;
    const float s1 = -1.6666666662148080e-1f;
    const float s2 = 8.3333337680210430e-3f;
    const float s3 = -1.9841270119490766e-4f;
    const float s4 = 2.7557319230698740e-6f;
    const float s5 = -2.5053645798523945e-8f;
    return x * (1.0f + x2 * (s1 + x2 * (s2 + x2 * (s3 + x2 * (s4 + x2 * s5)))));
}

__device__ inline float fastfk_device_cos(float x) {
    x = fastfk_wrap_pi(x);
    const float x2 = x * x;
    const float c1 = -5.0000000000000000e-1f;
    const float c2 = 4.1666666664743006e-2f;
    const float c3 = -1.3888888888888889e-3f;
    const float c4 = 2.4801587301587302e-5f;
    const float c5 = -2.7557319230698740e-7f;
    return 1.0f + x2 * (c1 + x2 * (c2 + x2 * (c3 + x2 * (c4 + x2 * c5))));
}

__device__ inline void fastfk_device_sincos(float x, float* s, float* c) {
    x = fastfk_wrap_pi(x);
    *s = fastfk_device_sin(x);
    *c = fastfk_device_cos(x);
}

}  // namespace cs179::gpu
