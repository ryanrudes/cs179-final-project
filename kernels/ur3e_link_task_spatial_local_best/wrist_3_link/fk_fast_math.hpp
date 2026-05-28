#ifndef FASTFK_FAST_MATH_HPP
#define FASTFK_FAST_MATH_HPP
#include <cmath>

// Minimax-style sin/cos on [-pi, pi] (~1e-7 max abs err). Joint FK assumes q in [-pi, pi].

static inline double fastfk_wrap_pi(double x) {
  const double pi = (double)3.14159265358979323846;
  const double two_pi = (double)6.28318530717958647692;
  x = std::fmod(x + pi, two_pi);
  if (x < (double)0) x += two_pi;
  return x - pi;
}

static inline double fastfk_fast_sin(double x) {
  x = fastfk_wrap_pi(x);
  const double x2 = x * x;
  const double s1 = (double)-1.6666666662148080e-1;
  const double s2 = (double)8.3333337680210430e-3;
  const double s3 = (double)-1.9841270119490766e-4;
  const double s4 = (double)2.7557319230698740e-6;
  const double s5 = (double)-2.5053645798523945e-8;
  return x * ((double)1 + x2 * (s1 + x2 * (s2 + x2 * (s3 + x2 * (s4 + x2 * s5)))));
}

static inline double fastfk_fast_cos(double x) {
  x = fastfk_wrap_pi(x);
  const double x2 = x * x;
  const double c1 = (double)-5.0000000000000000e-1;
  const double c2 = (double)4.1666666664743006e-2;
  const double c3 = (double)-1.3888888888888889e-3;
  const double c4 = (double)2.4801587301587302e-5;
  const double c5 = (double)-2.7557319230698740e-7;
  return (double)1 + x2 * (c1 + x2 * (c2 + x2 * (c3 + x2 * (c4 + x2 * c5))));
}

static inline void fastfk_fast_sincos(double x, double* s, double* c) {
  x = fastfk_wrap_pi(x);
  *s = fastfk_fast_sin(x);
  *c = fastfk_fast_cos(x);
}

static inline void fastfk_sincos(double x, double* s, double* c) {
#if defined(__APPLE__)
  __sincos(x, s, c);
#elif defined(_GNU_SOURCE) || defined(__GLIBC__)
  ::sincos(x, s, c);
#else
  *s = std::sin(x);
  *c = std::cos(x);
#endif
}

#endif  // FASTFK_FAST_MATH_HPP
