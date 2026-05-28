#pragma once
#include "fastfk_device_math.cuh"

namespace cs179::gpu {
namespace ur3e {

constexpr int kTool0JacRows = 6;
constexpr int kTool0JacCols = 6;
constexpr int kTool0JacSize = kTool0JacRows * kTool0JacCols;

/// FK + Pinocchio ``LOCAL`` spatial Jacobian (6 x 6); zero ``J`` before call.
__device__ inline void tool0_fk_jacobian(
    const float* q,
    float* R,
    float* p,
    float* J)
{

  // frame: tool0
  {
  float fk_s42 = 0, fk_c41 = 0;
  fastfk_device_sincos(q[0], &fk_s42, &fk_c41);
  const float r0 = fk_c41;
  float fk_s48 = 0, fk_c47 = 0;
  fastfk_device_sincos(q[1], &fk_s48, &fk_c47);
  const float r3 = fk_s48;
  float fk_s58 = 0, fk_c57 = 0;
  fastfk_device_sincos(q[2], &fk_s58, &fk_c57);
  const float r4 = fk_c57;
  const float t184 = ((r3) * (r4) + ((fk_c47 * fk_s58)));
  const float t188 = ((t184) * (-0.2132) + (((r3) * (-0.24354999999999999) + (0.15185000000000001))));
  const float r1 = (-(fk_s42));
  const float t178 = (r1 * fk_c47);
  const float t179 = (r1 * (-(r3)));
  const float t182 = ((t178) * (r4) + ((t179 * fk_s58)));
  const float t187 = ((t178) * (-0.24354999999999999) + (((t182) * (-0.2132) + ((r0 * 0.13105)))));
  float fk_s85 = 0, fk_c84 = 0;
  fastfk_device_sincos(q[3], &fk_s85, &fk_c84);
  const float r8 = fk_s85;
  const float r9 = (-(r8));
  const float r6 = (-(fk_s58));
  const float t183 = ((t178) * (r6) + ((t179 * r4)));
  const float t192 = ((t182) * (r9) + ((t183 * fk_c84)));
  const float t196 = (t187 + ((t192) * (-0.085349999999999995) + ((r0 * -1.750557762378351e-11))));
  const float t185 = ((r3) * (r6) + ((fk_c47 * r4)));
  const float t194 = ((t184) * (r9) + ((t185 * fk_c84)));
  const float t193 = ((t184) * (fk_c84) + ((t185 * r8)));
  float fk_s118 = 0, fk_c117 = 0;
  fastfk_device_sincos(q[4], &fk_s118, &fk_c117);
  const float r11 = fk_s118;
  const float r12 = (-(r11));
  const float t203 = (t193 * r12);
  const float t204 = (-(t194));
  const float t207 = (((t194) * (-0.085349999999999995) + (t188)) + ((t203) * (0.092100000000000001) + ((t204 * -1.8890025766261998e-11))));
  const float t177 = ((-(r0)) * (-(r3)));
  const float r15 = ((-(r0)) * fk_c47);
  const float t180 = ((r15) * (r4) + ((t177 * fk_s58)));
  const float t186 = ((r15) * (-0.24354999999999999) + (((t180) * (-0.2132) + ((r1 * 0.13105)))));
  const float t181 = ((r15) * (r6) + ((t177 * r4)));
  const float t190 = ((t180) * (r9) + ((t181 * fk_c84)));
  const float t195 = (t186 + ((t190) * (-0.085349999999999995) + ((r1 * -1.750557762378351e-11))));
  const float t191 = ((t182) * (fk_c84) + ((t183 * r8)));
  const float t201 = ((t191) * (r12) + ((r0 * fk_c117)));
  const float t202 = (-(t192));
  const float t206 = (t196 + ((t201) * (0.092100000000000001) + ((t202 * -1.8890025766261998e-11))));
  const float t189 = ((t180) * (fk_c84) + ((t181 * r8)));
  const float t198 = ((t189) * (r12) + ((r1 * fk_c117)));
  const float t199 = (-(t190));
  const float t205 = (t195 + ((t198) * (0.092100000000000001) + ((t199 * -1.8890025766261998e-11))));
  float fk_s150 = 0, fk_c149 = 0;
  fastfk_device_sincos(q[5], &fk_s150, &fk_c149);
  const float r13 = fk_c149;
  const float r14 = (-(fk_s150));
  const float t213 = (((t193 * fk_c117)) * (r13) + ((t204 * r14)));
  const float t209 = (-(r13));
  const float t214 = (((t193 * fk_c117)) * (r14) + ((t204 * t209)));
  const float t197 = ((t189) * (fk_c117) + ((r1 * r11)));
  const float t200 = ((t191) * (fk_c117) + ((r0 * r11)));
  const float t208 = ((t197) * (r13) + ((t199 * r14)));
  const float t210 = ((t197) * (r14) + ((t199 * t209)));
  const float t211 = ((t200) * (r13) + ((t202 * r14)));
  const float t212 = ((t200) * (r14) + ((t202 * t209)));
  R[0] = t208;
  R[1] = t210;
  R[2] = t198;
  R[3] = t211;
  R[4] = t212;
  R[5] = t201;
  R[6] = t213;
  R[7] = t214;
  R[8] = t203;
  p[0] = t205;
  p[1] = t206;
  p[2] = t207;
  // Jacobian (6 x 6)
  // q-dependent entries only; zero J before call (see fastfk_init_static_jacobian).
  const float t215 = (0 - t206);
  const float t216 = ((r1) * (t206) + ((-((r0 * t205)))));
  const float t217 = (r0 * (t207 - 0.15185000000000001));
  const float t218 = (0 - (r1 * (t207 - 0.15185000000000001)));
  const float t220 = ((r1) * ((t206 - (t178 * -0.24354999999999999))) + ((-((r0 * (t205 - (r15 * -0.24354999999999999)))))));
  const float t219 = (t207 - ((r3) * (-0.24354999999999999) + (0.15185000000000001)));
  const float t221 = (r0 * t219);
  const float t222 = (0 - (r1 * t219));
  const float t223 = ((r1) * ((t206 - t187)) + ((-((r0 * (t205 - t186))))));
  const float t224 = (r0 * (t207 - t188));
  const float t225 = (0 - (r1 * (t207 - t188)));
  const float t227 = ((t199) * ((t206 - t196)) + ((-((t202 * (t205 - t195))))));
  const float t226 = (t207 - ((t194) * (-0.085349999999999995) + (t188)));
  const float t228 = ((t202) * (t226) + ((-((t204 * (t206 - t196))))));
  const float t229 = ((t204) * ((t205 - t195)) + ((-((t199 * t226)))));
  const float t230 = ((t198) * ((t206 - t206)) + ((-((t201 * (t205 - t205))))));
  const float t231 = ((t201) * ((t207 - t207)) + ((-((t203 * (t206 - t206))))));
  const float t232 = ((t203) * ((t205 - t205)) + ((-((t198 * (t207 - t207))))));
  const float t233 = ((t208) * (r1) + ((t211 * r0)));
  const float t234 = ((t210) * (r1) + ((t212 * r0)));
  const float t235 = ((t198) * (r1) + ((t201 * r0)));
  J[0 * 6 + 0] = ((t208) * (t215) + ((t211 * t205)));
  J[0 * 6 + 1] = ((t213) * (t216) + (((t208) * (t217) + ((t211 * t218)))));
  J[0 * 6 + 2] = ((t213) * (t220) + (((t208) * (t221) + ((t211 * t222)))));
  J[0 * 6 + 3] = ((t213) * (t223) + (((t208) * (t224) + ((t211 * t225)))));
  J[0 * 6 + 4] = ((t213) * (t227) + (((t208) * (t228) + ((t211 * t229)))));
  J[0 * 6 + 5] = ((t213) * (t230) + (((t208) * (t231) + ((t211 * t232)))));
  J[1 * 6 + 0] = ((t210) * (t215) + ((t212 * t205)));
  J[1 * 6 + 1] = ((t214) * (t216) + (((t210) * (t217) + ((t212 * t218)))));
  J[1 * 6 + 2] = ((t214) * (t220) + (((t210) * (t221) + ((t212 * t222)))));
  J[1 * 6 + 3] = ((t214) * (t223) + (((t210) * (t224) + ((t212 * t225)))));
  J[1 * 6 + 4] = ((t214) * (t227) + (((t210) * (t228) + ((t212 * t229)))));
  J[1 * 6 + 5] = ((t214) * (t230) + (((t210) * (t231) + ((t212 * t232)))));
  J[2 * 6 + 0] = ((t198) * (t215) + ((t201 * t205)));
  J[2 * 6 + 1] = ((t203) * (t216) + (((t198) * (t217) + ((t201 * t218)))));
  J[2 * 6 + 2] = ((t203) * (t220) + (((t198) * (t221) + ((t201 * t222)))));
  J[2 * 6 + 3] = ((t203) * (t223) + (((t198) * (t224) + ((t201 * t225)))));
  J[2 * 6 + 4] = ((t203) * (t227) + (((t198) * (t228) + ((t201 * t229)))));
  J[2 * 6 + 5] = ((t203) * (t230) + (((t198) * (t231) + ((t201 * t232)))));
  J[3 * 6 + 0] = t213;
  J[3 * 6 + 1] = t233;
  J[3 * 6 + 2] = t233;
  J[3 * 6 + 3] = t233;
  J[3 * 6 + 4] = ((t213) * (t204) + (((t208) * (t199) + ((t211 * t202)))));
  J[3 * 6 + 5] = ((t213) * (t203) + (((t208) * (t198) + ((t211 * t201)))));
  J[4 * 6 + 0] = t214;
  J[4 * 6 + 1] = t234;
  J[4 * 6 + 2] = t234;
  J[4 * 6 + 3] = t234;
  J[4 * 6 + 4] = ((t214) * (t204) + (((t210) * (t199) + ((t212 * t202)))));
  J[4 * 6 + 5] = ((t214) * (t203) + (((t210) * (t198) + ((t212 * t201)))));
  J[5 * 6 + 0] = t203;
  J[5 * 6 + 1] = t235;
  J[5 * 6 + 2] = t235;
  J[5 * 6 + 3] = t235;
  J[5 * 6 + 4] = ((t203) * (t204) + (((t198) * (t199) + ((t201 * t202)))));
  J[5 * 6 + 5] = ((t203) * (t203) + (((t198) * (t198) + ((t201 * t201)))));
  }
}

}  // namespace ur3e
}  // namespace cs179::gpu
