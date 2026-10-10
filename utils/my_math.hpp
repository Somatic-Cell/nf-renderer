#ifndef MY_MATH_HPP_
#define MY_MATH_HPP_

#include <cuda_runtime.h>
#include "helper_math.h"
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>


namespace mymath{
    inline __device__ __host__ void rotate(const float3 _u, const float theta, float3& vx, float3& vy, float3& vz)
  {
    float3 u = normalize(_u);
    float s = sin(theta);
    float c = cos(theta);
    const float3 rx = make_float3(u.x * u.x + (1.f - u.x * u.x) * c,   u.x * u.y * (1.f - c) - u.z * s,    u.x * u.z * (1.f - c) + u.y * s);
    const float3 ry = make_float3(u.x * u.y * (1.f - c) + u.z * s,     u.y * u.y + (1.f - u.y * u.y) * c,  u.y * u.z * (1.f - c) - u.x * s);
    const float3 rz = make_float3(u.x * u.z * (1.f - c) * u.y * s,     u.y * u.z * (1.f - c) + u.x * s,    u.z * u.z + (1.f - u.z * u.z) * c);

    // | rx.x  ry.x  rz.x |   | vx.x  vy.x  vz.x |
    // | rx.y  ry.y  rz.y | * | vx.y  vy.y  vz.y |
    // | rx.z  ry.z  rz.z |   | vx.z  vy.z  vz.z |
    // R * X
    const float3 tmpx = make_float3(rx.x * vx.x + ry.x * vx.y + rz.x * vx.z, rx.x * vy.x + ry.x * vy.y + rz.x * vy.z, rx.x * vz.x + ry.x * vz.y + rz.x * vz.z);
    const float3 tmpy = make_float3(rx.y * vx.x + ry.y * vx.y + rz.y * vx.z, rx.y * vy.x + ry.y * vy.y + rz.y * vy.z, rx.y * vz.x + ry.y * vz.y + rz.y * vz.z);
    const float3 tmpz = make_float3(rx.z * vx.x + ry.z * vx.y + rz.z * vx.z, rx.z * vy.x + ry.z * vy.y + rz.z * vy.z, rx.z * vz.x + ry.z * vz.y + rz.z * vz.z);

    vx = tmpx;
    vy = tmpy;
    vz = tmpz;
  }

  struct matrix3x4{
    float4 row0;
    float4 row1;
    float4 row2;
  };
  
  struct matrix3x3{
    float3 row0;
    float3 row1;
    float3 row2;
  };

  inline __device__ __host__ matrix3x3 linear3x3(const matrix3x4& M){
    matrix3x3 A;

    A.row0 = make_float3(M.row0.x, M.row0.y, M.row0.z);
    A.row1 = make_float3(M.row1.x, M.row1.y, M.row1.z);
    A.row2 = make_float3(M.row2.x, M.row2.y, M.row2.z);

    return A;
  }

  inline __device__ __host__ float3 mul3x3(const matrix3x3 M, const float3 x){
    float3 res;
    res.x = M.row0.x * x.x + M.row0.y * x.y + M.row0.z * x.z;
    res.y = M.row1.x * x.x + M.row1.y * x.y + M.row1.z * x.z;
    res.z = M.row2.x * x.x + M.row2.y * x.y + M.row2.z * x.z;
    return res;
  }

  inline __device__ __host__ float3 mul3x4(const matrix3x4 M, const float4 x){
    float3 res;
    res.x = M.row0.x * x.x + M.row0.y * x.y + M.row0.z * x.z + M.row0.w * x.w;
    res.y = M.row1.x * x.x + M.row1.y * x.y + M.row1.z * x.z + M.row1.w * x.w;
    res.z = M.row2.x * x.x + M.row2.y * x.y + M.row2.z * x.z + M.row2.w * x.w;
    return res;
  }

  inline __device__ __host__ matrix3x3 quatToR(const float4 qIn){
    float len = sqrtf(qIn.x * qIn.x + qIn.y * qIn.y + qIn.z * qIn.z + qIn.w * qIn.w);
    float4 q = qIn / len;

    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    
    float xy = q.x * q.y;
    float yz = q.y * q.z;
    float zx = q.z * q.x;
    
    float xw = q.x * q.w;
    float yw = q.y * q.w;
    float zw = q.z * q.w;

    matrix3x3 R;
    R.row0 = make_float3(1.0f - 2.0f * (yy + zz),        2.0f * (xy - zw),        2.0f * (zx + yw));
    R.row1 = make_float3(       2.0f * (xy + zw), 1.0f - 2.0f * (xx + zz),        2.0f * (yz - xw));
    R.row2 = make_float3(       2.0f * (zx - yw),        2.0f * (yz - xw), 1.0f - 2.0f * (xx + yy));
    return R;
  }

  inline __device__ __host__ matrix3x3 composeRS(const float4 q, const float3 s)
  {
    matrix3x3 R = quatToR(q);
    matrix3x3 RS;
    RS.row0 = make_float3(R.row0.x * s.x, R.row0.y * s.y, R.row0.z * s.z);
    RS.row1 = make_float3(R.row1.x * s.x, R.row1.y * s.y, R.row1.z * s.z);
    RS.row2 = make_float3(R.row2.x * s.x, R.row2.y * s.y, R.row2.z * s.z);
    return RS;
  }

  inline __device__ __host__ float3 aabbCorner(const float3 bBoxMin, const float3 bBoxMax, int i){
    return make_float3(
      (i & 1) ? bBoxMax.x : bBoxMin.x,
      (i & 2) ? bBoxMax.y : bBoxMin.y,
      (i & 4) ? bBoxMax.z : bBoxMin.z
    );
  }

  inline __device__ __host__ matrix3x4 makeInstanceMatrix(
    const float3 t,
    const float4 q,
    const float3 s,
    const float3 bBoxCenter,
    const float3 bBoxMin,
    bool centerBBoxAtOrigin,
    bool placeOnGroundY,
    float groundY = 0.0f
  ) {
    // 回転とスケール行列
    matrix3x3 RS = composeRS(q, s);

    // 中心へのシフト
    float3 bBoxCenterInWorld = mul3x3(RS, bBoxCenter);
    float3 dCenter = centerBBoxAtOrigin ? -bBoxCenterInWorld : make_float3(0.0f);

    // 地面の処理
    float3 bBoxMax = bBoxMin + 2.0f * bBoxCenter;
    float minY = 1e30f;
    for(int i = 0; i < 8; ++i){
      float3 pw = mul3x3(RS, aabbCorner(bBoxMin, bBoxMax, i)) + dCenter;
      if(pw.y < minY) minY = pw.y;
    }
    float3 dGround = placeOnGroundY ? make_float3(0.0f, groundY - minY, 0.0f) : make_float3(0.0f);
    
    float3 T = t + dCenter + dGround;

    matrix3x4 M;
    M.row0 = make_float4(RS.row0, T.x);
    M.row1 = make_float4(RS.row1, T.y);
    M.row2 = make_float4(RS.row2, T.z);
    return M;
  }

}
#endif // MY_MATH_HPP_