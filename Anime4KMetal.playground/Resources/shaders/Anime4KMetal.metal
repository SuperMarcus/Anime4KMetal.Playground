//#pragma once
#include <metal_stdlib>
using namespace metal;

constant float factor [[function_constant(0)]];
constant uint  inW    [[function_constant(1)]];
constant uint  inH    [[function_constant(2)]];
constant uint  outW   [[function_constant(3)]];
constant uint  outH   [[function_constant(4)]];
constant uint  outP   [[function_constant(5)]];
constant float bold   [[function_constant(6)]];
constant float blur   [[function_constant(7)]];

#define INPUT_SIZE uint2(inW, inH)
#define OUTPUT_SIZE uint2(outW, outH)
#define strength (min(factor / bold, 1.0))
#define PIXEL_SIZE (1.0f / float2(outW, outH))

#define HOOKED_tex(p) readPixel(hooked, p)
#define POSTKERNEL_tex(p) readPixel(postkernel, p)

/// Clamp the pixel coordinate to a given bound
uint2 normPixelCoord(uint2 input, uint2 bounds) {
    return clamp(input, 0, bounds - 1);
}

/// Convert uv (floating point coord) to absulute coord
uint2 uv2coord(float2 uv, uint2 bounds) {
    return uint2(round(uv * float2(bounds - 1)));
}

/// Convert xy coords to uv
float2 coord2uv(uint2 xy, uint2 bounds) {
    uint2 normalizedCoord = normPixelCoord(xy, bounds);
    return float2(
        float(normalizedCoord.x) / float(bounds.x - 1),
        float(normalizedCoord.y) / float(bounds.y - 1)
    );
}

/// Read the pixel given the uv coord
float4 readPixelSrc(texture2d<float, access::read> in, float2 uv) {
    return in.read(normPixelCoord(uv2coord(uv, INPUT_SIZE), INPUT_SIZE));
}

/// Read the pixel given the uv coord
float4 readPixel(texture2d<float, access::read> in, float2 uv) {
    return in.read(normPixelCoord(uv2coord(uv, OUTPUT_SIZE), OUTPUT_SIZE));
}

float luminance(float4 pixel) {
    return (pixel.r + pixel.r + pixel.g + pixel.g + pixel.g + pixel.b) / 6.0;
}

float4 GetPixelClamped(texture2d<float, access::read> in, uint x, uint y) {
    return in.read(normPixelCoord(uint2(x, y), INPUT_SIZE));
}

// t is a value that goes from 0 to 1 to interpolate in a C1 continuous way across uniformly sampled data points.
// when t is 0, this will return B.  When t is 1, this will return C.  Inbetween values will return an interpolation
// between B and C.  A and B are used to calculate slopes at the edges.
float CubicHermite(float A, float B, float C, float D, float t) {
    float a = -A / 2.0f + (3.0f * B) / 2.0f - (3.0f * C) / 2.0f + D / 2.0f;
    float b = A - (5.0f * B) / 2.0f + 2.0f * C - D / 2.0f;
    float c = -A / 2.0f + C / 2.0f;
    float d = B;
    
    return a * t * t * t + b * t * t + c * t + d;
}

float4 SampleBicubic(texture2d<float, access::read> in [[texture(0)]], float u, float v) {
    // calculate coordinates -> also need to offset by half a pixel to keep image from shifting down and left half a pixel
    float x = u * float(inW) - 0.5;
    int xint = int(x);
    float xfract = x - floor(x);
    
    float y = v * float(inH) - 0.5;
    int yint = int(y);
    float yfract = y - floor(y);
    
    // 1st row
    auto p00 = GetPixelClamped(in, xint - 1, yint - 1);
    auto p10 = GetPixelClamped(in, xint + 0, yint - 1);
    auto p20 = GetPixelClamped(in, xint + 1, yint - 1);
    auto p30 = GetPixelClamped(in, xint + 2, yint - 1);
    
    // 2nd row
    auto p01 = GetPixelClamped(in, xint - 1, yint + 0);
    auto p11 = GetPixelClamped(in, xint + 0, yint + 0);
    auto p21 = GetPixelClamped(in, xint + 1, yint + 0);
    auto p31 = GetPixelClamped(in, xint + 2, yint + 0);
    
    // 3rd row
    auto p02 = GetPixelClamped(in, xint - 1, yint + 1);
    auto p12 = GetPixelClamped(in, xint + 0, yint + 1);
    auto p22 = GetPixelClamped(in, xint + 1, yint + 1);
    auto p32 = GetPixelClamped(in, xint + 2, yint + 1);
    
    // 4th row
    auto p03 = GetPixelClamped(in, xint - 1, yint + 2);
    auto p13 = GetPixelClamped(in, xint + 0, yint + 2);
    auto p23 = GetPixelClamped(in, xint + 1, yint + 2);
    auto p33 = GetPixelClamped(in, xint + 2, yint + 2);
    
    // interpolate bi-cubically!
    // Clamp the values since the curve can put the value below 0 or above 255
    float4 ret;
    for (int i = 0; i < 4; ++i)
    {
        float col0 = CubicHermite(p00[i], p10[i], p20[i], p30[i], xfract);
        float col1 = CubicHermite(p01[i], p11[i], p21[i], p31[i], xfract);
        float col2 = CubicHermite(p02[i], p12[i], p22[i], p32[i], xfract);
        float col3 = CubicHermite(p03[i], p13[i], p23[i], p33[i], xfract);
        float value = clamp(CubicHermite(col0, col1, col2, col3, yfract), 0.f, 1.f);
        ret[i] = value;
    }
    return ret;
}

float4 getLargest(float4 cc, float4 lightestColor, float4 a, float4 b, float4 c) {
    float4 newColor = cc * (1.0 - strength) + ((a + b + c) / 3.0) * strength;
    if (newColor.a > lightestColor.a) {
        return newColor;
    }
    return lightestColor;
}

float4 getAverage(float4 cc, float4 a, float4 b, float4 c) {
    return cc * (1.0 - strength) + ((a + b + c) / 3.0) * strength;
}

float4 getRGBL(texture2d<float, access::read> hooked,
               texture2d<float, access::read> postkernel,
               float2 pos) {
    return float4(HOOKED_tex(pos).rgb, POSTKERNEL_tex(pos).x);
}

float getLum(texture2d<float, access::read> lumTex, float2 pos) {
    return readPixel(lumTex, pos).x;
}

float min3v(float4 a, float4 b, float4 c) {
    return min(min(a.a, b.a), c.a);
}

float max3v(float4 a, float4 b, float4 c) {
    return max(max(a.a, b.a), c.a);
}

// MARK: - Kernels

/// Kernal for Bicubic Scaling
kernel void ScaleMain(texture2d<float, access::read> in  [[texture(0)]],
                        texture2d<float, access::write> out [[texture(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
//    float2 uv = coord2uv(gid, OUTPUT_SIZE);
//    out.write(interpolate(in, uv), gid);
//    out.write(readPixelSrc(in, uv), gid);
    float v = float(gid.y) / float(outH - 1);
    float u = float(gid.x) / float(outW - 1);
    out.write(SampleBicubic(in, u, v), gid);
    out.write(GetPixelClamped(in, uint(u * inW), uint(v * inH)), gid);
}

/// Kernel for calculating luminance
kernel void LumMain(texture2d<float, access::read> in  [[texture(0)]],
                    texture2d<float, access::write> out [[texture(1)]],
                    uint2 gid [[thread_position_in_grid]]) {
    float2 uv = coord2uv(gid, OUTPUT_SIZE);
    out.write(luminance(readPixel(in, uv)), gid);
//    out.write(readPixel(in, uv), gid);
}

/// Kernel for push
kernel void PushMain(texture2d<float, access::read> hooked [[texture(0)]],
                     texture2d<float, access::read> postkernel [[texture(1)]],
                     texture2d<float, access::write> out [[texture(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    float2 HOOKED_pos = coord2uv(gid, OUTPUT_SIZE);
    float2 d = PIXEL_SIZE;

    float4 cc = getRGBL(hooked, postkernel, HOOKED_pos);
    float4 t = getRGBL(hooked, postkernel, HOOKED_pos + float2(0.0f, -d.y));
    float4 tl = getRGBL(hooked, postkernel, HOOKED_pos + float2(-d.x, -d.y));
    float4 tr = getRGBL(hooked, postkernel, HOOKED_pos + float2(d.x, -d.y));

    float4 l = getRGBL(hooked, postkernel, HOOKED_pos + float2(-d.x, 0.0f));
    float4 r = getRGBL(hooked, postkernel, HOOKED_pos + float2(d.x, 0.0f));

    float4 b = getRGBL(hooked, postkernel, HOOKED_pos + float2(0.0f, d.y));
    float4 bl = getRGBL(hooked, postkernel, HOOKED_pos + float2(-d.x, d.y));
    float4 br = getRGBL(hooked, postkernel, HOOKED_pos + float2(d.x, d.y));

    float4 lightestColor = cc;
    //Kernel 0 and 4
    float maxDark = max3v(br, b, bl);
    float minLight = min3v(tl, t, tr);

    if (minLight > cc.a && minLight > maxDark) {
        lightestColor = getLargest(cc, lightestColor, tl, t, tr);
    } else {
        maxDark = max3v(tl, t, tr);
        minLight = min3v(br, b, bl);
        if (minLight > cc.a && minLight > maxDark) {
            lightestColor = getLargest(cc, lightestColor, br, b, bl);
        }
    }

    //Kernel 1 and 5
    maxDark = max3v(cc, l, b);
    minLight = min3v(r, t, tr);

    if (minLight > maxDark) {
        lightestColor = getLargest(cc, lightestColor, r, t, tr);
    } else {
        maxDark = max3v(cc, r, t);
        minLight = min3v(bl, l, b);
        if (minLight > maxDark) {
            lightestColor = getLargest(cc, lightestColor, bl, l, b);
        }
    }

    //Kernel 2 and 6
    maxDark = max3v(l, tl, bl);
    minLight = min3v(r, br, tr);

    if (minLight > cc.a && minLight > maxDark) {
        lightestColor = getLargest(cc, lightestColor, r, br, tr);
    } else {
        maxDark = max3v(r, br, tr);
        minLight = min3v(l, tl, bl);
        if (minLight > cc.a && minLight > maxDark) {
            lightestColor = getLargest(cc, lightestColor, l, tl, bl);
        }
    }

    //Kernel 3 and 7
    maxDark = max3v(cc, l, t);
    minLight = min3v(r, br, b);

    if (minLight > maxDark) {
        lightestColor = getLargest(cc, lightestColor, r, br, b);
    } else {
        maxDark = max3v(cc, r, b);
        minLight = min3v(t, l, tl);
        if (minLight > maxDark) {
            lightestColor = getLargest(cc, lightestColor, t, l, tl);
        }
    }

    out.write(lightestColor, gid);
}

/// Kernel for Grad
kernel void GradMain(texture2d<float, access::read> hooked [[texture(0)]],
                     texture2d<float, access::read> postkernel [[texture(1)]],
                     texture2d<float, access::write> out [[texture(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    float2 HOOKED_pos = coord2uv(gid, OUTPUT_SIZE);
    float2 d = PIXEL_SIZE;
    
    // [tl  t tr]
    // [ l cc  r]
    // [bl  b br]
    float cc = getLum(postkernel, HOOKED_pos);
    float t = getLum(postkernel, HOOKED_pos + float2(0.0f, -d.y));
    float tl = getLum(postkernel, HOOKED_pos + float2(-d.x, -d.y));
    float tr = getLum(postkernel, HOOKED_pos + float2(d.x, -d.y));
    
    float l = getLum(postkernel, HOOKED_pos + float2(-d.x, 0.0f));
    float r = getLum(postkernel, HOOKED_pos + float2(d.x, 0.0f));
    
    float b = getLum(postkernel, HOOKED_pos + float2(0.0f, d.y));
    float bl = getLum(postkernel, HOOKED_pos + float2(-d.x, d.y));
    float br = getLum(postkernel, HOOKED_pos + float2(d.x, d.y));
    
    // Horizontal Gradient
    // [-1  0  1]
    // [-2  0  2]
    // [-1  0  1]
    float xgrad = (-tl + tr - l - l + r + r - bl + br);
    
    // Vertical Gradient
    // [-1 -2 -1]
    // [ 0  0  0]
    // [ 1  2  1]
    float ygrad = (-tl - t - t - tr + bl + b + b + br);
    
//    auto grad = float4(1.0 - clamp(sqrt(xgrad * xgrad + ygrad * ygrad), 0.0, 1.0));
//    out.write(float4(HOOKED_pos, 0, 1), gid);
    auto result = float4(1.0 - clamp(sqrt(xgrad * xgrad + ygrad * ygrad), 0.0, 1.0));
    out.write(result, gid);
//    out.write(getRGBL(hooked, postkernel, HOOKED_pos + d), gid);
}

void _finalize(texture2d<float, access::write> out,
               texture2d<float, access::read> scaled,
               float2 uv,
               float4 result,
               uint2 gid) {
    float4 finalOut = result;
    finalOut.a = readPixel(scaled, uv).a;
    out.write(finalOut, gid);
}

/// Kernel for Final
kernel void FinalMain(texture2d<float, access::read> hooked [[texture(0)]],
                     texture2d<float, access::read> postkernel [[texture(1)]],
                     texture2d<float, access::read> scaled [[texture(2)]],
                     texture2d<float, access::write> out [[texture(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    float2 HOOKED_pos = coord2uv(gid, OUTPUT_SIZE);
    float2 d = PIXEL_SIZE;
    
    float4 cc = getRGBL(hooked, postkernel, HOOKED_pos);
    float4 t = getRGBL(hooked, postkernel, HOOKED_pos + float2(0.0f, -d.y));
    float4 tl = getRGBL(hooked, postkernel, HOOKED_pos + float2(-d.x, -d.y));
    float4 tr = getRGBL(hooked, postkernel, HOOKED_pos + float2(d.x, -d.y));
    
    float4 l = getRGBL(hooked, postkernel, HOOKED_pos + float2(-d.x, 0.0f));
    float4 r = getRGBL(hooked, postkernel, HOOKED_pos + float2(d.x, 0.0f));
    
    float4 b = getRGBL(hooked, postkernel, HOOKED_pos + float2(0.0f, d.y));
    float4 bl = getRGBL(hooked, postkernel, HOOKED_pos + float2(-d.x, d.y));
    float4 br = getRGBL(hooked, postkernel, HOOKED_pos + float2(d.x, d.y));
    
    //Kernel 0 and 4
    float maxDark = max3v(br, b, bl);
    float minLight = min3v(tl, t, tr);
    
    if (minLight > cc.a && minLight > maxDark) {
        _finalize(out, scaled, HOOKED_pos, getAverage(cc, tl, t, tr), gid);
        return;
    } else {
        maxDark = max3v(tl, t, tr);
        minLight = min3v(br, b, bl);
        if (minLight > cc.a && minLight > maxDark) {
            _finalize(out, scaled, HOOKED_pos, getAverage(cc, br, b, bl), gid);
            return;
        }
    }
    
    //Kernel 1 and 5
    maxDark = max3v(cc, l, b);
    minLight = min3v(r, t, tr);
    
    if (minLight > maxDark) {
        _finalize(out, scaled, HOOKED_pos, getAverage(cc, r, t, tr), gid);
        return;
    } else {
        maxDark = max3v(cc, r, t);
        minLight = min3v(bl, l, b);
        if (minLight > maxDark) {
            _finalize(out, scaled, HOOKED_pos, getAverage(cc, bl, l, b), gid);
            return;
        }
    }
    
    //Kernel 2 and 6
    maxDark = max3v(l, tl, bl);
    minLight = min3v(r, br, tr);
    
    if (minLight > cc.a && minLight > maxDark) {
        _finalize(out, scaled, HOOKED_pos, getAverage(cc, r, br, tr), gid);
        return;
    } else {
        maxDark = max3v(r, br, tr);
        minLight = min3v(l, tl, bl);
        if (minLight > cc.a && minLight > maxDark) {
            _finalize(out, scaled, HOOKED_pos, getAverage(cc, l, tl, bl), gid);
            return;
        }
    }
    
    //Kernel 3 and 7
    maxDark = max3v(cc, l, t);
    minLight = min3v(r, br, b);
    
    if (minLight > maxDark) {
        _finalize(out, scaled, HOOKED_pos, getAverage(cc, r, br, b), gid);
        return;
    } else {
        maxDark = max3v(cc, r, b);
        minLight = min3v(t, l, tl);
        if (minLight > maxDark) {
            _finalize(out, scaled, HOOKED_pos, getAverage(cc, t, l, tl), gid);
            return;
        }
    }
    
    _finalize(out, scaled, HOOKED_pos, cc, gid);
}
