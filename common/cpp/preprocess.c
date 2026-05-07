#include "preprocess.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


/* ============== helpers ============== */

static int   imax_i(int a, int b)     { return a > b ? a : b; }
static int   imin_i(int a, int b)     { return a < b ? a : b; }
static float fmax_f(float a, float b) { return a > b ? a : b; }
static float fabs_f(float x)          { return x < 0 ? -x : x; }
static int   iround_f(float x) {
    /* round-half-up,跟 Python round 行为接近 */
    return (int)(x + (x >= 0.0f ? 0.5f : -0.5f));
}


/* ============== separable bilinear pass(等价 PIL.Image.BILINEAR) ==============
 *
 * 对 in_dim → out_dim 重采样。三角核(linear 衰减)kernel 半宽 = max(1, scale),
 * downscale 时拉宽,upscale 时固定 1。等价 PIL Resample.c::ImagingResample。
 *
 * horizontal=1:把 (srcW × srcH) 缩到 (dstW × srcH);
 * horizontal=0:把 (srcW × srcH) 缩到 (srcW × dstH)。
 */
static void resample_pass_(
    const uint8_t* src, int srcW, int srcH,
    uint8_t* dst, int dstW, int dstH,
    int horizontal
) {
    int   out_dim       = horizontal ? dstW : dstH;
    int   in_dim        = horizontal ? srcW : srcH;
    float scale         = (float) in_dim / (float) out_dim;
    float filterscale   = scale > 1.0f ? scale : 1.0f;
    float inv_filterscl = 1.0f / filterscale;

    /* 预算 (xmin, xmax, weights),内层循环只 weighted-sum,
     * 完全按 Pillow ImagingResample 的 calc 流程。 */
    int*    xmin    = (int*)   malloc(sizeof(int)   * (size_t) out_dim);
    int*    xmax    = (int*)   malloc(sizeof(int)   * (size_t) out_dim);
    int     max_k   = (int)(2.0f * filterscale + 2.0f);  /* 上界,实际可能更小 */
    float*  weights = (float*) malloc(sizeof(float) * (size_t) out_dim * (size_t) max_k);

    for (int xx = 0; xx < out_dim; xx++) {
        float center = ((float) xx + 0.5f) * scale;
        int   xm     = imax_i(0,      (int)(center - filterscale + 0.5f));
        int   xM     = imin_i(in_dim, (int)(center + filterscale + 0.5f));
        if (xM <= xm) xM = imin_i(in_dim, xm + 1);
        int     len = xM - xm;
        float*  w   = weights + (size_t) xx * (size_t) max_k;
        float   sum = 0.0f;
        for (int x = 0; x < len; x++) {
            float d  = (((float)(xm + x)) + 0.5f - center) * inv_filterscl;
            float wv = fmax_f(0.0f, 1.0f - fabs_f(d));
            w[x]     = wv;
            sum     += wv;
        }
        if (sum > 0) {
            for (int x = 0; x < len; x++) w[x] /= sum;
        }
        xmin[xx] = xm;
        xmax[xx] = xM;
    }

    for (int y = 0; y < dstH; y++) {
        for (int xx = 0; xx < dstW; xx++) {
            int    out_idx = horizontal ? xx : y;
            int    src_row = horizontal ? y  : xx;
            int    xm      = xmin[out_idx];
            int    xM      = xmax[out_idx];
            const float* w = weights + (size_t) out_idx * (size_t) max_k;
            float sum = 0.0f;
            for (int x = 0; x < (xM - xm); x++) {
                int v;
                if (horizontal) v = src[src_row * srcW + xm + x];
                else            v = src[(xm + x) * srcW + src_row];
                sum += (float) v * w[x];
            }
            int v = iround_f(sum);
            if (v < 0)   v = 0;
            if (v > 255) v = 255;
            dst[y * dstW + xx] = (uint8_t) v;
        }
    }

    free(xmin); free(xmax); free(weights);
}


/* ============== 主入口 ============== */

int hccr_preprocess(const uint8_t* gray, int w, int h, float* out) {
    /* 1. 找前景 bbox(像素 < 220) */
    int min_x = w, min_y = h, max_x = -1, max_y = -1;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int v = gray[y * w + x];
            if (v < HCCR_FG_THRESHOLD) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }
    if (max_x < 0) {
        memset(out, 0, sizeof(float) * (size_t) HCCR_CANVAS_SIZE * (size_t) HCCR_CANVAS_SIZE);
        return 0;
    }

    /* 2. crop bbox 到 cropped */
    int cropW = max_x - min_x + 1;
    int cropH = max_y - min_y + 1;
    uint8_t* cropped = (uint8_t*) malloc((size_t) cropW * (size_t) cropH);
    for (int y = 0; y < cropH; y++) {
        memcpy(cropped + (size_t) y * (size_t) cropW,
               gray + (size_t)(min_y + y) * (size_t) w + (size_t) min_x,
               (size_t) cropW);
    }

    /* 3. 长边缩到 56,separable bilinear */
    int   max_dim = imax_i(cropW, cropH);
    float scale   = (float) HCCR_CONTENT_SIZE / (float) max_dim;
    int   newW    = imax_i(1, iround_f((float) cropW * scale));
    int   newH    = imax_i(1, iround_f((float) cropH * scale));

    uint8_t* tmp     = (uint8_t*) malloc((size_t) newW * (size_t) cropH);
    resample_pass_(cropped, cropW, cropH, tmp, newW, cropH, 1);
    uint8_t* resized = (uint8_t*) malloc((size_t) newW * (size_t) newH);
    resample_pass_(tmp, newW, cropH, resized, newW, newH, 0);

    /* 4. 居中放到 64×64 白底 */
    uint8_t canvas[HCCR_CANVAS_SIZE * HCCR_CANVAS_SIZE];
    memset(canvas, 255, sizeof(canvas));
    int off_x = (HCCR_CANVAS_SIZE - newW) / 2;
    int off_y = (HCCR_CANVAS_SIZE - newH) / 2;
    for (int y = 0; y < newH; y++) {
        memcpy(canvas + (size_t)(off_y + y) * HCCR_CANVAS_SIZE + (size_t) off_x,
               resized + (size_t) y * (size_t) newW,
               (size_t) newW);
    }

    /* 5. 翻转 + 归一化 → float[1, 64, 64] */
    int total = HCCR_CANVAS_SIZE * HCCR_CANVAS_SIZE;
    for (int i = 0; i < total; i++) {
        out[i] = (float)(255 - canvas[i]) / 255.0f;
    }

    free(cropped); free(tmp); free(resized);
    return 1;
}
