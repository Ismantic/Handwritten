/* HCCR 预处理 —— 单一 C 实现,Python (ctypes) 和 Android (JNI) 共用,
 * 保证训练 / Python demo / Android 三端字节级一致。
 *
 * 算法(跟 common/normalize.py 等价,跟 PIL.Image.BILINEAR 同口径):
 *   1. 输入灰度图 (uint8, 255=白底, 0=黑笔画)
 *   2. 找前景 bbox(像素 < 220)
 *   3. 裁切 bbox,长边等比缩到 56,separable 三角核 bilinear(同 PIL)
 *   4. 居中放到 64×64 白底
 *   5. 翻转 + 归一化 → float[64*64], stroke=高 (跟 dataset.py 一致)
 */

#ifndef HCCR_PREPROCESS_H
#define HCCR_PREPROCESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HCCR_CANVAS_SIZE  64
#define HCCR_CONTENT_SIZE 56
#define HCCR_FG_THRESHOLD 220

/**
 * 把 (w x h) 灰度图预处理成模型 input。
 *
 * @param gray   row-major 灰度数据,长度 w*h(255=白,0=笔画)
 * @param w,h    输入尺寸
 * @param out    输出 float 数组,**调用方必须保证至少 64*64=4096 元素**
 *
 * @return 1 表示成功,0 表示输入全白(out 已置零,可直接喂模型,只是没东西识别)
 */
int hccr_preprocess(const uint8_t* gray, int w, int h, float* out);

#ifdef __cplusplus
}
#endif

#endif /* HCCR_PREPROCESS_H */
