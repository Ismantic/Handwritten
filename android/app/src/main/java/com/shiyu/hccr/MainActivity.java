package com.shiyu.hccr;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;

public class MainActivity extends Activity implements HandwritingView.StrokeListener {

    private static final String TAG = "HCCR";
    private static final String NCNN_PARAM = "mbv2_aug.int8.ncnn.param";
    private static final String NCNN_BIN   = "mbv2_aug.int8.ncnn.bin";
    private static final String CHARSET    = "charset.json";
    private static final int TOPK = 10;
    private static final long RECOGNIZE_DELAY_MS = 150;  // 抬笔后多久触发识别(跟 Python demo 一致)

    private HandwritingView handwriting;
    private LinearLayout candidates;
    private TextView status;
    private HCCRRecognizer recognizer;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private Runnable pendingRecognize;
    private final TextView[] candButtons = new TextView[TOPK];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        handwriting = findViewById(R.id.handwriting);
        candidates = findViewById(R.id.candidates);
        status = findViewById(R.id.status);

        handwriting.setStrokeListener(this);

        Button clear = findViewById(R.id.btn_clear);
        clear.setOnClickListener(v -> {
            handwriting.clear();
            clearCandidates();
            status.setText(R.string.status_empty);
        });

        Button undo = findViewById(R.id.btn_undo);
        undo.setOnClickListener(v -> {
            handwriting.undoLastStroke();
            if (handwriting.isEmpty()) {
                clearCandidates();
                status.setText(R.string.status_empty);
            } else {
                scheduleRecognize();
            }
        });

        // 初始化候选按钮(从 layout 拿,layout 里要有 10 个 TextView)
        for (int i = 0; i < TOPK; i++) {
            int id = getResources().getIdentifier("cand_" + i, "id", getPackageName());
            candButtons[i] = findViewById(id);
            final int idx = i;
            candButtons[i].setOnClickListener(v -> onPickCandidate(idx));
        }

        // 加载模型
        try {
            recognizer = new HCCRRecognizer(getAssets(), NCNN_PARAM, NCNN_BIN, CHARSET);
            status.setText(R.string.status_ready);
        } catch (IOException e) {
            Log.e(TAG, "model load failed", e);
            status.setText("模型加载失败:" + e.getMessage());
            Toast.makeText(this, "模型加载失败", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onDestroy() {
        if (recognizer != null) recognizer.close();
        super.onDestroy();
    }

    @Override
    public void onStrokeStart() {
        Log.i(TAG, "onStrokeStart");
        if (pendingRecognize != null) {
            handler.removeCallbacks(pendingRecognize);
            pendingRecognize = null;
        }
    }

    @Override
    public void onStrokeEnd() {
        Log.i(TAG, "onStrokeEnd, scheduleRecognize");
        scheduleRecognize();
    }

    private void scheduleRecognize() {
        if (pendingRecognize != null) handler.removeCallbacks(pendingRecognize);
        pendingRecognize = this::recognize;
        handler.postDelayed(pendingRecognize, RECOGNIZE_DELAY_MS);
    }

    private void recognize() {
        Log.i(TAG, "recognize() entered");
        pendingRecognize = null;
        if (recognizer == null) {
            Log.w(TAG, "recognizer is null!");
            return;
        }
        if (handwriting.isEmpty()) {
            Log.i(TAG, "handwriting empty, skip");
            clearCandidates();
            status.setText(R.string.status_empty);
            return;
        }
        try {
            Bitmap bmp = handwriting.renderToBitmap();
            Log.i(TAG, "bitmap " + bmp.getWidth() + "x" + bmp.getHeight());
            float[] input = preprocessToFloat64x64(bmp);
            int nonzero = 0;
            for (float v : input) if (v > 0.01f) nonzero++;
            Log.i(TAG, "input non-zero pixels: " + nonzero + "/4096");

            // DEBUG:把 64x64 预处理结果存 PNG 到 cache,方便 adb pull 看模型实际输入
            try {
                Bitmap preview = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
                int[] px = new int[64 * 64];
                for (int i = 0; i < input.length; i++) {
                    // input 是 stroke=高 / bg=0,翻回来:bg=255 stroke=0 显示
                    int gray = 255 - Math.round(input[i] * 255);
                    if (gray < 0) gray = 0;
                    if (gray > 255) gray = 255;
                    px[i] = 0xff000000 | (gray << 16) | (gray << 8) | gray;
                }
                preview.setPixels(px, 0, 64, 0, 0, 64, 64);
                java.io.File cacheFile = new java.io.File(getCacheDir(), "last_input.png");
                try (java.io.FileOutputStream fos = new java.io.FileOutputStream(cacheFile)) {
                    preview.compress(Bitmap.CompressFormat.PNG, 100, fos);
                }
                preview.recycle();
                Log.i(TAG, "saved preview: " + cacheFile.getAbsolutePath());
            } catch (Throwable e) {
                Log.w(TAG, "preview save failed: " + e);
            }

            long t0 = System.nanoTime();
            HCCRRecognizer.Result[] r = recognizer.predict(input, TOPK);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            Log.i(TAG, "predict OK, n=" + r.length + " ms=" + ms +
                    (r.length > 0 ? " top1=" + r[0].ch + " " + r[0].prob : ""));
            updateCandidates(r);
            if (r.length > 0) {
                status.setText(getString(R.string.status_top1, r[0].ch, r[0].prob * 100, ms));
            }
        } catch (Throwable e) {
            Log.e(TAG, "recognize failed", e);
            status.setText("识别失败:" + e.getMessage());
        }
    }

    /**
     * Bitmap (RGBA, 任意尺寸,白底黑笔)→ float[64*64]
     * 与训练时 dataset.py 的预处理对齐:
     *   1. 找笔画前景 bbox
     *   2. 长边等比缩到 56,居中放到 64×64(白底)
     *   3. 翻成 stroke=高:(255 - gray) / 255 → float
     *
     * 注意:这里复用 native 端的归一化逻辑会更精确(common.normalize 有 bbox + bilinear),
     * 但 native 接收 RGBA 输入还要再传一遍格式;先在 Java 里走 Bitmap.createScaledBitmap
     * 简单版本,精度够用(后期可下沉到 native 提高一致性)。
     */
    private float[] preprocessToFloat64x64(Bitmap src) {
        int w = src.getWidth();
        int h = src.getHeight();
        int[] argb = new int[w * h];
        src.getPixels(argb, 0, w, 0, 0, w, h);
        byte[] gray = new byte[w * h];
        for (int i = 0; i < argb.length; i++) {
            int c = argb[i];
            int r = (c >> 16) & 0xff;
            int g = (c >> 8) & 0xff;
            int b = c & 0xff;
            gray[i] = (byte) ((299 * r + 587 * g + 114 * b) / 1000);
        }

        int minX = w, minY = h, maxX = -1, maxY = -1;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int v = gray[y * w + x] & 0xff;
                if (v < 220) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (maxX < 0) return new float[64 * 64];

        // 裁 + ★area-average★ 缩长边 56(等价 cv2.INTER_AREA / PIL.BILINEAR-with-prefilter)
        // 不用 Bitmap.createScaledBitmap,那个是简单 2x2 bilinear,大缩放因子下产生硬边
        int cropW = maxX - minX + 1;
        int cropH = maxY - minY + 1;
        byte[] cropped = new byte[cropW * cropH];
        for (int y = 0; y < cropH; y++) {
            System.arraycopy(gray, (minY + y) * w + minX, cropped, y * cropW, cropW);
        }
        float scale = 56f / Math.max(cropW, cropH);
        int newW = Math.max(1, Math.round(cropW * scale));
        int newH = Math.max(1, Math.round(cropH * scale));
        // 用 PIL.Image.BILINEAR 等价实现(三角核 separable),严格跟 Python normalize 一致
        byte[] resized = resizePilBilinear(cropped, cropW, cropH, newW, newH);

        // 居中放到 64×64 白底
        byte[] canvas = new byte[64 * 64];
        java.util.Arrays.fill(canvas, (byte) 255);
        int offX = (64 - newW) / 2;
        int offY = (64 - newH) / 2;
        for (int y = 0; y < newH; y++) {
            System.arraycopy(resized, y * newW, canvas, (offY + y) * 64 + offX, newW);
        }

        // → float[1, 64, 64],翻转 + 归一化
        float[] out = new float[64 * 64];
        for (int i = 0; i < canvas.length; i++) {
            int v = canvas[i] & 0xff;
            out[i] = (255 - v) / 255f;
        }
        return out;
    }

    /**
     * PIL.Image.BILINEAR 等价实现(separable 三角核重采样)。
     *
     * PIL 源码(Pillow/src/libImaging/Resample.c):
     *   - 横竖两 pass(separable),每个 pass:
     *     output 像素 i 的中心(input 坐标)= (i+0.5) * scale
     *     kernel 半宽 support = max(1, scale)(downscale 时拉宽)
     *     权重 w(d) = max(0, 1 - |d| / max(1, scale)),三角核
     *     normalize w 之和为 1
     *
     * 我们之前用的 area-avg 是 box 核(覆盖 scale 宽),PIL 的三角核覆盖 2×scale 宽 →
     * 边缘像素受更宽邻域影响,产生更长的渐变带。
     */
    private static byte[] resizePilBilinear(byte[] src, int srcW, int srcH, int dstW, int dstH) {
        byte[] tmp = new byte[dstW * srcH];
        resamplePilPass(src, srcW, srcH, tmp, dstW, srcH, true);
        byte[] dst = new byte[dstW * dstH];
        resamplePilPass(tmp, dstW, srcH, dst, dstW, dstH, false);
        return dst;
    }

    private static void resamplePilPass(
        byte[] src, int srcW, int srcH,
        byte[] dst, int dstW, int dstH,
        boolean horizontal
    ) {
        int outDim = horizontal ? dstW : dstH;
        int inDim = horizontal ? srcW : srcH;
        float scale = (float) inDim / outDim;
        float filterscale = Math.max(1.0f, scale);
        float invFilterscale = 1.0f / filterscale;

        int[] xMin = new int[outDim];
        int[] xMax = new int[outDim];
        float[][] weights = new float[outDim][];
        for (int xx = 0; xx < outDim; xx++) {
            float center = (xx + 0.5f) * scale;
            int xmin = Math.max(0, (int) (center - filterscale + 0.5f));
            int xmax = Math.min(inDim, (int) (center + filterscale + 0.5f));
            int len = Math.max(1, xmax - xmin);
            float[] w = new float[len];
            float sum = 0;
            for (int x = 0; x < len; x++) {
                float d = ((xmin + x) + 0.5f - center) * invFilterscale;
                float wv = Math.max(0f, 1.0f - Math.abs(d));
                w[x] = wv;
                sum += wv;
            }
            if (sum > 0) for (int x = 0; x < len; x++) w[x] /= sum;
            xMin[xx] = xmin;
            xMax[xx] = xmax;
            weights[xx] = w;
        }

        for (int y = 0; y < dstH; y++) {
            for (int xx = 0; xx < dstW; xx++) {
                int outIdx = horizontal ? xx : y;
                int srcRow = horizontal ? y : xx;
                int xmin = xMin[outIdx];
                int xmax = xMax[outIdx];
                float[] w = weights[outIdx];
                float sum = 0;
                for (int x = 0; x < (xmax - xmin); x++) {
                    int srcVal = horizontal
                        ? (src[srcRow * srcW + xmin + x] & 0xff)
                        : (src[(xmin + x) * srcW + srcRow] & 0xff);
                    sum += srcVal * w[x];
                }
                int v = Math.round(sum);
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                dst[y * dstW + xx] = (byte) v;
            }
        }
    }

    private void updateCandidates(HCCRRecognizer.Result[] results) {
        for (int i = 0; i < TOPK; i++) {
            if (i < results.length) {
                candButtons[i].setText(results[i].ch);
                candButtons[i].setEnabled(true);
                candButtons[i].setAlpha(1.0f);
            } else {
                candButtons[i].setText("");
                candButtons[i].setEnabled(false);
                candButtons[i].setAlpha(0.3f);
            }
        }
    }

    private void clearCandidates() {
        for (TextView b : candButtons) {
            b.setText("");
            b.setEnabled(false);
            b.setAlpha(0.3f);
        }
    }

    private void onPickCandidate(int i) {
        CharSequence ch = candButtons[i].getText();
        if (ch == null || ch.length() == 0) return;
        Toast.makeText(this, "选中:" + ch, Toast.LENGTH_SHORT).show();
        handwriting.clear();
        clearCandidates();
        status.setText(R.string.status_ready);
    }
}
