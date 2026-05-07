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
    private static final long RECOGNIZE_DELAY_MS = 200;  // 抬笔后多久触发识别

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
        if (pendingRecognize != null) {
            handler.removeCallbacks(pendingRecognize);
            pendingRecognize = null;
        }
    }

    @Override
    public void onStrokeEnd() {
        scheduleRecognize();
    }

    private void scheduleRecognize() {
        if (pendingRecognize != null) handler.removeCallbacks(pendingRecognize);
        pendingRecognize = this::recognize;
        handler.postDelayed(pendingRecognize, RECOGNIZE_DELAY_MS);
    }

    private void recognize() {
        pendingRecognize = null;
        if (recognizer == null) return;
        if (handwriting.isEmpty()) {
            clearCandidates();
            status.setText(R.string.status_empty);
            return;
        }
        try {
            Bitmap bmp = handwriting.renderToBitmap();
            float[] input = preprocessToFloat64x64(bmp);
            long t0 = System.nanoTime();
            HCCRRecognizer.Result[] r = recognizer.predict(input, TOPK);
            long ms = (System.nanoTime() - t0) / 1_000_000;
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
        // 1. 转灰度并提取像素
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
            // luminance
            int y = (299 * r + 587 * g + 114 * b) / 1000;
            gray[i] = (byte) y;
        }

        // 2. 找前景 bbox(像素 < 220 算笔画;白底 ≈ 255)
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
        if (maxX < 0) {
            // 全白,返回全 0(stroke 空)
            return new float[64 * 64];
        }

        // 3. 裁切 + 等比缩 long_edge=56 + 居中放到 64×64 白底
        int cropW = maxX - minX + 1;
        int cropH = maxY - minY + 1;
        Bitmap cropped = Bitmap.createBitmap(src, minX, minY, cropW, cropH);
        float scale = 56f / Math.max(cropW, cropH);
        int newW = Math.max(1, Math.round(cropW * scale));
        int newH = Math.max(1, Math.round(cropH * scale));
        Bitmap resized = Bitmap.createScaledBitmap(cropped, newW, newH, true);

        // 64×64 白底
        Bitmap canvas = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
        canvas.eraseColor(Color.WHITE);
        int offX = (64 - newW) / 2;
        int offY = (64 - newH) / 2;
        android.graphics.Canvas c = new android.graphics.Canvas(canvas);
        c.drawBitmap(resized, offX, offY, null);

        // 4. → float[1, 64, 64],翻转(stroke=高,bg=0),归一化
        int[] px = new int[64 * 64];
        canvas.getPixels(px, 0, 64, 0, 0, 64, 64);
        float[] out = new float[64 * 64];
        for (int i = 0; i < px.length; i++) {
            int c2 = px[i];
            int r = (c2 >> 16) & 0xff;
            int g = (c2 >> 8) & 0xff;
            int b = c2 & 0xff;
            int y = (299 * r + 587 * g + 114 * b) / 1000;
            // (255 - y) / 255
            out[i] = (255 - y) / 255f;
        }

        // 释放临时 bitmap
        if (cropped != src) cropped.recycle();
        if (resized != cropped) resized.recycle();
        canvas.recycle();
        return out;
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
