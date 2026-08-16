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

import com.shiyu.handwritten.runtime.HCCRRecognizer;

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
            HCCRRecognizer.loadNativeLibrary("hccr_jni");
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
                    (r.length > 0 ? " top1=" + r[0].text + " " + r[0].probability : ""));
            updateCandidates(r);
            if (r.length > 0) {
                status.setText(getString(R.string.status_top1, r[0].text, r[0].probability * 100, ms));
            }
        } catch (Throwable e) {
            Log.e(TAG, "recognize failed", e);
            status.setText("识别失败:" + e.getMessage());
        }
    }

    /**
     * Bitmap (RGBA) → float[64*64]。Java 端只做 ARGB → 灰度,
     * 后面的 bbox / bilinear / 翻转 / 归一化全部走 ★src/cpp/preprocess.c★
     * (跟 Python ctypes 同一份源码,字节级一致)。
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
        float[] out = new float[64 * 64];
        recognizer.preprocess(gray, w, h, out);
        return out;
    }


    private void updateCandidates(HCCRRecognizer.Result[] results) {
        for (int i = 0; i < TOPK; i++) {
            if (i < results.length) {
                candButtons[i].setText(results[i].text);
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
