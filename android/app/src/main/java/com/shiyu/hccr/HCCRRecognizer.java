package com.shiyu.hccr;

import android.content.res.AssetManager;
import android.util.Log;

import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * Java 包装类:封装 ncnn 推理 + charset 反查。
 * native 部分接收 64x64 float[](已经预处理:笔画=高,白底=0,值范围 [0,1])
 * 返回 top-K 的索引数组(int[K])和概率数组(float[K])。
 */
public class HCCRRecognizer {

    private static final String TAG = "HCCRRecognizer";

    static {
        System.loadLibrary("hccr_jni");
    }

    private long nativeHandle;
    private String[] idxToChar;

    public HCCRRecognizer(AssetManager am, String paramName, String binName, String charsetName) throws IOException {
        // 把 ncnn .param/.bin 从 assets 通过 AssetManager 直接喂给 native
        // (native 用 ncnn::Net::load_param/load_model 的 AAsset 重载)
        nativeHandle = nativeInit(am, paramName, binName);
        if (nativeHandle == 0) {
            throw new IOException("ncnn 模型加载失败: " + paramName);
        }
        idxToChar = loadCharset(am, charsetName);
        Log.i(TAG, "loaded " + idxToChar.length + " classes from " + charsetName);
    }

    /** 输入 64x64 float[],返回 [(char, prob), ...] top-K。 */
    public Result[] predict(float[] input64x64, int k) {
        if (input64x64.length != 64 * 64) {
            throw new IllegalArgumentException("input must be 64x64 = 4096 floats, got " + input64x64.length);
        }
        int[] indices = new int[k];
        float[] probs = new float[k];
        int n = nativePredict(nativeHandle, input64x64, indices, probs, k);
        Result[] out = new Result[n];
        for (int i = 0; i < n; i++) {
            int idx = indices[i];
            String ch = (idx >= 0 && idx < idxToChar.length) ? idxToChar[idx] : "?";
            out[i] = new Result(ch, probs[i]);
        }
        return out;
    }

    public void close() {
        if (nativeHandle != 0) {
            nativeRelease(nativeHandle);
            nativeHandle = 0;
        }
    }

    private static String[] loadCharset(AssetManager am, String name) throws IOException {
        try (InputStream is = am.open(name)) {
            byte[] buf = new byte[is.available()];
            int total = 0;
            int r;
            while ((r = is.read(buf, total, buf.length - total)) > 0) total += r;
            String json = new String(buf, 0, total, StandardCharsets.UTF_8);
            JSONObject root = new JSONObject(json);
            int n = root.getInt("num_classes");
            JSONObject c2i = root.getJSONObject("char_to_idx");
            String[] arr = new String[n];
            for (java.util.Iterator<String> it = c2i.keys(); it.hasNext(); ) {
                String ch = it.next();
                int idx = c2i.getInt(ch);
                if (idx >= 0 && idx < n) arr[idx] = ch;
            }
            return arr;
        } catch (Exception e) {
            throw new IOException("charset 解析失败: " + name, e);
        }
    }

    public static class Result {
        public final String ch;
        public final float prob;
        public Result(String ch, float prob) {
            this.ch = ch;
            this.prob = prob;
        }
    }

    private native long nativeInit(AssetManager am, String paramName, String binName);
    private native int  nativePredict(long handle, float[] input, int[] outIndices, float[] outProbs, int k);
    private native void nativeRelease(long handle);
}
