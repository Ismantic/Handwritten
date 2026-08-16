package com.shiyu.handwritten.runtime;

import android.content.res.AssetManager;

import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/** Reusable offline HCCR runtime shared by Android consumers. */
public final class HCCRRecognizer implements AutoCloseable {
    private static boolean nativeLibraryLoaded;

    public static synchronized void loadNativeLibrary(String name) {
        if (!nativeLibraryLoaded) {
            System.loadLibrary(name);
            nativeLibraryLoaded = true;
        }
    }

    private long nativeHandle;
    private final String[] idxToChar;

    public HCCRRecognizer(AssetManager assets, String paramName, String binName,
                          String charsetName) throws IOException {
        nativeHandle = nativeInit(assets, paramName, binName);
        if (nativeHandle == 0) throw new IOException("手写模型加载失败: " + paramName);
        idxToChar = loadCharset(assets, charsetName);
    }

    public Result[] predict(float[] input, int k) {
        if (nativeHandle == 0 || input.length != 64 * 64 || k <= 0) return new Result[0];
        int[] indices = new int[k];
        float[] probabilities = new float[k];
        int n = nativePredict(nativeHandle, input, indices, probabilities, k);
        Result[] result = new Result[Math.max(0, n)];
        for (int i = 0; i < result.length; i++) {
            int index = indices[i];
            String text = index >= 0 && index < idxToChar.length ? idxToChar[index] : "?";
            result[i] = new Result(text, probabilities[i]);
        }
        return result;
    }

    public boolean preprocess(byte[] gray, int width, int height, float[] output) {
        return nativeHandle != 0 && nativePreprocess(gray, width, height, output);
    }

    @Override public void close() {
        if (nativeHandle != 0) {
            nativeRelease(nativeHandle);
            nativeHandle = 0;
        }
    }

    private static String[] loadCharset(AssetManager assets, String name) throws IOException {
        try (InputStream stream = assets.open(name)) {
            JSONObject root = new JSONObject(new String(stream.readAllBytes(), StandardCharsets.UTF_8));
            String[] chars = new String[root.getInt("num_classes")];
            JSONObject map = root.getJSONObject("char_to_idx");
            java.util.Iterator<String> it = map.keys();
            while (it.hasNext()) {
                String text = it.next();
                int index = map.getInt(text);
                if (index >= 0 && index < chars.length) chars[index] = text;
            }
            return chars;
        } catch (Exception e) {
            throw new IOException("手写字表加载失败: " + name, e);
        }
    }

    public static final class Result {
        public final String text;
        public final float probability;
        Result(String text, float probability) {
            this.text = text;
            this.probability = probability;
        }
    }

    private static native long nativeInit(AssetManager assets, String param, String bin);
    private static native int nativePredict(long handle, float[] input, int[] indices,
                                            float[] probabilities, int requested);
    private static native void nativeRelease(long handle);
    private static native boolean nativePreprocess(byte[] gray, int width, int height,
                                                   float[] output);
}
