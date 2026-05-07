// JNI bridge for HCCR ncnn inference.
//
// Java 侧已经把 input 预处理成 [1, 64, 64] float(stroke=高,bg=0,值 [0,1]),
// 这里直接喂给 ncnn,做 forward + softmax + top-k。

#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "net.h"

#define TAG "HCCRJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

struct HCCRSession {
    ncnn::Net net;
};

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_shiyu_hccr_HCCRRecognizer_nativeInit(
    JNIEnv* env, jobject /*thiz*/,
    jobject jAssetManager, jstring jParam, jstring jBin) {

    AAssetManager* mgr = AAssetManager_fromJava(env, jAssetManager);
    if (!mgr) {
        LOGE("AAssetManager_fromJava 失败");
        return 0;
    }

    auto session = std::make_unique<HCCRSession>();
    // 不用 vulkan(MobileNetV2 64x64 在 CPU 上已经 ms 级)
    session->net.opt.use_vulkan_compute = false;
    session->net.opt.num_threads = 4;

    const char* paramName = env->GetStringUTFChars(jParam, nullptr);
    int rc1 = session->net.load_param(mgr, paramName);
    env->ReleaseStringUTFChars(jParam, paramName);
    if (rc1 != 0) {
        LOGE("load_param 失败: rc=%d", rc1);
        return 0;
    }

    const char* binName = env->GetStringUTFChars(jBin, nullptr);
    int rc2 = session->net.load_model(mgr, binName);
    env->ReleaseStringUTFChars(jBin, binName);
    if (rc2 != 0) {
        LOGE("load_model 失败: rc=%d", rc2);
        return 0;
    }

    LOGI("ncnn loaded OK");
    return reinterpret_cast<jlong>(session.release());
}

JNIEXPORT jint JNICALL
Java_com_shiyu_hccr_HCCRRecognizer_nativePredict(
    JNIEnv* env, jobject /*thiz*/,
    jlong handle, jfloatArray jInput, jintArray jOutIdx, jfloatArray jOutProbs, jint k) {

    auto* session = reinterpret_cast<HCCRSession*>(handle);
    if (!session) return 0;

    jsize inputLen = env->GetArrayLength(jInput);
    if (inputLen != 64 * 64) {
        LOGE("input length=%d expected 4096", inputLen);
        return 0;
    }

    // 拷到 ncnn::Mat([1, 64, 64]),fastest 方式是把 jfloatArray 直接做内存复制
    ncnn::Mat in(64, 64, 1);
    {
        jfloat* src = env->GetFloatArrayElements(jInput, nullptr);
        std::memcpy(in.channel(0).data, src, sizeof(float) * 64 * 64);
        env->ReleaseFloatArrayElements(jInput, src, JNI_ABORT);
    }

    ncnn::Extractor ex = session->net.create_extractor();
    if (ex.input("in0", in) != 0) {
        LOGE("ex.input failed");
        return 0;
    }
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) {
        LOGE("ex.extract failed");
        return 0;
    }

    int n = out.w;
    if (n <= 0) return 0;

    // softmax(numerically stable:减去 max)
    const float* logits = out;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    std::vector<float> probs(n);
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = std::exp(logits[i] - mx);
        sum += probs[i];
    }
    if (sum <= 0) return 0;
    float invSum = 1.0f / sum;
    for (int i = 0; i < n; i++) probs[i] *= invSum;

    // top-k(部分排序)
    int kk = std::min<int>(k, n);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&probs](int a, int b) { return probs[a] > probs[b]; });

    std::vector<jint>   outIdx(kk);
    std::vector<jfloat> outProbs(kk);
    for (int i = 0; i < kk; i++) {
        outIdx[i] = idx[i];
        outProbs[i] = probs[idx[i]];
    }
    env->SetIntArrayRegion(jOutIdx,   0, kk, outIdx.data());
    env->SetFloatArrayRegion(jOutProbs, 0, kk, outProbs.data());

    return kk;
}

JNIEXPORT void JNICALL
Java_com_shiyu_hccr_HCCRRecognizer_nativeRelease(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {

    auto* session = reinterpret_cast<HCCRSession*>(handle);
    if (session) {
        delete session;
    }
}

}  // extern "C"
