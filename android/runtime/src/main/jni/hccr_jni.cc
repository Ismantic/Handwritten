// Shared offline HCCR JNI runtime. Model weights remain valid for Session's life.
#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "net.h"
#include "datareader.h"
#include "preprocess.h"

namespace {
struct Session {
    ncnn::Net net;
    std::vector<unsigned char> model_bytes;
};

std::vector<unsigned char> read_asset(AAssetManager* manager, const char* name) {
    AAsset* asset = AAssetManager_open(manager, name, AASSET_MODE_BUFFER);
    if (!asset) return {};
    const auto length = static_cast<size_t>(AAsset_getLength64(asset));
    std::vector<unsigned char> bytes(length);
    const int got = AAsset_read(asset, bytes.data(), length);
    AAsset_close(asset);
    return got == static_cast<int>(length) ? bytes : std::vector<unsigned char>();
}
}  // namespace

extern "C" {
JNIEXPORT jlong JNICALL
Java_com_shiyu_handwritten_runtime_HCCRRecognizer_nativeInit(
        JNIEnv* env, jclass, jobject assets, jstring param, jstring bin) {
    AAssetManager* manager = AAssetManager_fromJava(env, assets);
    if (!manager) return 0;
    auto session = std::make_unique<Session>();
    session->net.opt.use_vulkan_compute = false;
    session->net.opt.num_threads = 4;
    const char* p = env->GetStringUTFChars(param, nullptr);
    std::vector<unsigned char> param_bytes = read_asset(manager, p);
    env->ReleaseStringUTFChars(param, p);
    if (param_bytes.empty()) return 0;
    const unsigned char* param_memory = param_bytes.data();
    ncnn::DataReaderFromMemory param_reader(param_memory);
    if (session->net.load_param(param_reader) != 0) return 0;
    const char* b = env->GetStringUTFChars(bin, nullptr);
    session->model_bytes = read_asset(manager, b);
    env->ReleaseStringUTFChars(bin, b);
    if (session->model_bytes.empty()) return 0;
    const unsigned char* bin_memory = session->model_bytes.data();
    ncnn::DataReaderFromMemory bin_reader(bin_memory);
    if (session->net.load_model(bin_reader) != 0) return 0;
    return reinterpret_cast<jlong>(session.release());
}

JNIEXPORT jint JNICALL
Java_com_shiyu_handwritten_runtime_HCCRRecognizer_nativePredict(
        JNIEnv* env, jclass, jlong handle, jfloatArray input, jintArray out_indices,
        jfloatArray out_probabilities, jint requested) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (!session || env->GetArrayLength(input) != 64 * 64 || requested <= 0) return 0;
    ncnn::Mat in(64, 64, 1);
    jfloat* source = env->GetFloatArrayElements(input, nullptr);
    std::memcpy(in.channel(0).data, source, sizeof(float) * 64 * 64);
    env->ReleaseFloatArrayElements(input, source, JNI_ABORT);
    ncnn::Extractor extractor = session->net.create_extractor();
    if (extractor.input("in0", in) != 0) return 0;
    ncnn::Mat out;
    if (extractor.extract("out0", out) != 0 || out.w <= 0) return 0;
    const int count = out.w;
    const float* logits = out;
    float maximum = logits[0];
    for (int i = 1; i < count; ++i) maximum = std::max(maximum, logits[i]);
    std::vector<float> probabilities(static_cast<size_t>(count));
    float total = 0;
    for (int i = 0; i < count; ++i) total += (probabilities[static_cast<size_t>(i)] = std::exp(logits[i] - maximum));
    if (total <= 0) return 0;
    for (float& value : probabilities) value /= total;
    const int k = std::min(requested, count);
    std::vector<int> indices(static_cast<size_t>(count));
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
            [&probabilities](int a, int b) { return probabilities[static_cast<size_t>(a)] > probabilities[static_cast<size_t>(b)]; });
    std::vector<jint> result_indices(static_cast<size_t>(k));
    std::vector<jfloat> result_probabilities(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        result_indices[static_cast<size_t>(i)] = indices[static_cast<size_t>(i)];
        result_probabilities[static_cast<size_t>(i)] = probabilities[static_cast<size_t>(indices[static_cast<size_t>(i)])];
    }
    env->SetIntArrayRegion(out_indices, 0, k, result_indices.data());
    env->SetFloatArrayRegion(out_probabilities, 0, k, result_probabilities.data());
    return k;
}

JNIEXPORT void JNICALL
Java_com_shiyu_handwritten_runtime_HCCRRecognizer_nativeRelease(JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<Session*>(handle);
}

JNIEXPORT jboolean JNICALL
Java_com_shiyu_handwritten_runtime_HCCRRecognizer_nativePreprocess(
        JNIEnv* env, jclass, jbyteArray gray, jint width, jint height, jfloatArray output) {
    if (width <= 0 || height <= 0 || env->GetArrayLength(gray) != width * height
            || env->GetArrayLength(output) != 64 * 64) return JNI_FALSE;
    jbyte* pixels = env->GetByteArrayElements(gray, nullptr);
    jfloat* result = env->GetFloatArrayElements(output, nullptr);
    int ok = hccr_preprocess(reinterpret_cast<const uint8_t*>(pixels), width, height, result);
    env->ReleaseByteArrayElements(gray, pixels, JNI_ABORT);
    env->ReleaseFloatArrayElements(output, result, 0);
    return ok ? JNI_TRUE : JNI_FALSE;
}
}  // extern "C"
