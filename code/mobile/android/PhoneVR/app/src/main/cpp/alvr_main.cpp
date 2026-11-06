#include "alvr_client_core.h"
#include "cardboard.h"
#include <GLES3/gl3.h>
#include <algorithm>
#include <android/log.h>
#include <cmath>
#include <deque>
#include <jni.h>
#include <map>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "nlohmann/json.hpp"
#include "utils.h"

using namespace nlohmann;

uint64_t HEAD_ID = alvr_path_string_to_id("/user/head");

// Note: the Cardboard SDK cannot estimate display time and an heuristic is used instead.
const uint64_t VSYNC_QUEUE_INTERVAL_NS = 50e6;
const float FLOOR_HEIGHT = 1.5;
const int MAXIMUM_TRACKING_FRAMES = 360;
// Time step used to estimate the head angular velocity from the Cardboard pose predictor.
// ALVR >= 20.9 does its own head pose prediction using the reported velocity.
const uint64_t VELOCITY_SAMPLE_INTERVAL_NS = 5e6;
// Rotate the decoded frame by the head movement that happened since it was rendered on the PC.
const bool ENABLE_REPROJECTION = true;

struct NativeContext {
    JavaVM *javaVm = nullptr;
    jobject javaContext = nullptr;

    CardboardHeadTracker *headTracker = nullptr;
    CardboardLensDistortion *lensDistortion = nullptr;
    CardboardDistortionRenderer *distortionRenderer = nullptr;

    int screenWidth = 0;
    int screenHeight = 0;

    bool renderingParamsChanged = true;
    bool glContextRecreated = false;

    bool running = false;
    bool streaming = false;
    bool hasStreamFrame = false;
    std::thread inputThread;

    // Decoder settings, received with the stream settings on ALVR_EVENT_STREAMING_STARTED
    bool forceSoftwareDecoder = false;
    float maxBufferingFrames = 2.0;
    float bufferingHistoryWeight = 0.9;
    std::vector<std::string> mediacodecStrings;
    std::vector<AlvrMediacodecOption> mediacodecOptions;

    // Une one texture per eye, no need for swapchains.
    GLuint lobbyTextures[2] = {0, 0};
    GLuint streamTextures[2] = {0, 0};

    float eyeOffsets[2] = {0.0, 0.0};
    AlvrFov fovArr[2] = {};
    AlvrViewParams viewParams[2] = {};
    AlvrDeviceMotion deviceMotion = {};

    NativeContext() {
        memset(&fovArr, 0, (sizeof(fovArr)) / sizeof(int));
        memset(&viewParams, 0, (sizeof(viewParams)) / sizeof(int));
        memset(&deviceMotion, 0, (sizeof(deviceMotion)) / sizeof(int));
    }
};

NativeContext CTX = {};

int64_t GetBootTimeNano() {
    struct timespec res = {};
    clock_gettime(CLOCK_BOOTTIME, &res);
    return (res.tv_sec * 1e9) + res.tv_nsec;
}

// Inverse unit quaternion
AlvrQuat inverseQuat(AlvrQuat q) { return {-q.x, -q.y, -q.z, q.w}; }

void cross(float a[3], float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

void quatVecMultiply(AlvrQuat q, float v[3], float out[3]) {
    float rv[3], rrv[3];
    float r[3] = {q.x, q.y, q.z};
    cross(r, v, rv);
    cross(r, rv, rrv);
    for (int i = 0; i < 3; i++) {
        out[i] = v[i] + 2 * (q.w * rv[i] + rrv[i]);
    }
}

void offsetPosWithQuat(AlvrQuat q, float offset[3], float outPos[3]) {
    float rotatedOffset[3];
    quatVecMultiply(q, offset, rotatedOffset);

    outPos[0] -= rotatedOffset[0];
    outPos[1] -= rotatedOffset[1] - FLOOR_HEIGHT;
    outPos[2] -= rotatedOffset[2];
}

AlvrFov getFov(CardboardEye eye) {
    float f[4];
    CardboardLensDistortion_getFieldOfView(CTX.lensDistortion, eye, f);

    AlvrFov fov = {};
    fov.left = -f[0];
    fov.right = f[1];
    fov.up = f[3];
    fov.down = -f[2];

    return fov;
}

AlvrPose getPose(uint64_t timestampNs) {
    AlvrPose pose = {};

    float pos[3];
    float q[4];
    CardboardHeadTracker_getPose(CTX.headTracker, (int64_t) timestampNs, kLandscapeLeft, pos, q);

    auto inverseOrientation = AlvrQuat{q[0], q[1], q[2], q[3]};
    pose.orientation = inverseQuat(inverseOrientation);

    return pose;
}

// Angular velocity (world frame, rad/s) that rotates `from` into `to` in `dtS` seconds
void angularVelocity(AlvrQuat from, AlvrQuat to, float dtS, float out[3]) {
    // delta = to * inverse(from)
    AlvrQuat f = inverseQuat(from);
    AlvrQuat d = {to.w * f.x + to.x * f.w + to.y * f.z - to.z * f.y,
                  to.w * f.y - to.x * f.z + to.y * f.w + to.z * f.x,
                  to.w * f.z + to.x * f.y - to.y * f.x + to.z * f.w,
                  to.w * f.w - to.x * f.x - to.y * f.y - to.z * f.z};
    if (d.w < 0) {
        d = {-d.x, -d.y, -d.z, -d.w};
    }
    float sinHalf = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    float angle = 2 * atan2f(sinHalf, d.w);
    float scale = sinHalf > 1e-6f ? angle / sinHalf / dtS : 2 / dtS;
    out[0] = d.x * scale;
    out[1] = d.y * scale;
    out[2] = d.z * scale;
}

// View poses in head space (head at origin), as expected by alvr_send_view_params()
void sendViewParams() {
    for (int eye = 0; eye < 2; eye++) {
        CTX.viewParams[eye] = {};
        CTX.viewParams[eye].pose.orientation = {0, 0, 0, 1};
        CTX.viewParams[eye].pose.position[0] = -CTX.eyeOffsets[eye];
        CTX.viewParams[eye].fov = CTX.fovArr[eye];
    }
    alvr_send_view_params(CTX.viewParams);
}

void inputThread() {
    auto deadline = std::chrono::steady_clock::now();

    info("inputThread: thread staring...");
    while (CTX.streaming) {
        auto pollTimestampNs = GetBootTimeNano();

        AlvrPose headPose = getPose(pollTimestampNs);
        AlvrPose nextHeadPose = getPose(pollTimestampNs + VELOCITY_SAMPLE_INTERVAL_NS);

        CTX.deviceMotion = {};
        CTX.deviceMotion.device_id = HEAD_ID;
        CTX.deviceMotion.pose = headPose;
        CTX.deviceMotion.pose.position[1] = FLOOR_HEIGHT;
        angularVelocity(headPose.orientation,
                        nextHeadPose.orientation,
                        VELOCITY_SAMPLE_INTERVAL_NS / 1e9f,
                        CTX.deviceMotion.angular_velocity);

        alvr_send_tracking(pollTimestampNs, &CTX.deviceMotion, 1, nullptr, nullptr);

        deadline += std::chrono::nanoseconds((uint64_t) (1e9 / 60.f / 3));
        std::this_thread::sleep_until(deadline);
    }
}

// Reads the decoder related settings sent by the streamer
void parseDecoderSettings(const json &settings) {
    CTX.forceSoftwareDecoder = false;
    CTX.maxBufferingFrames = 2.0;
    CTX.bufferingHistoryWeight = 0.9;
    CTX.mediacodecStrings.clear();
    CTX.mediacodecOptions.clear();

    if (!settings.contains("video") || !settings["video"].is_object()) {
        error("settings_json doesn't have a video key");
        return;
    }
    const json &video = settings["video"];

    if (video.contains("force_software_decoder") && video["force_software_decoder"].is_boolean())
        CTX.forceSoftwareDecoder = video["force_software_decoder"];
    if (video.contains("max_buffering_frames") && video["max_buffering_frames"].is_number())
        CTX.maxBufferingFrames = video["max_buffering_frames"];
    if (video.contains("buffering_history_weight") && video["buffering_history_weight"].is_number())
        CTX.bufferingHistoryWeight = video["buffering_history_weight"];

    // mediacodec_extra_options: [[key, {"ty": "Int32", "value": "1"}], ...]
    if (!video.contains("mediacodec_extra_options") ||
        !video["mediacodec_extra_options"].is_array())
        return;

    // Keep the strings alive and their addresses stable before taking c_str() pointers
    CTX.mediacodecStrings.reserve(video["mediacodec_extra_options"].size() * 2);
    for (const auto &entry : video["mediacodec_extra_options"]) {
        if (!entry.is_array() || entry.size() != 2 || !entry[0].is_string() ||
            !entry[1].contains("ty") || !entry[1].contains("value"))
            continue;

        std::string ty = entry[1]["ty"];
        std::string value = entry[1]["value"];
        AlvrMediacodecOption option = {};

        CTX.mediacodecStrings.push_back(entry[0]);
        option.key = CTX.mediacodecStrings.back().c_str();
        if (ty == "Float") {
            option.ty = ALVR_MEDIACODEC_PROP_TYPE_FLOAT;
            option.value.float_ = std::stof(value);
        } else if (ty == "Int32") {
            option.ty = ALVR_MEDIACODEC_PROP_TYPE_INT32;
            option.value.int32 = std::stoi(value);
        } else if (ty == "Int64") {
            option.ty = ALVR_MEDIACODEC_PROP_TYPE_INT64;
            option.value.int64 = std::stoll(value);
        } else if (ty == "String") {
            CTX.mediacodecStrings.push_back(value);
            option.ty = ALVR_MEDIACODEC_PROP_TYPE_STRING;
            option.value.string = CTX.mediacodecStrings.back().c_str();
        } else {
            continue;
        }
        CTX.mediacodecOptions.push_back(option);
    }
}

void createDecoder(AlvrCodec codec) {
    auto configLen = alvr_get_decoder_config(nullptr);
    auto configBuffer = std::vector<char>(configLen);
    if (configLen > 0)
        alvr_get_decoder_config(&configBuffer[0]);

    AlvrDecoderConfig config = {};
    config.codec = codec;
    config.force_software_decoder = CTX.forceSoftwareDecoder;
    config.max_buffering_frames = CTX.maxBufferingFrames;
    config.buffering_history_weight = CTX.bufferingHistoryWeight;
    config.options = CTX.mediacodecOptions.empty() ? nullptr : &CTX.mediacodecOptions[0];
    config.options_count = CTX.mediacodecOptions.size();
    config.config_buffer = (const uint8_t *) configBuffer.data();
    config.config_buffer_size = configLen;

    info("Creating decoder: codec %d, software %d, %d extra options",
         (int) codec,
         CTX.forceSoftwareDecoder,
         (int) CTX.mediacodecOptions.size());
    alvr_create_decoder(config);
}

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
    CTX.javaVm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_initializeNative(
    JNIEnv *env, jobject obj, jint screenWidth, jint screenHeight, jfloat refreshRate) {
    CTX.javaContext = env->NewGlobalRef(obj);

    uint32_t viewWidth = std::max(screenWidth, screenHeight) / 2;
    uint32_t viewHeight = std::min(screenWidth, screenHeight);

    static bool loggingInitialized = false;
    if (!loggingInitialized) {
        alvr_initialize_logging();
        loggingInitialized = true;
    }

    alvr_initialize_android_context((void *) CTX.javaVm, (void *) CTX.javaContext);

    float refreshRatesBuffer[1] = {refreshRate};

    AlvrClientCapabilities caps = {};
    caps.default_view_height = viewHeight;
    caps.default_view_width = viewWidth;
    caps.refresh_rates = refreshRatesBuffer;
    caps.refresh_rates_count = 1;
    caps.foveated_encoding =
        true;   // By default disable FFE (can be force-enabled by Server Settings
    caps.encoder_high_profile = true;
    // Most phone decoders can't reliably handle 10 bit or AV1 streams
    caps.encoder_10_bits = false;
    caps.encoder_av1 = false;
    caps.prefer_10bit = false;
    caps.prefer_full_range = true;
    caps.preferred_encoding_gamma = 1.0;
    caps.prefer_hdr = false;

    alvr_initialize(caps);

    Cardboard_initializeAndroid(CTX.javaVm, CTX.javaContext);
    CTX.headTracker = CardboardHeadTracker_create();
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_destroyNative(JNIEnv *,
                                                                                        jobject) {
    alvr_destroy_opengl();
    alvr_destroy();

    CardboardHeadTracker_destroy(CTX.headTracker);
    CTX.headTracker = nullptr;
    CardboardLensDistortion_destroy(CTX.lensDistortion);
    CTX.lensDistortion = nullptr;
    CardboardDistortionRenderer_destroy(CTX.distortionRenderer);
    CTX.distortionRenderer = nullptr;
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_resumeNative(JNIEnv *,
                                                                                       jobject) {
    CardboardHeadTracker_resume(CTX.headTracker);

    CTX.renderingParamsChanged = true;

    uint8_t *buffer;
    int size;
    CardboardQrCode_getSavedDeviceParams(&buffer, &size);
    if (size == 0) {
        CardboardQrCode_scanQrCodeAndSaveDeviceParams();
    }
    CardboardQrCode_destroy(buffer);

    CTX.running = true;

    alvr_resume();
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_pauseNative(JNIEnv *,
                                                                                      jobject) {
    alvr_pause();

    if (CTX.running) {
        CTX.running = false;
    }

    CardboardHeadTracker_pause(CTX.headTracker);
}

extern "C" JNIEXPORT void JNICALL
Java_viritualisres_phonevr_ALVRActivity_surfaceCreatedNative(JNIEnv *, jobject) {
    alvr_initialize_opengl();

    CTX.glContextRecreated = true;
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_setScreenResolutionNative(
    JNIEnv *, jobject, jint width, jint height) {
    CTX.screenWidth = width;
    CTX.screenHeight = height;

    CTX.renderingParamsChanged = true;
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_sendBatteryLevel(
    JNIEnv *, jobject, jfloat level, jboolean plugged) {
    alvr_send_battery(HEAD_ID, level, plugged);
}

extern "C" JNIEXPORT void JNICALL Java_viritualisres_phonevr_ALVRActivity_renderNative(JNIEnv *,
                                                                                       jobject) {
    try {
        if (CTX.renderingParamsChanged) {
            info("renderingParamsChanged, processing new params");
            uint8_t *buffer;
            int size;
            CardboardQrCode_getSavedDeviceParams(&buffer, &size);

            if (size == 0) {
                return;
            }

            info("renderingParamsChanged, sending new params to alvr");
            if (CTX.lensDistortion) {
                CardboardLensDistortion_destroy(CTX.lensDistortion);
                CTX.lensDistortion = nullptr;
            }
            info("renderingParamsChanged, destroyed distortion");
            CTX.lensDistortion =
                CardboardLensDistortion_create(buffer, size, CTX.screenWidth, CTX.screenHeight);

            CardboardQrCode_destroy(buffer);
            *buffer = 0;

            if (CTX.distortionRenderer) {
                CardboardDistortionRenderer_destroy(CTX.distortionRenderer);
                CTX.distortionRenderer = nullptr;
            }
            const CardboardOpenGlEsDistortionRendererConfig config{kGlTexture2D};
            CTX.distortionRenderer = CardboardOpenGlEs2DistortionRenderer_create(&config);

            for (int eye = 0; eye < 2; eye++) {
                CardboardMesh mesh;
                CardboardLensDistortion_getDistortionMesh(
                    CTX.lensDistortion, (CardboardEye) eye, &mesh);
                CardboardDistortionRenderer_setMesh(
                    CTX.distortionRenderer, &mesh, (CardboardEye) eye);

                float matrix[16] = {};
                CardboardLensDistortion_getEyeFromHeadMatrix(
                    CTX.lensDistortion, (CardboardEye) eye, matrix);
                CTX.eyeOffsets[eye] = matrix[12];
            }

            CTX.fovArr[kLeft] = getFov(kLeft);
            CTX.fovArr[kRight] = getFov(kRight);

            info("renderingParamsChanged, updating new view configs (FOV) to alvr");
            sendViewParams();
        }

        // Note: if GL context is recreated, old resources are already freed.
        if (CTX.renderingParamsChanged && !CTX.glContextRecreated) {
            info("Pausing ALVR since glContext is not recreated, deleting textures");
            alvr_pause_opengl();

            GL(glDeleteTextures(2, CTX.lobbyTextures));
        }

        if (CTX.renderingParamsChanged || CTX.glContextRecreated) {
            info("Rebuilding, binding textures, Resuming ALVR since glContextRecreated %b, "
                 "renderingParamsChanged %b",
                 CTX.renderingParamsChanged,
                 CTX.glContextRecreated);
            GL(glGenTextures(2, CTX.lobbyTextures));

            for (auto &lobbyTexture : CTX.lobbyTextures) {
                GL(glBindTexture(GL_TEXTURE_2D, lobbyTexture));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
                GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
                GL(glTexImage2D(GL_TEXTURE_2D,
                                0,
                                GL_RGB,
                                CTX.screenWidth / 2,
                                CTX.screenHeight,
                                0,
                                GL_RGB,
                                GL_UNSIGNED_BYTE,
                                nullptr));
            }

            const uint32_t *targetViews[2] = {(uint32_t *) &CTX.lobbyTextures[0],
                                              (uint32_t *) &CTX.lobbyTextures[1]};
            alvr_resume_opengl(CTX.screenWidth / 2, CTX.screenHeight, targetViews, 1);

            CTX.renderingParamsChanged = false;
            CTX.glContextRecreated = false;
        }

        AlvrEvent event;
        while (alvr_poll_event(&event)) {
            if (event.tag == ALVR_EVENT_HUD_MESSAGE_UPDATED) {
                auto message_length = alvr_hud_message(nullptr);
                auto message_buffer = std::vector<char>(message_length);

                alvr_hud_message(&message_buffer[0]);
                info("ALVR Poll Event: HUD Message Update - %s", &message_buffer[0]);

                if (message_length > 0)
                    alvr_update_hud_message_opengl(&message_buffer[0]);
            }
            if (event.tag == ALVR_EVENT_STREAMING_STARTED) {
                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STARTED, generating and binding "
                     "textures...");
                auto config = event.STREAMING_STARTED;

                auto settings_len = alvr_get_settings_json(nullptr);
                auto settings_buffer = std::vector<char>(settings_len);
                alvr_get_settings_json(&settings_buffer[0]);

                info("Got settings from ALVR Server - %s", &settings_buffer[0]);
                if (settings_len > 900)   // to workthough logcat buffer limit
                    info("Got settings from ALVR Server - %s", &settings_buffer[900]);
                json settings_json = json::parse(&settings_buffer[0]);

                GL(glGenTextures(2, CTX.streamTextures));

                for (auto &streamTexture : CTX.streamTextures) {
                    GL(glBindTexture(GL_TEXTURE_2D, streamTexture));
                    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
                    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
                    GL(glTexImage2D(GL_TEXTURE_2D,
                                    0,
                                    GL_RGB,
                                    config.view_width,
                                    config.view_height,
                                    0,
                                    GL_RGB,
                                    GL_UNSIGNED_BYTE,
                                    nullptr));
                }

                CTX.fovArr[0] = getFov((CardboardEye) 0);
                CTX.fovArr[1] = getFov((CardboardEye) 1);

                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STARTED, View configs updated...");

                auto leftIntHandle = (uint32_t) CTX.streamTextures[0];
                auto rightIntHandle = (uint32_t) CTX.streamTextures[1];
                const uint32_t *textureHandles[2] = {&leftIntHandle, &rightIntHandle};

                auto render_config = AlvrStreamConfig{};
                render_config.view_resolution_width = config.view_width;
                render_config.view_resolution_height = config.view_height;
                render_config.swapchain_textures = textureHandles;
                render_config.swapchain_length = 1;

                parseDecoderSettings(settings_json);

                // The streamer tells whether foveated encoding is actually used for this stream,
                // the shape parameters come from the settings ("Enabled": {...} or "Disabled")
                render_config.enable_foveation = false;
                const json &video =
                    settings_json.contains("video") ? settings_json["video"] : json();
                if (config.enable_foveated_encoding && video.is_object() &&
                    video.contains("foveated_encoding") && video["foveated_encoding"].is_object() &&
                    video["foveated_encoding"].contains("Enabled")) {
                    const json &ffe = video["foveated_encoding"]["Enabled"];
                    info("settings_json.video.foveated_encoding is %s", ffe.dump().c_str());

                    render_config.enable_foveation = true;
                    render_config.foveation_center_size_x = ffe.value("center_size_x", 0.45f);
                    render_config.foveation_center_size_y = ffe.value("center_size_y", 0.4f);
                    render_config.foveation_center_shift_x = ffe.value("center_shift_x", 0.4f);
                    render_config.foveation_center_shift_y = ffe.value("center_shift_y", 0.1f);
                    render_config.foveation_edge_ratio_x = ffe.value("edge_ratio_x", 4.0f);
                    render_config.foveation_edge_ratio_y = ffe.value("edge_ratio_y", 5.0f);
                } else
                    info("foveated_encoding is Disabled");
                render_config.enable_upscaling = false;

                info("Settings for foveation:");
                info("render_config.enable_foveation: %b", render_config.enable_foveation);
                info("render_config.foveation_center_size_x: %f",
                     render_config.foveation_center_size_x);
                info("render_config.foveation_center_size_y: %f",
                     render_config.foveation_center_size_y);
                info("render_config.foveation_center_shift_x: %f",
                     render_config.foveation_center_shift_x);
                info("render_config.foveation_center_shift_y: %f",
                     render_config.foveation_center_shift_y);
                info("render_config.foveation_edge_ratio_x: %f",
                     render_config.foveation_edge_ratio_x);
                info("render_config.foveation_edge_ratio_y: %f",
                     render_config.foveation_edge_ratio_y);

                alvr_start_stream_opengl(render_config);

                sendViewParams();

                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STARTED, opengl stream started and "
                     "input "
                     "Thread started...");
                CTX.hasStreamFrame = false;
                CTX.streaming = true;
                CTX.inputThread = std::thread(inputThread);

            } else if (event.tag == ALVR_EVENT_DECODER_CONFIG) {
                info("ALVR Poll Event: ALVR_EVENT_DECODER_CONFIG, creating decoder...");
                createDecoder(event.DECODER_CONFIG.codec);
            } else if (event.tag == ALVR_EVENT_STREAMING_STOPPED) {
                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STOPPED, Waiting for inputThread to "
                     "join...");
                CTX.streaming = false;
                CTX.inputThread.join();

                alvr_destroy_decoder();
                CTX.hasStreamFrame = false;

                GL(glDeleteTextures(2, CTX.streamTextures));
                info("ALVR Poll Event: ALVR_EVENT_STREAMING_STOPPED, Stream stopped deleted "
                     "textures.");
            }
        }

        CardboardEyeTextureDescription viewsDescs[2] = {};
        for (auto &viewsDesc : viewsDescs) {
            viewsDesc.left_u = 0.0;
            viewsDesc.right_u = 1.0;
            viewsDesc.top_v = 1.0;
            viewsDesc.bottom_v = 0.0;
        }

        if (CTX.streaming) {
            void *streamHardwareBuffer = nullptr;
            uint64_t timestampNs = 0;

            if (alvr_get_frame(&timestampNs, &streamHardwareBuffer)) {
                // Poses the streamer rendered this frame with
                AlvrViewParams frameViews[2] = {};
                alvr_report_compositor_start(timestampNs, frameViews);

                AlvrQuat reprojection = {0, 0, 0, 1};
                if (ENABLE_REPROJECTION) {
                    AlvrPose displayPose = getPose(GetBootTimeNano() + VSYNC_QUEUE_INTERVAL_NS);
                    reprojection = alvr_rotation_delta(frameViews[0].pose.orientation,
                                                       displayPose.orientation);
                }

                AlvrStreamViewParams streamViews[2] = {};
                for (int eye = 0; eye < 2; eye++) {
                    streamViews[eye].swapchain_index = 0;
                    streamViews[eye].reprojection_rotation = reprojection;
                    streamViews[eye].fov = frameViews[eye].fov;
                }
                alvr_render_stream_opengl(streamHardwareBuffer, streamViews);

                alvr_report_submit(timestampNs, 0);
                CTX.hasStreamFrame = true;
            }

            // Without a new frame, show the last decoded one again
            if (!CTX.hasStreamFrame) {
                return;
            }

            viewsDescs[0].texture = CTX.streamTextures[0];
            viewsDescs[1].texture = CTX.streamTextures[1];
        } else {
            AlvrPose pose = getPose(GetBootTimeNano() + VSYNC_QUEUE_INTERVAL_NS);

            AlvrLobbyViewParams viewInputs[2] = {};
            for (int eye = 0; eye < 2; eye++) {
                float headToEye[3] = {CTX.eyeOffsets[eye], 0.0, 0.0};
                // offset head pos to Eye Position
                offsetPosWithQuat(pose.orientation, headToEye, viewInputs[eye].pose.position);

                viewInputs[eye].pose.orientation = pose.orientation;
                viewInputs[eye].fov = getFov((CardboardEye) eye);
                viewInputs[eye].swapchain_index = 0;
            }
            alvr_render_lobby_opengl(viewInputs, true);

            viewsDescs[0].texture = CTX.lobbyTextures[0];
            viewsDescs[1].texture = CTX.lobbyTextures[1];
        }

        // Note: the Cardboard SDK does not support reprojection!
        // todo: manually implement it?

        // info("nativeRendered: Rendering to Display...");
        CardboardDistortionRenderer_renderEyeToDisplay(CTX.distortionRenderer,
                                                       0,
                                                       0,
                                                       0,
                                                       CTX.screenWidth,
                                                       CTX.screenHeight,
                                                       &viewsDescs[0],
                                                       &viewsDescs[1]);
    } catch (const json::exception &e) {
        error(std::string(std::string(__FUNCTION__) + std::string(__FILE_NAME__) +
                          std::string(e.what()))
                  .c_str());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_viritualisres_phonevr_ALVRActivity_switchViewerNative(JNIEnv *, jobject) {
    CardboardQrCode_scanQrCodeAndSaveDeviceParams();
}
