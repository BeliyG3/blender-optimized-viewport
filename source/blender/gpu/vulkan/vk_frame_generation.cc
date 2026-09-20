/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "vk_frame_generation.hh"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>

#include <vulkan/vulkan.h>

#include <nvsdk_ngx_defs_dlssg.h>
#include <nvsdk_ngx_params_dlssg.h>
#include <nvsdk_ngx_vk.h>

#include "GPU_texture.hh"

#include "vk_backend.hh"
#include "vk_context.hh"
#include "vk_image_view.hh"
#include "vk_ngx_runtime.hh"
#include "vk_texture.hh"

namespace blender::gpu {

namespace {

std::atomic<int> active_sessions = 0;

class VKFrameGenerationSession final : public FrameGenerationSession {
 public:
  VKFrameGenerationSession()
  {
    active_sessions.fetch_add(1, std::memory_order_relaxed);
  }

  ~VKFrameGenerationSession() override
  {
    release();
    active_sessions.fetch_sub(1, std::memory_order_relaxed);
  }

  bool evaluate(const FrameGenerationEvaluation &evaluation) override
  {
    if (evaluation.color == nullptr || evaluation.depth == nullptr ||
        evaluation.motion == nullptr || evaluation.color_width <= 0 ||
        evaluation.color_height <= 0)
    {
      error_ = "DLSS Frame Generation received incomplete viewport resources";
      return false;
    }
    if (!ngx_runtime().frame_generation_available()) {
      error_ = ngx_runtime().error_get();
      return false;
    }
    if (!ensure_outputs(evaluation.color_width, evaluation.color_height)) {
      return false;
    }

    pending_ = evaluation;
    pending_result_ = false;

    VKTexture &color = *unwrap(evaluation.color);
    VKTexture &depth = *unwrap(evaluation.depth);
    VKTexture &motion = *unwrap(evaluation.motion);
    VKTexture &generated = *unwrap(generated_texture_);
    VKTexture &real = *unwrap(real_texture_);
    /* NGX may leave OutputReal untouched and may decline to write an interpolated frame without
     * failing the evaluation. Initialize both outputs with the exact display-referred backbuffer
     * so either case degrades to the matching real frame instead of undefined texture contents. */
    GPU_texture_copy(generated_texture_, evaluation.color);
    GPU_texture_copy(real_texture_, evaluation.color);

    render_graph::VKFrameGenerationNode::CreateInfo create_info = {};
    create_info.node_data.command = execute_callback;
    create_info.node_data.user_data = this;
    create_info.color_image = color.vk_image_handle();
    create_info.depth_image = depth.vk_image_handle();
    create_info.motion_image = motion.vk_image_handle();
    create_info.generated_image = generated.vk_image_handle();
    create_info.real_image = real.vk_image_handle();

    VKContext &context = *VKContext::get();
    context.rendering_end();
    context.render_graph().add_node(create_info);
    /* Completion, not submission: what comes next reads this image back - Cycles pulls the
     * denoised result out of it on the CUDA side the moment this returns - and a submitted command
     * buffer is one the GPU has not necessarily run. Everywhere else the engine reads a result
     * back it waits for completion (`vk_context.cc`); waiting only for submission here let a frame
     * be shown with its lower half from this evaluation and its upper half from the last. */
    context.flush_render_graph(RenderGraphFlushFlags::SUBMIT |
                               RenderGraphFlushFlags::WAIT_FOR_COMPLETION |
                               RenderGraphFlushFlags::RENEW_RENDER_GRAPH);
    return pending_result_;
  }

  void reset() override
  {
    reset_requested_ = true;
  }

  Texture *generated_texture_get() const override
  {
    return generated_texture_;
  }

  Texture *real_texture_get() const override
  {
    return real_texture_;
  }

  const std::string &error_get() const override
  {
    return error_;
  }

 private:
  static void execute_callback(void *user_data, const VkCommandBuffer command_buffer)
  {
    static_cast<VKFrameGenerationSession *>(user_data)->execute(command_buffer);
  }

  bool ensure_outputs(const int width, const int height)
  {
    if (width_ == width && height_ == height && generated_texture_ != nullptr &&
        real_texture_ != nullptr)
    {
      return true;
    }

    release_feature();
    if (generated_texture_ != nullptr) {
      GPU_texture_free(generated_texture_);
    }
    if (real_texture_ != nullptr) {
      GPU_texture_free(real_texture_);
    }
    const eGPUTextureUsage usage = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE;
    generated_texture_ = GPU_texture_create_2d("DLSSG generated frame",
                                               width,
                                               height,
                                               1,
                                               TextureFormat::SFLOAT_16_16_16_16,
                                               usage,
                                               nullptr);
    real_texture_ = GPU_texture_create_2d("DLSSG retained real frame",
                                          width,
                                          height,
                                          1,
                                          TextureFormat::SFLOAT_16_16_16_16,
                                          usage,
                                          nullptr);
    if (generated_texture_ == nullptr || real_texture_ == nullptr) {
      error_ = "Failed to allocate Vulkan DLSS Frame Generation output textures";
      return false;
    }

    width_ = width;
    height_ = height;
    reset_requested_ = true;
    return true;
  }

  void execute(const VkCommandBuffer command_buffer)
  {
    NGXVulkanRuntime &runtime = ngx_runtime();
    if (parameters_ == nullptr) {
      const NVSDK_NGX_Result result = runtime.allocate_parameters(&parameters_);
      if (NVSDK_NGX_FAILED(result) || parameters_ == nullptr) {
        error_ = ngx_error("DLSSG parameter allocation", result);
        return;
      }
    }

    if (feature_ == nullptr) {
      parameters_->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
      parameters_->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
      parameters_->Set(NVSDK_NGX_Parameter_Width, unsigned(width_));
      parameters_->Set(NVSDK_NGX_Parameter_Height, unsigned(height_));
      parameters_->Set(NVSDK_NGX_DLSSG_Parameter_Width, unsigned(width_));
      parameters_->Set(NVSDK_NGX_DLSSG_Parameter_Height, unsigned(height_));
      parameters_->Set(NVSDK_NGX_DLSSG_Parameter_BackbufferFormat,
                       unsigned(VK_FORMAT_R16G16B16A16_SFLOAT));
      parameters_->Set(
          NVSDK_NGX_DLSSG_Parameter_ResourceAlwaysProvided_Flags,
          unsigned(NVSDK_NGX_DLSSG_ResourceFlags_Backbuffer | NVSDK_NGX_DLSSG_ResourceFlags_MVecs |
                   NVSDK_NGX_DLSSG_ResourceFlags_Depth | NVSDK_NGX_DLSSG_ResourceFlags_HUDLess |
                   NVSDK_NGX_DLSSG_ResourceFlags_OutputInterpolated |
                   NVSDK_NGX_DLSSG_ResourceFlags_OutputReal));
      parameters_->Set(NVSDK_NGX_DLSSG_Parameter_UserInterfaceRecompositionEnabled, 0u);
      const NVSDK_NGX_Result result = runtime.create_feature(
          command_buffer, NVSDK_NGX_Feature_FrameGeneration, parameters_, &feature_);
      if (NVSDK_NGX_FAILED(result) || feature_ == nullptr) {
        error_ = ngx_error("DLSSG feature creation", result);
        return;
      }
    }

    VKTexture &color = *unwrap(pending_.color);
    VKTexture &depth = *unwrap(pending_.depth);
    VKTexture &motion = *unwrap(pending_.motion);
    VKTexture &generated = *unwrap(generated_texture_);
    VKTexture &real = *unwrap(real_texture_);

    NVSDK_NGX_Resource_VK color_resource = ngx_resource(
        color, pending_.color_width, pending_.color_height, false);
    NVSDK_NGX_Resource_VK depth_resource = ngx_resource(
        depth, pending_.render_width, pending_.render_height, false);
    NVSDK_NGX_Resource_VK motion_resource = ngx_resource(
        motion, pending_.render_width, pending_.render_height, false);
    NVSDK_NGX_Resource_VK generated_resource = ngx_resource(
        generated, pending_.color_width, pending_.color_height, true);
    NVSDK_NGX_Resource_VK real_resource = ngx_resource(
        real, pending_.color_width, pending_.color_height, true);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_Backbuffer, &color_resource);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_HUDLess, &color_resource);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_Depth, &depth_resource);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MVecs, &motion_resource);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputInterpolated, &generated_resource);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputReal, &real_resource);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MultiFrameCount, 1u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MultiFrameIndex, 1u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID, pending_.frame_id);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_InternalWidth, unsigned(pending_.render_width));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_InternalHeight, unsigned(pending_.render_height));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_DynamicResolution, 0u);

    const unsigned color_subrect_x = unsigned(std::max(pending_.color_subrect_x, 0));
    const unsigned color_subrect_y = unsigned(std::max(pending_.color_subrect_y, 0));
    const unsigned color_subrect_width = unsigned(
        pending_.color_subrect_width > 0 ? pending_.color_subrect_width : pending_.color_width);
    const unsigned color_subrect_height = unsigned(
        pending_.color_subrect_height > 0 ? pending_.color_subrect_height : pending_.color_height);

    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseX, color_subrect_x);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseY, color_subrect_y);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectWidth, color_subrect_width);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectHeight, color_subrect_height);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MVecsSubrectBaseX, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MVecsSubrectBaseY, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MVecsSubrectWidth, unsigned(pending_.render_width));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MVecsSubrectHeight,
                     unsigned(pending_.render_height));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_DepthSubrectBaseX, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_DepthSubrectBaseY, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_DepthSubrectWidth, unsigned(pending_.render_width));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_DepthSubrectHeight,
                     unsigned(pending_.render_height));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectBaseX, color_subrect_x);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectBaseY, color_subrect_y);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectWidth, color_subrect_width);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectHeight, color_subrect_height);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectBaseX, color_subrect_x);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectBaseY, color_subrect_y);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectWidth,
                     color_subrect_width);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectHeight,
                     color_subrect_height);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectBaseX, color_subrect_x);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectBaseY, color_subrect_y);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectWidth, color_subrect_width);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectHeight, color_subrect_height);

    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_CameraViewToClip,
                     pending_.camera.camera_view_to_clip);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_ClipToCameraView,
                     pending_.camera.clip_to_camera_view);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_ClipToLensClip, pending_.camera.clip_to_lens_clip);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_ClipToPrevClip, pending_.camera.clip_to_prev_clip);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_PrevClipToClip, pending_.camera.prev_clip_to_clip);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_JitterOffsetX, 0.0f);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_JitterOffsetY, 0.0f);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MvecScaleX,
                     1.0f / std::max(float(pending_.render_width), 1.0f));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MvecScaleY,
                     1.0f / std::max(float(pending_.render_height), 1.0f));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetX, 0.0f);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetY, 0.0f);

    for (int axis = 0; axis < 3; axis++) {
      const char *position_names[] = {NVSDK_NGX_DLSSG_Parameter_CameraPosX,
                                      NVSDK_NGX_DLSSG_Parameter_CameraPosY,
                                      NVSDK_NGX_DLSSG_Parameter_CameraPosZ};
      const char *up_names[] = {NVSDK_NGX_DLSSG_Parameter_CameraUpX,
                                NVSDK_NGX_DLSSG_Parameter_CameraUpY,
                                NVSDK_NGX_DLSSG_Parameter_CameraUpZ};
      const char *right_names[] = {NVSDK_NGX_DLSSG_Parameter_CameraRightX,
                                   NVSDK_NGX_DLSSG_Parameter_CameraRightY,
                                   NVSDK_NGX_DLSSG_Parameter_CameraRightZ};
      const char *forward_names[] = {NVSDK_NGX_DLSSG_Parameter_CameraFwdX,
                                     NVSDK_NGX_DLSSG_Parameter_CameraFwdY,
                                     NVSDK_NGX_DLSSG_Parameter_CameraFwdZ};
      parameters_->Set(position_names[axis], pending_.camera.position[axis]);
      parameters_->Set(up_names[axis], pending_.camera.up[axis]);
      parameters_->Set(right_names[axis], pending_.camera.right[axis]);
      parameters_->Set(forward_names[axis], pending_.camera.forward[axis]);
    }

    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_CameraNear, pending_.camera.near_clip);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_CameraFar, pending_.camera.far_clip);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_CameraFOV, pending_.camera.field_of_view);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_CameraAspectRatio, pending_.camera.aspect_ratio);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_ColorBuffersHDR,
                     unsigned(pending_.color_buffers_hdr));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_DepthInverted, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_CameraMotionIncluded, 1u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_Reset,
                     unsigned(pending_.reset || reset_requested_));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_AutomodeOverrideReset, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_NotRenderingGameFrames, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_OrthoProjection,
                     unsigned(pending_.camera.orthographic));
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MvecInvalidValue, 0.0f);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MvecDilated, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MvecJittered, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_MenuDetectionEnabled, 0u);
    parameters_->Set(NVSDK_NGX_DLSSG_Parameter_EvalFlags, 0u);

    const NVSDK_NGX_Result result = runtime.evaluate_feature(
        command_buffer, feature_, parameters_, nullptr);
    if (NVSDK_NGX_FAILED(result)) {
      error_ = ngx_error("DLSSG evaluation", result);
      return;
    }

    reset_requested_ = false;
    pending_result_ = true;
    error_.clear();
  }

  void release_feature()
  {
    NGXVulkanRuntime &runtime = ngx_runtime();
    if (feature_ != nullptr && runtime.release_feature != nullptr) {
      if (VKContext::get() != nullptr) {
        VKContext::get()->finish();
      }
      runtime.release_feature(feature_);
      feature_ = nullptr;
    }
    if (parameters_ != nullptr && runtime.destroy_parameters != nullptr) {
      runtime.destroy_parameters(parameters_);
      parameters_ = nullptr;
    }
  }

  void release()
  {
    release_feature();
    if (generated_texture_ != nullptr) {
      GPU_texture_free(generated_texture_);
      generated_texture_ = nullptr;
    }
    if (real_texture_ != nullptr) {
      GPU_texture_free(real_texture_);
      real_texture_ = nullptr;
    }
  }

  FrameGenerationEvaluation pending_;
  bool pending_result_ = false;
  bool reset_requested_ = true;
  int width_ = 0;
  int height_ = 0;
  Texture *generated_texture_ = nullptr;
  Texture *real_texture_ = nullptr;
  NVSDK_NGX_Parameter *parameters_ = nullptr;
  NVSDK_NGX_Handle *feature_ = nullptr;
  std::string error_;
};

}  // namespace

FrameGenerationCapabilities vk_frame_generation_capabilities_get()
{
  FrameGenerationCapabilities capabilities;
  capabilities.compiled = true;
  capabilities.backend = "Vulkan";

  const VkPhysicalDeviceProperties &properties =
      VKBackend::get().device.physical_device_properties_get();
  const std::string device_name = properties.deviceName;
  if (properties.vendorID != 0x10de || (device_name.find("RTX 4") == std::string::npos &&
                                        device_name.find("RTX 5") == std::string::npos))
  {
    capabilities.reason = "DLSS Frame Generation requires an NVIDIA RTX 40 or RTX 50 GPU";
    return capabilities;
  }

  NGXVulkanRuntime &runtime = ngx_runtime();
  if (!runtime.frame_generation_available()) {
    capabilities.reason = runtime.error_get();
    return capabilities;
  }

  capabilities.supported = true;
  capabilities.active = active_sessions.load(std::memory_order_relaxed) > 0;
  capabilities.max_generated_frames = std::min(runtime.max_generated_frames_get(), 1);
  capabilities.reason = "DLSS Frame Generation 2x is available";
  return capabilities;
}

std::unique_ptr<FrameGenerationSession> vk_frame_generation_session_create()
{
  if (!vk_frame_generation_capabilities_get().supported) {
    return nullptr;
  }
  return std::make_unique<VKFrameGenerationSession>();
}

void vk_frame_generation_shutdown()
{
  if (active_sessions.load(std::memory_order_relaxed) != 0) {
    std::fprintf(stderr,
                 "DLSS Frame Generation: shutting down NGX while viewport sessions remain\n");
  }
  ngx_runtime().shutdown_runtime(VKBackend::get().device.vk_handle());
}

}  // namespace blender::gpu
