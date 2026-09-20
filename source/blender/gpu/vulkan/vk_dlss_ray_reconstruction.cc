/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "vk_dlss_ray_reconstruction.hh"

#ifdef WITH_DLSS_FRAME_GENERATION

#  include <algorithm>
#  include <array>
#  include <cstdio>

/* Before the NGX headers, which name Vulkan types without declaring them. */
#  include <vulkan/vulkan.h>

/* The Vulkan entry points first: the denoiser headers name types it declares, and say so only by
 * using them. */
#  include <nvsdk_ngx_vk.h>

#  include <nvsdk_ngx_defs_dlssd.h>

#  include "GPU_texture.hh"

#  include "vk_backend.hh"
#  include "vk_context.hh"
#  include "vk_memory.hh"
#  include "vk_ngx_runtime.hh"
#  include "vk_texture.hh"

#  include "render_graph/nodes/vk_ray_reconstruction_node.hh"

namespace blender::gpu {

namespace {

/* The inputs are float32 because of what writes them.
 *
 * Cycles fills these images with the preprocess kernel it already has, and that kernel writes
 * float2 and float4 through CUDA surfaces. Half would halve the traffic and would mean rewriting
 * the kernel to match; that is a change worth making once the picture is known to be right, not
 * while establishing whether it arrives at all. */
constexpr TextureFormat color_format = TextureFormat::SFLOAT_32_32_32_32;
constexpr TextureFormat depth_format = TextureFormat::SFLOAT_32;
constexpr TextureFormat albedo_format = TextureFormat::SFLOAT_32_32_32_32;
constexpr TextureFormat normal_roughness_format = TextureFormat::SFLOAT_32_32_32_32;
constexpr TextureFormat motion_format = TextureFormat::SFLOAT_32_32;

/* The denoised frame is float32 like the inputs: it goes back to Cycles as well as to the screen,
 * and the kernel that reads it there reads float4 through a surface. */
constexpr TextureFormat output_format = TextureFormat::SFLOAT_32_32_32_32;

int channel_count(const TextureFormat format)
{
  switch (format) {
    case TextureFormat::SFLOAT_32:
      return 1;
    case TextureFormat::SFLOAT_32_32:
      return 2;
    default:
      return 4;
  }
}

class VKRayReconstruction : public DlssRayReconstructionSession {
 public:
  ~VKRayReconstruction() override
  {
    release_feature();
    release_images();

    if (parameters_ != nullptr) {
      ngx_runtime().destroy_parameters(parameters_);
      parameters_ = nullptr;
    }
  }

  bool ensure_images(const int render_width,
                     const int render_height,
                     const int output_width,
                     const int output_height,
                     const int preset) override
  {
    images_were_remade_ = false;

    if (!ngx_runtime().ensure_initialized()) {
      error_ = ngx_runtime().error_get();
      return false;
    }

    if (render_width == render_width_ && render_height == render_height_ &&
        output_width == output_width_ && output_height == output_height_ && preset == preset_ &&
        color_ != nullptr)
    {
      return true;
    }

    /* The feature is made for one pair of resolutions and one model; any of them changing means a
     * new feature, and new images for it to read. */
    release_feature();
    release_images();

    render_width_ = render_width;
    render_height_ = render_height;
    output_width_ = output_width;
    output_height_ = output_height;
    preset_ = preset;

    struct Request {
      const char *name;
      Texture **texture;
      TextureFormat format;
      DlssImageHandle *handle;
    };
    const Request requests[] = {
        {"DLSSD colour", &color_, color_format, &images_.color},
        {"DLSSD depth", &depth_, depth_format, &images_.depth},
        {"DLSSD diffuse albedo", &diffuse_albedo_, albedo_format, &images_.diffuse_albedo},
        {"DLSSD specular albedo", &specular_albedo_, albedo_format, &images_.specular_albedo},
        {"DLSSD normal and roughness",
         &normal_roughness_,
         normal_roughness_format,
         &images_.normal_roughness},
        {"DLSSD motion", &motion_, motion_format, &images_.motion},
        {"DLSSD specular motion", &specular_motion_, motion_format, &images_.specular_motion},
    };

    /* Exportable, because CUDA writes them: the memory is Vulkan's, and Cycles maps it as a
     * surface rather than copying anything across. */
    const eGPUTextureUsage input_usage = GPU_TEXTURE_USAGE_SHADER_READ |
                                         GPU_TEXTURE_USAGE_SHADER_WRITE |
                                         GPU_TEXTURE_USAGE_MEMORY_EXPORT;

    for (const Request &request : requests) {
      *request.texture = GPU_texture_create_2d(
          request.name, render_width, render_height, 1, request.format, input_usage, nullptr);
      if (*request.texture == nullptr) {
        error_ = "Could not allocate the Ray Reconstruction inputs";
        release_images();
        return false;
      }

      /* Cleared once, so that a frame evaluated before Cycles has written everything reads zeroes
       * rather than whatever the memory held. */
      const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      GPU_texture_clear(*request.texture, GPU_DATA_FLOAT, zero);

      const VKMemoryExport exported = unwrap(*request.texture)
                                          ->export_memory(external_handle_type);
      request.handle->handle = exported.handle;
      request.handle->memory_size = exported.memory_size;
      request.handle->memory_offset = exported.memory_offset;
      request.handle->width = render_width;
      request.handle->height = render_height;
      request.handle->channels = channel_count(request.format);
      request.handle->channel_size = 4;
    }

    output_ = GPU_texture_create_2d("DLSSD output",
                                    output_width,
                                    output_height,
                                    1,
                                    output_format,
                                    input_usage,
                                    nullptr);
    if (output_ == nullptr) {
      error_ = "Could not allocate the Ray Reconstruction output";
      release_images();
      return false;
    }

    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    GPU_texture_clear(output_, GPU_DATA_FLOAT, zero);

    const VKMemoryExport exported_output = unwrap(output_)->export_memory(external_handle_type);
    images_.output.handle = exported_output.handle;
    images_.output.memory_size = exported_output.memory_size;
    images_.output.memory_offset = exported_output.memory_offset;
    images_.output.width = output_width;
    images_.output.height = output_height;
    images_.output.channels = 4;
    images_.output.channel_size = 4;

    images_were_remade_ = true;
    return true;
  }

  const DlssRayReconstructionImages &images() const override
  {
    return images_;
  }

  bool images_were_remade() const override
  {
    return images_were_remade_;
  }

  bool evaluate(const DlssRayReconstructionFrame &frame) override
  {
    if (color_ == nullptr || output_ == nullptr) {
      error_ = "The Ray Reconstruction images have not been made";
      return false;
    }

    if (parameters_ == nullptr) {
      const NVSDK_NGX_Result result = ngx_runtime().allocate_parameters(&parameters_);
      if (NVSDK_NGX_FAILED(result) || parameters_ == nullptr) {
        error_ = ngx_error("Ray Reconstruction parameter allocation", result);
        return false;
      }
    }

    frame_ = frame;
    evaluated_ = false;

    render_graph::VKRayReconstructionNode::CreateInfo create_info = {};
    create_info.node_data.command = execute_callback;
    create_info.node_data.user_data = this;
    for (Texture *texture : input_textures()) {
      create_info.input_images[create_info.input_count++] = unwrap(texture)->vk_image_handle();
    }
    create_info.output_image = unwrap(output_)->vk_image_handle();

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
    return evaluated_;
  }

  Texture *output_texture_get() const override
  {
    return output_;
  }

  const std::string &error_get() const override
  {
    return error_;
  }

 private:
#  ifdef _WIN32
  static constexpr VkExternalMemoryHandleTypeFlagBits external_handle_type =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#  else
  static constexpr VkExternalMemoryHandleTypeFlagBits external_handle_type =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#  endif

  static void execute_callback(void *user_data, const VkCommandBuffer command_buffer)
  {
    static_cast<VKRayReconstruction *>(user_data)->execute(command_buffer);
  }

  std::array<Texture *, 7> input_textures() const
  {
    return {color_,
            depth_,
            diffuse_albedo_,
            specular_albedo_,
            normal_roughness_,
            motion_,
            specular_motion_};
  }

  void release_images()
  {
    for (Texture **texture : {&color_,
                              &depth_,
                              &diffuse_albedo_,
                              &specular_albedo_,
                              &normal_roughness_,
                              &motion_,
                              &specular_motion_,
                              &output_})
    {
      if (*texture != nullptr) {
        GPU_texture_free(*texture);
        *texture = nullptr;
      }
    }
    images_ = {};
  }

  void release_feature()
  {
    if (feature_ != nullptr) {
      ngx_runtime().release_feature(feature_);
      feature_ = nullptr;
    }
  }

  NVSDK_NGX_PerfQuality_Value quality_for_upscale() const
  {
    const float upscale = float(output_width_) / float(std::max(render_width_, 1));
    if (upscale <= 1.0f) {
      return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
    if (upscale <= 1.0f / 0.65f) {
      return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    }
    if (upscale <= 1.0f / 0.57f) {
      return NVSDK_NGX_PerfQuality_Value_Balanced;
    }
    if (upscale <= 1.0f / 0.5f) {
      return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    }
    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
  }

  bool create_feature(const VkCommandBuffer command_buffer)
  {
    parameters_->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
    parameters_->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
    parameters_->Set(NVSDK_NGX_Parameter_Width, unsigned(render_width_));
    parameters_->Set(NVSDK_NGX_Parameter_Height, unsigned(render_height_));
    parameters_->Set(NVSDK_NGX_Parameter_OutWidth, unsigned(output_width_));
    parameters_->Set(NVSDK_NGX_Parameter_OutHeight, unsigned(output_height_));
    parameters_->Set(NVSDK_NGX_Parameter_PerfQualityValue, quality_for_upscale());
    parameters_->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                     NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes);
    parameters_->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
    parameters_->Set(NVSDK_NGX_Parameter_DLSS_Denoise_Mode,
                     int(NVSDK_NGX_DLSS_Denoise_Mode_DLUnified));
    parameters_->Set(NVSDK_NGX_Parameter_DLSS_Roughness_Mode,
                     unsigned(NVSDK_NGX_DLSS_Roughness_Mode_Packed));
    parameters_->Set(NVSDK_NGX_Parameter_Use_HW_Depth, unsigned(NVSDK_NGX_DLSS_Depth_Type_Linear));

    /* All six quality modes, because the library reports the model per mode and setting five of
     * them leaves the sixth on whatever the driver last decided - which is how the interface came
     * to report a model nothing was using. */
    const NVSDK_NGX_RayReconstruction_Hint_Render_Preset preset =
        NVSDK_NGX_RayReconstruction_Hint_Render_Preset(preset_);
    parameters_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA, preset);
    parameters_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, preset);
    parameters_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, preset);
    parameters_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, preset);
    parameters_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
                     preset);
    parameters_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality,
                     preset);

    const NVSDK_NGX_Result result = ngx_runtime().create_feature(
        command_buffer, NVSDK_NGX_Feature_RayReconstruction, parameters_, &feature_);
    if (NVSDK_NGX_FAILED(result) || feature_ == nullptr) {
      error_ = ngx_error("Ray Reconstruction feature creation", result);
      feature_ = nullptr;
      return false;
    }
    return true;
  }

  void execute(const VkCommandBuffer command_buffer)
  {
    if (feature_ == nullptr && !create_feature(command_buffer)) {
      return;
    }

    NVSDK_NGX_Resource_VK color = ngx_resource(
        *unwrap(color_), render_width_, render_height_, false);
    NVSDK_NGX_Resource_VK depth = ngx_resource(
        *unwrap(depth_), render_width_, render_height_, false);
    NVSDK_NGX_Resource_VK diffuse_albedo = ngx_resource(
        *unwrap(diffuse_albedo_), render_width_, render_height_, false);
    NVSDK_NGX_Resource_VK specular_albedo = ngx_resource(
        *unwrap(specular_albedo_), render_width_, render_height_, false);
    NVSDK_NGX_Resource_VK normal_roughness = ngx_resource(
        *unwrap(normal_roughness_), render_width_, render_height_, false);
    NVSDK_NGX_Resource_VK motion = ngx_resource(
        *unwrap(motion_), render_width_, render_height_, false);
    NVSDK_NGX_Resource_VK specular_motion = ngx_resource(
        *unwrap(specular_motion_), render_width_, render_height_, false);
    NVSDK_NGX_Resource_VK output = ngx_resource(
        *unwrap(output_), output_width_, output_height_, true);

    parameters_->Set(NVSDK_NGX_Parameter_Color, &color);
    parameters_->Set(NVSDK_NGX_Parameter_Depth, &depth);
    parameters_->Set(NVSDK_NGX_Parameter_DiffuseAlbedo, &diffuse_albedo);
    parameters_->Set(NVSDK_NGX_Parameter_SpecularAlbedo, &specular_albedo);
    parameters_->Set(NVSDK_NGX_Parameter_GBuffer_Normals, &normal_roughness);
    parameters_->Set(NVSDK_NGX_Parameter_GBuffer_Roughness, &normal_roughness);
    parameters_->Set(NVSDK_NGX_Parameter_MotionVectors, &motion);
    parameters_->Set(NVSDK_NGX_Parameter_GBuffer_SpecularMvec, &specular_motion);
    parameters_->Set(NVSDK_NGX_Parameter_Output, &output);

    parameters_->Set(NVSDK_NGX_Parameter_Reset, frame_.reset ? 1 : 0);
    parameters_->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    parameters_->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    parameters_->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, frame_.jitter_x);
    parameters_->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, frame_.jitter_y);
    parameters_->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    parameters_->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    parameters_->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                     unsigned(render_width_));
    parameters_->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                     unsigned(render_height_));

    /* Depth and motion are screen-space numbers; these say what they mean in the world. The
     * storage is a member, so the pointers stay valid for as long as NGX reads them. */
    if (frame_.have_camera_transforms) {
      parameters_->Set(NVSDK_NGX_Parameter_DLSS_WORLD_TO_VIEW_MATRIX,
                       static_cast<void *>(frame_.world_to_view));
      parameters_->Set(NVSDK_NGX_Parameter_DLSS_VIEW_TO_CLIP_MATRIX,
                       static_cast<void *>(frame_.view_to_clip));
    }

    const NVSDK_NGX_Result result = ngx_runtime().evaluate_feature(
        command_buffer, feature_, parameters_, nullptr);
    if (NVSDK_NGX_FAILED(result)) {
      error_ = ngx_error("Ray Reconstruction evaluation", result);
      return;
    }
    evaluated_ = true;
  }

  int render_width_ = 0;
  int render_height_ = 0;
  int output_width_ = 0;
  int output_height_ = 0;
  int preset_ = -1;

  NVSDK_NGX_Parameter *parameters_ = nullptr;
  NVSDK_NGX_Handle *feature_ = nullptr;

  Texture *color_ = nullptr;
  Texture *depth_ = nullptr;
  Texture *diffuse_albedo_ = nullptr;
  Texture *specular_albedo_ = nullptr;
  Texture *normal_roughness_ = nullptr;
  Texture *motion_ = nullptr;
  Texture *specular_motion_ = nullptr;
  Texture *output_ = nullptr;

  DlssRayReconstructionImages images_;
  bool images_were_remade_ = false;

  DlssRayReconstructionFrame frame_;
  bool evaluated_ = false;
  std::string error_;
};

}  // namespace

std::unique_ptr<DlssRayReconstructionSession> vk_dlss_ray_reconstruction_session_create()
{
  return std::make_unique<VKRayReconstruction>();
}

std::string vk_dlss_ray_reconstruction_probe(const int preset)
{
  /* Sizes chosen only so that there is something to upscale; nothing here looks at the picture. */
  VKRayReconstruction session;
  if (!session.ensure_images(683, 384, 1024, 576, preset)) {
    return session.error_get();
  }

  DlssRayReconstructionFrame frame;
  frame.reset = true;
  const bool evaluated = session.evaluate(frame);

  char buffer[256];
  std::snprintf(buffer,
                sizeof(buffer),
                "Ray Reconstruction preset %d on Vulkan: %s",
                preset,
                evaluated ? "evaluated" : session.error_get().c_str());
  return buffer;
}

}  // namespace blender::gpu

#else /* WITH_DLSS_FRAME_GENERATION */

namespace blender::gpu {

std::unique_ptr<DlssRayReconstructionSession> vk_dlss_ray_reconstruction_session_create()
{
  return nullptr;
}

std::string vk_dlss_ray_reconstruction_probe(int /*preset*/)
{
  return "This build has no NGX support";
}

}  // namespace blender::gpu

#endif /* WITH_DLSS_FRAME_GENERATION */
