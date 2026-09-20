/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <atomic>

#include "session/display_driver.h"

#include "util/thread.h"
#include "util/unique_ptr.h"

namespace blender {
struct GPUContext;
struct GPUFence;
struct RenderEngine;
struct Scene;
namespace gpu {
struct FrameGenerationCamera;
class Shader;
}  // namespace gpu
}  // namespace blender

CCL_NAMESPACE_BEGIN

/* The scene frame of the last sync this session accepted, for the delivery diagnostic below. The
 * display driver cannot reach the Blender scene, and the question it has to answer - whether two
 * consecutive deliveries showed consecutive poses - needs exactly this one number. Set from
 * `BlenderSession::synchronize`, read from the render thread. */
void note_accepted_scene_frame(int frame);
int accepted_scene_frame();

/* Base class of shader used for display driver rendering. */
class BlenderDisplayShader {
 public:
  static constexpr const char *position_attribute_name = "pos";
  static constexpr const char *tex_coord_attribute_name = "texCoord";

  /* Create shader implementation suitable for the given render engine and scene configuration. */
  static unique_ptr<BlenderDisplayShader> create(blender::RenderEngine &b_engine,
                                                 blender::Scene &b_scene);

  BlenderDisplayShader() = default;
  virtual ~BlenderDisplayShader() = default;

  virtual blender::gpu::Shader *bind(const int width, const int height) = 0;
  virtual void unbind() = 0;

  /* Get attribute location for position and texture coordinate respectively.
   * NOTE: The shader needs to be bound to have access to those. */
  virtual int get_position_attrib_location();
  virtual int get_tex_coord_attrib_location();

 protected:
  /* Get program of this display shader.
   * NOTE: The shader needs to be bound to have access to this. */
  virtual blender::gpu::Shader *get_shader_program() = 0;

  /* Cached values of various OpenGL resources. */
  int position_attribute_location_ = -1;
  int tex_coord_attribute_location_ = -1;
};

class BlenderDisplaySpaceShader : public BlenderDisplayShader {
 public:
  BlenderDisplaySpaceShader(blender::RenderEngine &b_engine, blender::Scene &b_scene);

  blender::gpu::Shader *bind(const int width, const int height) override;
  void unbind() override;

 protected:
  blender::gpu::Shader *get_shader_program() override;

  blender::RenderEngine &b_engine_;
  blender::Scene &b_scene_;

  /* Cached values of various OpenGL resources. */
  blender::gpu::Shader *shader_program_ = nullptr;
};

/* Display driver implementation which is specific for Blender viewport integration. */
class BlenderDisplayDriver : public DisplayDriver {
 public:
  BlenderDisplayDriver(blender::RenderEngine &b_engine,
                       blender::Scene &b_scene,
                       blender::RegionView3D *b_rv3d,
                       const bool background);
  ~BlenderDisplayDriver() override;

  void graphics_interop_activate() override;
  void graphics_interop_deactivate() override;

  void zero() override;

  void set_zoom(const float zoom_x, const float zoom_y);

  /* Rate at which rendered content actually reaches the viewport texture, in frames per second.
   *
   * This is not Blender's playback FPS, which only reports how fast the timeline advances and says
   * nothing about the renderer: during a Rendered viewport playback the timeline can run far ahead
   * while the same image is re-blitted. Measured in `update_end()`, so it counts deliveries of new
   * pixels and works whether or not an animation is playing. Zero until two updates have arrived. */
  double get_delivered_fps() const;

  void set_frame_generation_enabled(bool enabled);
  void set_frame_generation_camera(const blender::gpu::FrameGenerationCamera &camera);
  void request_frame_generation_reset();

 protected:
  void next_tile_begin() override;

  bool update_begin(const Params &params,
                    const int texture_width,
                    const int texture_height) override;
  void update_end() override;

  half4 *map_texture_buffer() override;
  void unmap_texture_buffer() override;

  GraphicsInteropDevice graphics_interop_get_device() override;
  void graphics_interop_update_buffer() override;
  bool frame_generation_interop_begin(int width, int height) override;

  /* Ray Reconstruction on the Vulkan side, where the model of DLSS 4.5 runs. The images belong
   * here and Cycles writes into them; the evaluation is made here once it has. */
  bool dlss_denoiser_images_ensure(int render_width,
                                   int render_height,
                                   int output_width,
                                   int output_height,
                                   int preset,
                                   DenoiserExternalImages &r_images) override;
  bool dlss_denoiser_evaluate(float jitter_x,
                              float jitter_y,
                              bool reset,
                              const float *world_to_view,
                              const float *view_to_clip) override;
  void frame_generation_interop_update_buffer(FrameGenerationBuffer buffer) override;
  void frame_generation_interop_end(bool success) override;

  void draw(const Params &params) override;

  void flush() override;

  /* Helper function which allocates new GPU context. */
  void gpu_context_create();
  bool gpu_context_enable();
  void gpu_context_disable();
  void gpu_context_destroy();
  void gpu_context_lock();
  void gpu_context_unlock();

  /* Create GPU resources used by the display driver. */
  bool gpu_resources_create();

  /* Destroy all GPU resources which are being used by this object. */
  void gpu_resources_destroy();

  blender::RenderEngine &b_engine_;
  blender::RegionView3D *b_rv3d_;
  bool background_;

  /* Content of the display is to be filled with zeroes. */
  std::atomic<bool> need_zero_ = true;

  /* Written in `update_end()` on the render thread, read from the status line on another thread. */
  std::atomic<double> delivered_fps_ = 0.0;
  double last_delivery_time_ = 0.0;

  /* The intervals themselves, kept instead of smoothed, with the scene frame each delivery carried.
   * Answers how evenly rather than how fast, and separates two different defects: uneven spacing in
   * time, and even spacing that skips scene frames. Render thread only, diagnostic. */
  bool report_deliveries_ = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  double delivery_ms_[256] = {};
  int delivery_cfra_[256] = {};
  int delivery_count_ = 0;

  unique_ptr<BlenderDisplayShader> display_shader_;

  /* Opaque storage for an internal state and data for tiles. */
  struct Tiles;
  unique_ptr<Tiles> tiles_;

  struct FrameGenerationState;
  unique_ptr<FrameGenerationState> frame_generation_;

  /* The denoiser that runs on the display side, and whether its output is waiting to be drawn. */
  struct DlssDenoiserState;
  unique_ptr<DlssDenoiserState> dlss_denoiser_;

  blender::GPUFence *gpu_render_sync_ = nullptr;
  blender::GPUFence *gpu_upload_sync_ = nullptr;

  float2 zoom_ = make_float2(1.0f, 1.0f);

  thread_condition_variable has_update_cond_;
  thread_mutex has_update_mutex_;
};

CCL_NAMESPACE_END
