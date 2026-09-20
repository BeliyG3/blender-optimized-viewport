/* SPDX-FileCopyrightText: 2021-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "GPU_compute.hh"
#include "GPU_context.hh"
#include "GPU_frame_generation.hh"
#include "GPU_immediate.hh"
#include "GPU_platform.hh"
#include "GPU_platform_backend_enum.h"
#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_dlss_ray_reconstruction.hh"
#include "GPU_texture.hh"
#include "GPU_viewport.hh"

#include "DNA_view3d_types.h"

#include "RE_engine.h"

#include "blender/display_driver.h"

#include "integrator/frame_generation_scheduler.h"

#include "util/log.h"
#include "util/math.h"
#include "util/time.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

/* Written by the main thread when a sync is accepted, read by the render thread when it delivers
 * pixels. Relaxed on purpose: it labels a diagnostic sample, and being one frame stale in a label
 * costs nothing, whereas ordering it would put a barrier on the delivery path. */
static std::atomic<int> g_accepted_scene_frame{-1};

void note_accepted_scene_frame(const int frame)
{
  g_accepted_scene_frame.store(frame, std::memory_order_relaxed);
}

int accepted_scene_frame()
{
  return g_accepted_scene_frame.load(std::memory_order_relaxed);
}

/* --------------------------------------------------------------------
 * BlenderDisplayShader.
 */

unique_ptr<BlenderDisplayShader> BlenderDisplayShader::create(blender::RenderEngine &b_engine,
                                                              blender::Scene &b_scene)
{
  /* See #engine_support_display_space_shader in rna_render.cc. */
  return make_unique<BlenderDisplaySpaceShader>(b_engine, b_scene);
}

int BlenderDisplayShader::get_position_attrib_location()
{
  if (position_attribute_location_ == -1) {
    blender::gpu::Shader *shader_program = get_shader_program();
    position_attribute_location_ = blender::GPU_shader_get_attribute(shader_program,
                                                                     position_attribute_name);
  }
  return position_attribute_location_;
}

int BlenderDisplayShader::get_tex_coord_attrib_location()
{
  if (tex_coord_attribute_location_ == -1) {
    blender::gpu::Shader *shader_program = get_shader_program();
    tex_coord_attribute_location_ = blender::GPU_shader_get_attribute(shader_program,
                                                                      tex_coord_attribute_name);
  }
  return tex_coord_attribute_location_;
}

/* --------------------------------------------------------------------
 * BlenderDisplaySpaceShader.
 */

BlenderDisplaySpaceShader::BlenderDisplaySpaceShader(blender::RenderEngine &b_engine,
                                                     blender::Scene &b_scene)
    : b_engine_(b_engine), b_scene_(b_scene)
{
}

blender::gpu::Shader *BlenderDisplaySpaceShader::bind(int /*width*/, int /*height*/)
{
  blender::gpu::Shader *shader = blender::GPU_shader_get_builtin_shader(
      blender::GPU_SHADER_3D_IMAGE);
  blender::GPU_shader_bind(shader);
  /** \note "image" binding slot is 0. */
  return blender::GPU_shader_get_bound();
}

void BlenderDisplaySpaceShader::unbind()
{
  blender::GPU_shader_unbind();
}

blender::gpu::Shader *BlenderDisplaySpaceShader::get_shader_program()
{
  if (!shader_program_) {
    shader_program_ = blender::GPU_shader_get_bound();
  }
  if (!shader_program_) {
    LOG_ERROR << "Error retrieving shader program for display space shader.";
  }

  return shader_program_;
}

/* --------------------------------------------------------------------
 * DrawTile.
 */

/* Higher level representation of a texture from the graphics library. */
class DisplayGPUTexture {
 public:
  /* Global counter for all allocated blender::GPUTextures used by instances of this class. */
  static inline std::atomic<int> num_used = 0;

  DisplayGPUTexture() = default;

  ~DisplayGPUTexture()
  {
    assert(gpu_texture == nullptr);
  }

  DisplayGPUTexture(const DisplayGPUTexture &other) = delete;
  DisplayGPUTexture &operator=(DisplayGPUTexture &other) = delete;

  DisplayGPUTexture(DisplayGPUTexture &&other) noexcept
      : gpu_texture(other.gpu_texture),
        width(other.width),
        height(other.height),
        format(other.format)
  {
    other.reset();
  }

  DisplayGPUTexture &operator=(DisplayGPUTexture &&other)
  {
    if (this == &other) {
      return *this;
    }

    gpu_texture = other.gpu_texture;
    width = other.width;
    height = other.height;
    format = other.format;

    other.reset();

    return *this;
  }

  bool gpu_resources_ensure(const uint texture_width,
                            const uint texture_height,
                            const blender::gpu::TextureFormat texture_format =
                                blender::gpu::TextureFormat::SFLOAT_16_16_16_16)
  {
    if (width != texture_width || height != texture_height || format != texture_format) {
      gpu_resources_destroy();
    }

    if (gpu_texture) {
      return true;
    }

    width = texture_width;
    height = texture_height;
    format = texture_format;

    /* Texture must have a minimum size of 1x1. */
    gpu_texture = blender::GPU_texture_create_2d("CyclesBlitTexture",
                                                 max(width, 1),
                                                 max(height, 1),
                                                 1,
                                                 format,
                                                 blender::GPU_TEXTURE_USAGE_GENERAL,
                                                 nullptr);

    if (!gpu_texture) {
      LOG_ERROR << "Error creating texture.";
      return false;
    }

    blender::GPU_texture_filter_mode(gpu_texture, false);
    blender::GPU_texture_extend_mode(gpu_texture, blender::GPU_SAMPLER_EXTEND_MODE_EXTEND);

    ++num_used;

    return true;
  }

  void gpu_resources_destroy()
  {
    if (gpu_texture == nullptr) {
      return;
    }

    GPU_TEXTURE_FREE_SAFE(gpu_texture);

    reset();

    --num_used;
  }

  /* Texture resource allocated by the blender::GPU module.
   *
   * NOTE: Allocated on the render engine's context. */
  blender::gpu::Texture *gpu_texture = nullptr;

  /* Dimensions of the texture in pixels. */
  int width = 0;
  int height = 0;
  blender::gpu::TextureFormat format = blender::gpu::TextureFormat::Invalid;

 protected:
  void reset()
  {
    gpu_texture = nullptr;
    width = 0;
    height = 0;
    format = blender::gpu::TextureFormat::Invalid;
  }
};

/* Higher level representation of a Pixel Buffer Object (PBO) from the graphics library. */
class DisplayGPUPixelBuffer {
 public:
  /* Global counter for all allocated blender::GPU module PBOs used by instances of this class. */
  static inline std::atomic<int> num_used = 0;

  DisplayGPUPixelBuffer() = default;

  ~DisplayGPUPixelBuffer()
  {
    assert(gpu_pixel_buffer == nullptr);
  }

  DisplayGPUPixelBuffer(const DisplayGPUPixelBuffer &other) = delete;
  DisplayGPUPixelBuffer &operator=(DisplayGPUPixelBuffer &other) = delete;

  DisplayGPUPixelBuffer(DisplayGPUPixelBuffer &&other) noexcept
      : gpu_pixel_buffer(other.gpu_pixel_buffer),
        width(other.width),
        height(other.height),
        bytes_per_pixel(other.bytes_per_pixel)
  {
    other.reset();
  }

  DisplayGPUPixelBuffer &operator=(DisplayGPUPixelBuffer &&other)
  {
    if (this == &other) {
      return *this;
    }

    gpu_pixel_buffer = other.gpu_pixel_buffer;
    width = other.width;
    height = other.height;
    bytes_per_pixel = other.bytes_per_pixel;

    other.reset();

    return *this;
  }

  bool gpu_resources_ensure(const uint new_width,
                            const uint new_height,
                            bool &buffer_recreated,
                            const size_t new_bytes_per_pixel = sizeof(half4))
  {
    buffer_recreated = false;

    const size_t required_size = new_bytes_per_pixel * new_width * new_height;

    /* Try to re-use the existing PBO if it has usable size. */
    if (gpu_pixel_buffer) {
      if (new_width != width || new_height != height || new_bytes_per_pixel != bytes_per_pixel ||
          blender::GPU_pixel_buffer_size(gpu_pixel_buffer) < required_size)
      {
        buffer_recreated = true;
        gpu_resources_destroy();
      }
    }

    /* Update size. */
    width = new_width;
    height = new_height;
    bytes_per_pixel = new_bytes_per_pixel;

    /* Create pixel buffer if not already created. */
    if (!gpu_pixel_buffer) {
      gpu_pixel_buffer = blender::GPU_pixel_buffer_create(required_size);
      buffer_recreated = true;
    }

    if (gpu_pixel_buffer == nullptr) {
      LOG_ERROR << "Error creating texture pixel buffer object.";
      return false;
    }

    ++num_used;

    return true;
  }

  void gpu_resources_destroy()
  {
    if (!gpu_pixel_buffer) {
      return;
    }

    blender::GPU_pixel_buffer_free(gpu_pixel_buffer);
    gpu_pixel_buffer = nullptr;

    reset();

    --num_used;
  }

  /* Pixel Buffer Object allocated by the blender::GPU module.
   *
   * NOTE: Allocated on the render engine's context. */
  blender::GPUPixelBuffer *gpu_pixel_buffer = nullptr;

  /* Dimensions of the PBO. */
  int width = 0;
  int height = 0;
  size_t bytes_per_pixel = 0;

 protected:
  void reset()
  {
    gpu_pixel_buffer = nullptr;
    width = 0;
    height = 0;
    bytes_per_pixel = 0;
  }
};

class DrawTile {
 public:
  DrawTile() = default;
  ~DrawTile() = default;

  DrawTile(const DrawTile &other) = delete;
  DrawTile &operator=(const DrawTile &other) = delete;

  DrawTile(DrawTile &&other) noexcept = default;

  DrawTile &operator=(DrawTile &&other) = default;

  void gpu_resources_destroy()
  {
    texture.gpu_resources_destroy();
  }

  bool ready_to_draw() const
  {
    return texture.gpu_texture != nullptr;
  }

  /* Texture which contains pixels of the tile. */
  DisplayGPUTexture texture;

  /* Display parameters the texture of this tile has been updated for. */
  BlenderDisplayDriver::Params params;
};

class DrawTileAndPBO {
 public:
  void gpu_resources_destroy()
  {
    tile.gpu_resources_destroy();
    buffer_object.gpu_resources_destroy();
  }

  DrawTile tile;
  DisplayGPUPixelBuffer buffer_object;
  bool need_update_texture_pixels = false;
};

/* --------------------------------------------------------------------
 * BlenderDisplayDriver.
 */

struct BlenderDisplayDriver::Tiles {
  /* Resources of a tile which is being currently rendered. */
  DrawTileAndPBO current_tile;

  /* All tiles which rendering is finished and which content will not be changed. */
  struct {
    vector<DrawTile> tiles;

    void gl_resources_destroy_and_clear()
    {
      for (DrawTile &tile : tiles) {
        tile.gpu_resources_destroy();
      }

      tiles.clear();
    }
  } finished_tiles;
};

struct BlenderDisplayDriver::DlssDenoiserState {
  /* Null until the first frame that asks for it, and null for good on a backend that cannot run
   * the model - which is every backend but Vulkan. */
  std::unique_ptr<blender::gpu::DlssRayReconstructionSession> session;
  bool unavailable = false;

};

struct BlenderDisplayDriver::FrameGenerationState {
  bool enabled = false;
  bool guides_active = false;
  bool guides_ready = false;
  bool permanently_failed = false;
  uint64_t frame_id = 0;
  int guide_width = 0;
  int guide_height = 0;

  DisplayGPUPixelBuffer linear_depth_buffer;
  DisplayGPUPixelBuffer motion_buffer;
  DisplayGPUTexture linear_depth_texture;
  DisplayGPUTexture motion_texture;
  DisplayGPUTexture hardware_depth_texture;
  DisplayGPUTexture screen_motion_texture;

  blender::gpu::FrameGenerationCamera camera;
  BlenderDisplayDriver::Params last_evaluated_params;
  BlenderDisplayDriver::Params pending_params;
  ViewportFrameGenerationScheduler scheduler;
  ViewportFrameGenerationScheduler::Decision pending_decision;
  uint64_t pending_revision = 0;
  bool pending_valid = false;
  bool retained_frame_available = false;
  blender::gpu::Shader *preprocess_shader = nullptr;

  void gpu_resources_destroy()
  {
    pending_valid = false;
    retained_frame_available = false;
    if (preprocess_shader != nullptr) {
      blender::GPU_shader_free(preprocess_shader);
      preprocess_shader = nullptr;
    }
    linear_depth_buffer.gpu_resources_destroy();
    motion_buffer.gpu_resources_destroy();
    linear_depth_texture.gpu_resources_destroy();
    motion_texture.gpu_resources_destroy();
    hardware_depth_texture.gpu_resources_destroy();
    screen_motion_texture.gpu_resources_destroy();
  }
};

BlenderDisplayDriver::BlenderDisplayDriver(blender::RenderEngine &b_engine,
                                           blender::Scene &b_scene,
                                           blender::RegionView3D *b_rv3d,
                                           const bool background)
    : b_engine_(b_engine),
      b_rv3d_(b_rv3d),
      background_(background),
      display_shader_(BlenderDisplayShader::create(b_engine, b_scene)),
      tiles_(make_unique<Tiles>()),
      frame_generation_(make_unique<FrameGenerationState>()),
      dlss_denoiser_(make_unique<DlssDenoiserState>())
{
  /* Create context while on the main thread. */
  gpu_context_create();
}

BlenderDisplayDriver::~BlenderDisplayDriver()
{
  gpu_resources_destroy();
}

/* --------------------------------------------------------------------
 * Update procedure.
 */

void BlenderDisplayDriver::next_tile_begin()
{
  if (!tiles_->current_tile.tile.ready_to_draw()) {
    LOG_ERROR
        << "Unexpectedly moving to the next tile without any data provided for current tile.";
    return;
  }

  /* Moving to the next tile without giving render data for the current tile is not an expected
   * situation. */
  DCHECK(!need_zero_);
  /* Texture should have been updated from the PBO at this point. */
  DCHECK(!tiles_->current_tile.need_update_texture_pixels);

  tiles_->finished_tiles.tiles.emplace_back(std::move(tiles_->current_tile.tile));
}

bool BlenderDisplayDriver::update_begin(const Params &params,
                                        const int texture_width,
                                        const int texture_height)
{
  /* Note that it's the responsibility of BlenderDisplayDriver to ensure updating and drawing
   * the texture does not happen at the same time. This is achieved indirectly.
   *
   * When enabling the OpenGL/GPU context, it uses an internal mutex lock DST.gpu_context_lock.
   * This same lock is also held when do_draw() is called, which together ensure mutual
   * exclusion.
   *
   * This locking is not performed on the Cycles side, because that would cause lock inversion. */
  if (!gpu_context_enable()) {
    return false;
  }

  /* Note: The render window might not draw between tiles. Wait for the previous
   * PBO-to-texture copy before reusing the PBO for the next tile. */
  blender::GPU_fence_wait(gpu_upload_sync_);
  blender::GPU_fence_wait(gpu_render_sync_);

  DrawTile &current_tile = tiles_->current_tile.tile;
  DisplayGPUPixelBuffer &current_tile_buffer_object = tiles_->current_tile.buffer_object;

  /* Clear storage of all finished tiles when display clear is requested.
   * Do it when new tile data is provided to handle the display clear flag in a single place.
   * It also makes the logic reliable from the whether drawing did happen or not point of view. */
  if (need_zero_) {
    tiles_->finished_tiles.gl_resources_destroy_and_clear();
    need_zero_ = false;
  }

  /* Update PBO dimensions if needed.
   *
   * NOTE: Allocate the PBO for the size which will fit the final render resolution (as in,
   * at a resolution divider 1. This was we don't need to recreate graphics interoperability
   * objects which are costly and which are tied to the specific underlying buffer size.
   * The downside of this approach is that when graphics interoperability is not used we are
   * sending too much data to blender::GPU when resolution divider is not 1. */
  /* TODO(sergey): Investigate whether keeping the PBO exact size of the texture makes non-interop
   * mode faster. */
  const int buffer_width = params.size.x;
  const int buffer_height = params.size.y;
  bool interop_recreated = false;

  if (!current_tile_buffer_object.gpu_resources_ensure(
          buffer_width, buffer_height, interop_recreated) ||
      !current_tile.texture.gpu_resources_ensure(texture_width, texture_height))
  {
    graphics_interop_buffer_.clear();
    tiles_->current_tile.gpu_resources_destroy();
    gpu_context_disable();
    return false;
  }

  if (interop_recreated) {
    graphics_interop_buffer_.clear();
  }

  /* Store an updated parameters of the current tile.
   * In theory it is only needed once per update of the tile, but doing it on every update is
   * the easiest and is not expensive. */
  tiles_->current_tile.tile.params = params;

  return true;
}

static void update_tile_texture_pixels(const DrawTileAndPBO &tile)
{
  const DisplayGPUTexture &texture = tile.tile.texture;

  if (!DCHECK_NOTNULL(tile.buffer_object.gpu_pixel_buffer)) {
    LOG_ERROR << "Display driver tile pixel buffer unavailable.";
    return;
  }
  blender::GPU_texture_update_sub_from_pixel_buffer(texture.gpu_texture,
                                                    blender::GPU_DATA_HALF_FLOAT,
                                                    tile.buffer_object.gpu_pixel_buffer,
                                                    0,
                                                    0,
                                                    0,
                                                    texture.width,
                                                    texture.height,
                                                    0);
}

double BlenderDisplayDriver::get_delivered_fps() const
{
  return delivered_fps_.load(std::memory_order_relaxed);
}

void BlenderDisplayDriver::update_end()
{
  /* Time between deliveries of new pixels to the viewport. Smoothed, because a single interval is
   * dominated by whichever work happened to finish; the exponential factor keeps roughly the last
   * second visible. A gap longer than a second means rendering had stopped, so restart rather than
   * average across the pause. */
  {
    const double now = time_dt();
    if (last_delivery_time_ != 0.0) {
      const double delta = now - last_delivery_time_;
      if (delta > 0.0 && delta < 1.0) {
        const double instant_fps = 1.0 / delta;
        const double previous = delivered_fps_.load(std::memory_order_relaxed);
        const double smoothed = (previous > 0.0) ? previous * 0.8 + instant_fps * 0.2 : instant_fps;
        delivered_fps_.store(smoothed, std::memory_order_relaxed);
      }
      else if (delta >= 1.0) {
        delivered_fps_.store(0.0, std::memory_order_relaxed);
      }

      /* The same interval, kept rather than smoothed away. Averaging answers "how fast"; the owner
       * is asking "how evenly", and those are different questions - 18 fps with even spacing and 18
       * fps alternating 30/80 ms look nothing alike. Paired with the scene frame that was last
       * accepted, because the eye judges the step in the animation, not the wall clock: a sequence
       * of deliveries reading 1,1,2,1,2 scene frames apart is jerky however even the milliseconds
       * are. Printed in batches so the series survives; nothing is allocated here. */
      if (report_deliveries_) {
        delivery_ms_[delivery_count_ % std::size(delivery_ms_)] = delta * 1000.0;
        delivery_cfra_[delivery_count_ % std::size(delivery_cfra_)] = accepted_scene_frame();
        delivery_count_++;

        if (delivery_count_ % std::size(delivery_ms_) == 0) {
          fprintf(stderr, "DELIVERY n=%d dt=", int(std::size(delivery_ms_)));
          for (const double sample : delivery_ms_) {
            fprintf(stderr, "%.1f,", sample);
          }
          fprintf(stderr, " cfra=");
          for (const int frame : delivery_cfra_) {
            fprintf(stderr, "%d,", frame);
          }
          fprintf(stderr, "\n");
          fflush(stderr);
        }
      }
    }
    last_delivery_time_ = now;
  }

  /* Unpack the PBO into the texture as soon as the new content is provided.
   *
   * This allows to ensure that the unpacking happens while resources like graphics interop (which
   * lifetime is outside of control of the display driver) are still valid, as well as allows to
   * move the tile from being current to finished blender::immediately after this call.
   *
   * One concern with this approach is that if the update happens more often than drawing then
   * doing the unpack here occupies blender::GPU transfer for no good reason. However, the render
   * scheduler takes care of ensuring updates don't happen that often. In regular applications
   * redraw will happen much more often than this update.
   *
   * On some older blender::GPUs on macOS, there is a driver crash when updating the texture for
   * viewport renders while Blender is drawing. As a workaround update texture during draw, under
   * assumption that there is no graphics interop on macOS and viewport render has a single tile.
   */
  if (!background_ && blender::GPU_type_matches_ex(blender::GPU_DEVICE_NVIDIA,
                                                   blender::GPU_OS_MAC,
                                                   blender::GPU_DRIVER_ANY,
                                                   blender::GPU_BACKEND_ANY))
  {
    tiles_->current_tile.need_update_texture_pixels = true;
  }
  else {
    update_tile_texture_pixels(tiles_->current_tile);
  }

  FrameGenerationState &frame_generation = *frame_generation_;
  const DrawTile &current_tile = tiles_->current_tile.tile;
  if (frame_generation.enabled && current_tile.ready_to_draw()) {
    frame_generation.retained_frame_available = true;
  }
  if (frame_generation.enabled && frame_generation.guides_ready &&
      !frame_generation.permanently_failed && current_tile.ready_to_draw())
  {
    const uint64_t revision = current_tile.params.render_revision;
    const bool mapping_changed =
        frame_generation.scheduler.has_evaluated_revision() &&
        (current_tile.params.full_size != frame_generation.last_evaluated_params.full_size ||
         current_tile.params.full_offset != frame_generation.last_evaluated_params.full_offset ||
         current_tile.params.size != frame_generation.last_evaluated_params.size);
    if (mapping_changed) {
      request_frame_generation_reset();
    }

    const ViewportFrameGenerationScheduler::Decision decision =
        frame_generation.scheduler.begin_real_frame(revision);
    if (decision.evaluate) {
      if (frame_generation.preprocess_shader == nullptr) {
        frame_generation.preprocess_shader = blender::GPU_shader_create_from_info_name(
            "vk_frame_generation_preprocess");
      }

      if (frame_generation.preprocess_shader == nullptr) {
        frame_generation.permanently_failed = true;
        LOG_WARNING << "DLSS Frame Generation unavailable; continuing with real viewport frames";
      }
      else {
        blender::gpu::Shader *shader = frame_generation.preprocess_shader;
        blender::GPU_shader_bind(shader);
        blender::GPU_shader_uniform_1f(shader, "near_clip", frame_generation.camera.near_clip);
        blender::GPU_shader_uniform_1f(shader, "far_clip", frame_generation.camera.far_clip);
        blender::GPU_shader_uniform_1i(
            shader, "orthographic", int(frame_generation.camera.orthographic));
        blender::GPU_texture_image_bind(frame_generation.linear_depth_texture.gpu_texture, 0);
        blender::GPU_texture_image_bind(frame_generation.motion_texture.gpu_texture, 1);
        blender::GPU_texture_image_bind(frame_generation.hardware_depth_texture.gpu_texture, 2);
        blender::GPU_texture_image_bind(frame_generation.screen_motion_texture.gpu_texture, 3);
        blender::GPU_compute_dispatch(shader,
                                      (frame_generation.guide_width + 15) / 16,
                                      (frame_generation.guide_height + 15) / 16,
                                      1);
        blender::GPU_texture_image_unbind_all();
        blender::GPU_shader_unbind();

        // The viewport owns DLSSG and evaluates it after color management.
        // Keep only the newest request when rendering advances faster than presentation.
        frame_generation.pending_decision = decision;
        frame_generation.pending_revision = revision;
        frame_generation.pending_params = current_tile.params;
        frame_generation.pending_valid = true;
      }
    }
  }

  /* Ensure blender::GPU fence exists to synchronize upload. */
  blender::GPU_fence_signal(gpu_upload_sync_);

  blender::GPU_flush();

  gpu_context_disable();

  has_update_cond_.notify_all();
}

/* --------------------------------------------------------------------
 * Texture buffer mapping.
 */

half4 *BlenderDisplayDriver::map_texture_buffer()
{
  /* With multi device rendering, Cycles can switch between using graphics interop
   * and not. For the denoised image it may be able to use graphics interop as that
   * buffer is written to by one device, while the noisy renders can not use it.
   *
   * We need to clear the graphics interop buffer on that switch, as blender::GPU_pixel_buffer_map
   * may recreate the buffer or handle. */
  graphics_interop_buffer_.clear();

  blender::GPUPixelBuffer *pix_buf = tiles_->current_tile.buffer_object.gpu_pixel_buffer;
  if (!DCHECK_NOTNULL(pix_buf)) {
    LOG_ERROR << "Display driver tile pixel buffer unavailable.";
    return nullptr;
  }
  half4 *mapped_rgba_pixels = reinterpret_cast<half4 *>(blender::GPU_pixel_buffer_map(pix_buf));
  if (!mapped_rgba_pixels) {
    LOG_ERROR << "Error mapping BlenderDisplayDriver pixel buffer object.";
  }
  return mapped_rgba_pixels;
}

void BlenderDisplayDriver::unmap_texture_buffer()
{
  blender::GPUPixelBuffer *pix_buf = tiles_->current_tile.buffer_object.gpu_pixel_buffer;
  if (!DCHECK_NOTNULL(pix_buf)) {
    LOG_ERROR << "Display driver tile pixel buffer unavailable.";
    return;
  }
  blender::GPU_pixel_buffer_unmap(pix_buf);
}

/* --------------------------------------------------------------------
 * Graphics interoperability.
 */

GraphicsInteropDevice BlenderDisplayDriver::graphics_interop_get_device()
{
  GraphicsInteropDevice interop_device;

  switch (blender::GPU_backend_get_type()) {
    case blender::GPU_BACKEND_OPENGL:
      interop_device.type = GraphicsInteropDevice::OPENGL;
      break;
    case blender::GPU_BACKEND_VULKAN:
      interop_device.type = GraphicsInteropDevice::VULKAN;
      break;
    case blender::GPU_BACKEND_METAL:
      interop_device.type = GraphicsInteropDevice::METAL;
      break;
    case blender::GPU_BACKEND_NONE:
    case blender::GPU_BACKEND_ANY:
      interop_device.type = GraphicsInteropDevice::NONE;
      break;
  }

  blender::Span<uint8_t> uuid = blender::GPU_platform_uuid();
  interop_device.uuid.resize(uuid.size());
  std::copy_n(uuid.data(), uuid.size(), interop_device.uuid.data());

  return interop_device;
}

void BlenderDisplayDriver::graphics_interop_update_buffer()
{
  if (graphics_interop_buffer_.is_empty()) {
    GraphicsInteropDevice::Type type = GraphicsInteropDevice::NONE;
    switch (blender::GPU_backend_get_type()) {
      case blender::GPU_BACKEND_OPENGL:
        type = GraphicsInteropDevice::OPENGL;
        break;
      case blender::GPU_BACKEND_VULKAN:
        type = GraphicsInteropDevice::VULKAN;
        break;
      case blender::GPU_BACKEND_METAL:
        type = GraphicsInteropDevice::METAL;
        break;
      case blender::GPU_BACKEND_NONE:
      case blender::GPU_BACKEND_ANY:
        break;
    }

    blender::GPUPixelBufferNativeHandle handle = blender::GPU_pixel_buffer_get_native_handle(
        tiles_->current_tile.buffer_object.gpu_pixel_buffer);
    graphics_interop_buffer_.assign(type, handle.handle, handle.size);
  }
}

bool BlenderDisplayDriver::dlss_denoiser_images_ensure(const int render_width,
                                                      const int render_height,
                                                      const int output_width,
                                                      const int output_height,
                                                      const int preset,
                                                      DenoiserExternalImages &r_images)
{
  DlssDenoiserState &state = *dlss_denoiser_;
  if (state.unavailable) {
    return false;
  }

  if (state.session == nullptr) {
    state.session = blender::gpu::dlss_ray_reconstruction_session_create();
    if (state.session == nullptr) {
      /* No Vulkan, so no model: the denoiser stays on its own path and this is never asked
       * again. */
      state.unavailable = true;
      return false;
    }
  }

  if (!state.session->ensure_images(
          render_width, render_height, output_width, output_height, preset))
  {
    LOG_ERROR << "DLSS Ray Reconstruction: " << state.session->error_get();
    state.unavailable = true;
    state.session.reset();
    return false;
  }

  /* Only when they are new: handing back the same handles a second time would have the renderer
   * map memory it has already mapped, and close a handle it no longer owns. */
  if (state.session->images_were_remade()) {
    const blender::gpu::DlssRayReconstructionImages &images = state.session->images();
    const struct {
      const blender::gpu::DlssImageHandle *from;
      DenoiserExternalImage *to;
    } bindings[] = {
        {&images.color, &r_images.color},
        {&images.depth, &r_images.depth},
        {&images.diffuse_albedo, &r_images.diffuse_albedo},
        {&images.specular_albedo, &r_images.specular_albedo},
        {&images.normal_roughness, &r_images.normal_roughness},
        {&images.motion, &r_images.motion},
        {&images.specular_motion, &r_images.specular_motion},
        {&images.output, &r_images.output},
    };
    for (const auto &binding : bindings) {
      binding.to->handle = binding.from->handle;
      binding.to->memory_size = binding.from->memory_size;
      binding.to->memory_offset = binding.from->memory_offset;
      binding.to->width = binding.from->width;
      binding.to->height = binding.from->height;
      binding.to->channels = binding.from->channels;
    }
  }

  return true;
}

bool BlenderDisplayDriver::dlss_denoiser_evaluate(const float jitter_x,
                                                  const float jitter_y,
                                                  const bool reset,
                                                  const float *world_to_view,
                                                  const float *view_to_clip)
{
  DlssDenoiserState &state = *dlss_denoiser_;
  if (state.session == nullptr) {
    return false;
  }

  blender::gpu::DlssRayReconstructionFrame frame;
  frame.jitter_x = jitter_x;
  frame.jitter_y = jitter_y;
  frame.reset = reset;
  if (world_to_view != nullptr && view_to_clip != nullptr) {
    std::copy_n(world_to_view, 16, frame.world_to_view);
    std::copy_n(view_to_clip, 16, frame.view_to_clip);
    frame.have_camera_transforms = true;
  }

  if (!state.session->evaluate(frame)) {
    LOG_ERROR << "DLSS Ray Reconstruction: " << state.session->error_get();
    return false;
  }

  /* Nothing else to do here: Cycles reads the result out of the shared image and writes it into
   * the render buffer, which is what both the screen and a saved frame are made from. */
  return true;
}

bool BlenderDisplayDriver::frame_generation_interop_begin(const int width, const int height)
{
  FrameGenerationState &state = *frame_generation_;
  state.guides_active = false;
  state.guides_ready = false;

  if (!state.enabled || state.permanently_failed ||
      blender::GPU_backend_get_type() != blender::GPU_BACKEND_VULKAN)
  {
    return false;
  }

  bool depth_recreated = false;
  bool motion_recreated = false;
  if (!state.linear_depth_buffer.gpu_resources_ensure(
          width, height, depth_recreated, sizeof(float)) ||
      !state.motion_buffer.gpu_resources_ensure(
          width, height, motion_recreated, sizeof(float) * 4) ||
      !state.linear_depth_texture.gpu_resources_ensure(
          width, height, blender::gpu::TextureFormat::SFLOAT_32) ||
      !state.motion_texture.gpu_resources_ensure(
          width, height, blender::gpu::TextureFormat::SFLOAT_32_32_32_32) ||
      !state.hardware_depth_texture.gpu_resources_ensure(
          width, height, blender::gpu::TextureFormat::SFLOAT_32) ||
      !state.screen_motion_texture.gpu_resources_ensure(
          width, height, blender::gpu::TextureFormat::SFLOAT_32_32))
  {
    return false;
  }

  if (depth_recreated) {
    frame_generation_depth_buffer_.clear();
  }
  if (motion_recreated) {
    frame_generation_motion_buffer_.clear();
  }

  if (state.guide_width != width || state.guide_height != height) {
    state.guide_width = width;
    state.guide_height = height;
    state.pending_valid = false;
    state.scheduler.request_reset();
  }
  state.guides_active = true;
  return true;
}

void BlenderDisplayDriver::frame_generation_interop_update_buffer(
    const FrameGenerationBuffer buffer)
{
  FrameGenerationState &state = *frame_generation_;
  DisplayGPUPixelBuffer &pixel_buffer = (buffer == FrameGenerationBuffer::DEPTH) ?
                                            state.linear_depth_buffer :
                                            state.motion_buffer;
  GraphicsInteropBuffer &interop_buffer = (buffer == FrameGenerationBuffer::DEPTH) ?
                                              frame_generation_depth_buffer_ :
                                              frame_generation_motion_buffer_;
  if (!interop_buffer.is_empty()) {
    return;
  }

  blender::GPUPixelBufferNativeHandle handle = blender::GPU_pixel_buffer_get_native_handle(
      pixel_buffer.gpu_pixel_buffer);
  interop_buffer.assign(GraphicsInteropDevice::VULKAN, handle.handle, handle.size);
}

void BlenderDisplayDriver::frame_generation_interop_end(const bool success)
{
  FrameGenerationState &state = *frame_generation_;
  if (!state.guides_active || !success) {
    state.guides_ready = false;
    return;
  }

  blender::GPU_texture_update_sub_from_pixel_buffer(state.linear_depth_texture.gpu_texture,
                                                    blender::GPU_DATA_FLOAT,
                                                    state.linear_depth_buffer.gpu_pixel_buffer,
                                                    0,
                                                    0,
                                                    0,
                                                    state.guide_width,
                                                    state.guide_height,
                                                    0);
  blender::GPU_texture_update_sub_from_pixel_buffer(state.motion_texture.gpu_texture,
                                                    blender::GPU_DATA_FLOAT,
                                                    state.motion_buffer.gpu_pixel_buffer,
                                                    0,
                                                    0,
                                                    0,
                                                    state.guide_width,
                                                    state.guide_height,
                                                    0);
  state.guides_ready = true;
}

void BlenderDisplayDriver::graphics_interop_activate()
{
  gpu_context_enable();
}

void BlenderDisplayDriver::graphics_interop_deactivate()
{
  gpu_context_disable();
}

/* --------------------------------------------------------------------
 * Drawing.
 */

void BlenderDisplayDriver::zero()
{
  need_zero_ = true;
}

void BlenderDisplayDriver::set_zoom(const float zoom_x, const float zoom_y)
{
  zoom_ = make_float2(zoom_x, zoom_y);
}

void BlenderDisplayDriver::set_frame_generation_enabled(const bool enabled)
{
  FrameGenerationState &state = *frame_generation_;
  if (state.enabled == enabled) {
    return;
  }
  state.enabled = enabled;
  state.pending_valid = false;
  state.retained_frame_available = false;
  b_engine_.viewport_frame_generation_hold_last_real = false;
  state.scheduler.request_reset();
  if (enabled) {
    state.permanently_failed = false;
  }
}

void BlenderDisplayDriver::set_frame_generation_camera(
    const blender::gpu::FrameGenerationCamera &camera)
{
  frame_generation_->camera = camera;
}

void BlenderDisplayDriver::request_frame_generation_reset()
{
  FrameGenerationState &state = *frame_generation_;
  state.pending_valid = false;
  state.retained_frame_available = false;
  b_engine_.viewport_frame_generation_hold_last_real = false;
  state.scheduler.request_reset();
}

/* Update vertex buffer with new coordinates of vertex positions and texture coordinates.
 * This buffer is used to render texture in the viewport.
 *
 * NOTE: The buffer needs to be bound. */
static void vertex_draw(const DisplayDriver::Params &params,
                        const int texcoord_attribute,
                        const int position_attribute)
{
  const int x = params.full_offset.x;
  const int y = params.full_offset.y;

  const int width = params.size.x;
  const int height = params.size.y;

  blender::immBegin(blender::GPU_PRIM_TRI_STRIP, 4);

  blender::immAttr2f(texcoord_attribute, 1.0f, 0.0f);
  blender::immVertex2f(position_attribute, x + width, y);

  blender::immAttr2f(texcoord_attribute, 1.0f, 1.0f);
  blender::immVertex2f(position_attribute, x + width, y + height);

  blender::immAttr2f(texcoord_attribute, 0.0f, 0.0f);
  blender::immVertex2f(position_attribute, x, y);

  blender::immAttr2f(texcoord_attribute, 0.0f, 1.0f);
  blender::immVertex2f(position_attribute, x, y + height);

  blender::immEnd();
}

static void draw_tile(const float2 &zoom,
                      const int texcoord_attribute,
                      const int position_attribute,
                      const DrawTile &draw_tile)
{
  if (!draw_tile.ready_to_draw()) {
    return;
  }

  const DisplayGPUTexture &texture = draw_tile.texture;

  if (!DCHECK_NOTNULL(texture.gpu_texture)) {
    LOG_ERROR << "Display driver tile blender::GPU texture resource unavailable.";
    return;
  }

  /* Trick to keep sharp rendering without jagged edges on all blender::GPUs.
   *
   * The idea here is to enforce driver to use linear interpolation when the image is zoomed out.
   * For the render result with a resolution divider in effect we always use nearest interpolation.
   *
   * Use explicit MIN assignment to make sure the driver does not have an undefined behavior at
   * the zoom level 1. The MAG filter is always NEAREST. */
  const float zoomed_width = draw_tile.params.size.x * zoom.x;
  const float zoomed_height = draw_tile.params.size.y * zoom.y;
  if (texture.width != draw_tile.params.size.x || texture.height != draw_tile.params.size.y) {
    /* Resolution divider is different from 1, force nearest interpolation. */
    blender::GPU_texture_bind_ex(
        texture.gpu_texture, blender::GPUSamplerState::default_sampler(), 0);
  }
  else if (zoomed_width - draw_tile.params.size.x > -0.5f ||
           zoomed_height - draw_tile.params.size.y > -0.5f)
  {
    blender::GPU_texture_bind_ex(
        texture.gpu_texture, blender::GPUSamplerState::default_sampler(), 0);
  }
  else {
    blender::GPU_texture_bind_ex(texture.gpu_texture, {blender::GPU_SAMPLER_FILTERING_LINEAR}, 0);
  }

  /* Draw at the parameters for which the texture has been updated for. This allows to always draw
   * texture during bordered-rendered camera view without flickering. The validness of the display
   * parameters for a texture is guaranteed by the initial "clear" state which makes drawing to
   * have an early output.
   *
   * Such approach can cause some extra "jelly" effect during panning, but it is not more jelly
   * than overlay of selected objects. Also, it's possible to redraw texture at an intersection of
   * the texture draw parameters and the latest updated draw parameters (although, complexity of
   * doing it might not worth it. */
  vertex_draw(draw_tile.params, texcoord_attribute, position_attribute);
}

void BlenderDisplayDriver::flush()
{
  /* This is called from the render thread that also calls update_begin/end, right before ending
   * the render loop. We wait for any queued PBO and render commands to be done, before destroying
   * the render thread and activating the context in the main thread to destroy resources.
   *
   * If we don't do this, the NVIDIA driver hangs for a few seconds for when ending 3D viewport
   * rendering, for unknown reasons. This was found with NVIDIA driver version 470.73 and a Quadro
   * RTX 6000 on Linux. */
  if (!gpu_context_enable()) {
    return;
  }

  blender::GPU_fence_wait(gpu_upload_sync_);
  blender::GPU_fence_wait(gpu_render_sync_);

  gpu_context_disable();
}

void BlenderDisplayDriver::draw(const Params &params)
{
  if (b_rv3d_ && (b_rv3d_->rflag & (blender::RV3D_NAVIGATING | blender::RV3D_PAINTING))) {
    /* Before drawing, wait that an update to the texture has actually occurred, to synchronize
     * rendering of Cycles with Blender. Use a timeout to prevent user interface in the main thread
     * from becoming unresponsive when rendering is too heavy. */
    thread_scoped_lock lock(has_update_mutex_);
    has_update_cond_.wait_for(lock, std::chrono::milliseconds(33));
    lock.unlock();
  }

  gpu_context_lock();

  FrameGenerationState &frame_generation = *frame_generation_;
  const bool retain_previous_real_this_draw = (need_zero_ || params.texture_outdated) &&
                                              frame_generation.enabled &&
                                              !frame_generation.permanently_failed &&
                                              frame_generation.retained_frame_available;
  if (need_zero_ && !retain_previous_real_this_draw) {
    /* Texture is requested to be cleared and was not yet cleared. Frame Generation keeps the
     * preceding complete tile visible until the next real input arrives. */
    gpu_context_unlock();
    return;
  }

  blender::GPU_fence_wait(gpu_upload_sync_);
  blender::GPU_blend(blender::GPU_BLEND_ALPHA_PREMULT);

  blender::gpu::Shader *active_shader = display_shader_->bind(params.full_size.x,
                                                              params.full_size.y);

  blender::GPUVertFormat *format = blender::immVertexFormat();
  const int texcoord_attribute = blender::GPU_vertformat_attr_add(
      format,
      ccl::BlenderDisplayShader::tex_coord_attribute_name,
      blender::gpu::VertAttrType::SFLOAT_32_32);
  const int position_attribute = blender::GPU_vertformat_attr_add(
      format,
      ccl::BlenderDisplayShader::position_attribute_name,
      blender::gpu::VertAttrType::SFLOAT_32_32);

  /* NOTE: Shader is bound again through IMM to register this shader with the IMM module
   * and perform required setup for IMM rendering. This is required as the IMM module
   * needs to be aware of which shader is bound, and the main display shader
   * is bound externally. */
  blender::immBindShader(active_shader);

  if (tiles_->current_tile.need_update_texture_pixels) {
    update_tile_texture_pixels(tiles_->current_tile);
    tiles_->current_tile.need_update_texture_pixels = false;
  }

  draw_tile(zoom_, texcoord_attribute, position_attribute, tiles_->current_tile.tile);

  for (const DrawTile &tile : tiles_->finished_tiles.tiles) {
    draw_tile(zoom_, texcoord_attribute, position_attribute, tile);
  }

  /* Reset IMM shader bind state. */
  blender::immUnbindProgram();

  display_shader_->unbind();

  if (frame_generation.enabled && !frame_generation.permanently_failed &&
      b_engine_.viewport != nullptr)
  {
    if (retain_previous_real_this_draw) {
      blender::GPU_viewport_frame_generation_hold_last_real(b_engine_.viewport);
    }
    else {
      blender::GPU_viewport_frame_generation_mark_active(b_engine_.viewport);
    }
    if (frame_generation.pending_valid) {
      blender::GPUViewportFrameGenerationInput input;
      input.depth = frame_generation.hardware_depth_texture.gpu_texture;
      input.motion = frame_generation.screen_motion_texture.gpu_texture;
      input.full_size = blender::int2(frame_generation.pending_params.full_size.x,
                                      frame_generation.pending_params.full_size.y);
      input.full_offset = blender::int2(frame_generation.pending_params.full_offset.x,
                                        frame_generation.pending_params.full_offset.y);
      input.region_size = blender::int2(frame_generation.pending_params.size.x,
                                        frame_generation.pending_params.size.y);
      input.render_size = blender::int2(frame_generation.guide_width,
                                        frame_generation.guide_height);
      input.revision = frame_generation.pending_revision;
      input.frame_id = frame_generation.frame_id + 1;
      input.reset = frame_generation.pending_decision.reset;
      input.present_generated = frame_generation.pending_decision.present_generated;
      input.camera = frame_generation.camera;

      const blender::GPUViewportFrameGenerationSubmitResult result =
          blender::GPU_viewport_frame_generation_submit(b_engine_.viewport, input);
      if (result == blender::GPUViewportFrameGenerationSubmitResult::ACCEPTED) {
        frame_generation.frame_id++;
        frame_generation.scheduler.evaluation_succeeded(frame_generation.pending_revision,
                                                        frame_generation.pending_decision);
        frame_generation.last_evaluated_params = frame_generation.pending_params;
        frame_generation.pending_valid = false;
        frame_generation.retained_frame_available = true;
        b_engine_.viewport_frame_generation_hold_last_real = true;
        if (frame_generation.pending_decision.present_generated) {
          b_engine_.flag |= blender::RE_ENGINE_DO_DRAW;
        }
      }
      else if (result == blender::GPUViewportFrameGenerationSubmitResult::FAILED) {
        frame_generation.permanently_failed = true;
        frame_generation.pending_valid = false;
        frame_generation.retained_frame_available = false;
        b_engine_.viewport_frame_generation_hold_last_real = false;
        frame_generation.scheduler.evaluation_failed();
        LOG_WARNING << "DLSS Frame Generation disabled; continuing with real viewport frames";
      }
    }
  }

  blender::GPU_blend(blender::GPU_BLEND_NONE);

  blender::GPU_fence_signal(gpu_render_sync_);
  blender::GPU_flush();

  gpu_context_unlock();

  LOG_TRACE << "Display driver number of textures: " << DisplayGPUTexture::num_used;
  LOG_TRACE << "Display driver number of PBOs: " << DisplayGPUPixelBuffer::num_used;
}

void BlenderDisplayDriver::gpu_context_create()
{
  if (!RE_engine_gpu_context_create(&b_engine_)) {
    LOG_ERROR << "Error creating blender::GPU context.";
    return;
  }

  /* Create global blender::GPU resources for display driver. */
  if (!gpu_resources_create()) {
    LOG_ERROR << "Error creating blender::GPU resources for Display Driver.";
    return;
  }
}

bool BlenderDisplayDriver::gpu_context_enable()
{
  return RE_engine_gpu_context_enable(&b_engine_);
}

void BlenderDisplayDriver::gpu_context_disable()
{
  RE_engine_gpu_context_disable(&b_engine_);
}

void BlenderDisplayDriver::gpu_context_destroy()
{
  RE_engine_gpu_context_destroy(&b_engine_);
}

void BlenderDisplayDriver::gpu_context_lock()
{
  RE_engine_gpu_context_lock(&b_engine_);
}

void BlenderDisplayDriver::gpu_context_unlock()
{
  RE_engine_gpu_context_unlock(&b_engine_);
}

bool BlenderDisplayDriver::gpu_resources_create()
{
  /* Ensure context is active for resource creation. */
  if (!gpu_context_enable()) {
    LOG_ERROR << "Error enabling blender::GPU context.";
    return false;
  }

  gpu_upload_sync_ = blender::GPU_fence_create();
  gpu_render_sync_ = blender::GPU_fence_create();

  if (!DCHECK_NOTNULL(gpu_upload_sync_) || !DCHECK_NOTNULL(gpu_render_sync_)) {
    LOG_ERROR << "Error creating blender::GPU synchronization primitives.";
    assert(0);
    return false;
  }

  gpu_context_disable();
  return true;
}

void BlenderDisplayDriver::gpu_resources_destroy()
{
  b_engine_.viewport_frame_generation_hold_last_real = false;
  gpu_context_enable();

  display_shader_.reset();

  graphics_interop_buffer_.clear();
  frame_generation_depth_buffer_.clear();
  frame_generation_motion_buffer_.clear();

  tiles_->current_tile.gpu_resources_destroy();
  tiles_->finished_tiles.gl_resources_destroy_and_clear();
  frame_generation_->gpu_resources_destroy();

  /* Fences. */
  if (gpu_render_sync_) {
    blender::GPU_fence_free(gpu_render_sync_);
    gpu_render_sync_ = nullptr;
  }
  if (gpu_upload_sync_) {
    blender::GPU_fence_free(gpu_upload_sync_);
    gpu_upload_sync_ = nullptr;
  }

  gpu_context_disable();

  gpu_context_destroy();
}

CCL_NAMESPACE_END
