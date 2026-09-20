/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * Ray Reconstruction as reached through the GPU backend.
 *
 * Cycles owns a denoiser of the same name that goes to NGX over CUDA. There, the model of DLSS 4.5
 * refuses every evaluation with a platform error - on inputs identical to the ones the previous
 * model accepts - and no arrangement of those inputs changes it. The Vulkan path runs the same
 * model on the same library and hardware, which is why the denoiser comes here.
 *
 * The images belong to Vulkan and are written by CUDA: they are allocated with exportable memory,
 * handed to Cycles as system handles, and mapped there as surfaces the existing preprocess kernel
 * writes into. Nothing is copied between the two.
 */

#pragma once

#include <memory>
#include <string>

namespace blender::gpu {

class Texture;

/** One image of the set, described the way CUDA needs to map it. */
struct DlssImageHandle {
  /** System handle of the memory backing the image. Windows: a HANDLE; Linux: a file descriptor. */
  uint64_t handle = 0;
  /** Size of the whole allocation, which may hold more than this image. */
  uint64_t memory_size = 0;
  /** Where this image starts inside that allocation. */
  uint64_t memory_offset = 0;

  int width = 0;
  int height = 0;
  int channels = 0;
  /** Bytes per channel: 2 for half, 4 for float. */
  int channel_size = 0;
};

/** The whole set, in the order the denoiser fills them. */
struct DlssRayReconstructionImages {
  DlssImageHandle color;
  DlssImageHandle depth;
  DlssImageHandle diffuse_albedo;
  DlssImageHandle specular_albedo;
  DlssImageHandle normal_roughness;
  DlssImageHandle motion;
  DlssImageHandle specular_motion;

  /* The denoised frame, at output resolution. Shared as well, because a final render needs it back
   * in the buffer that gets saved rather than only on the screen. */
  DlssImageHandle output;
};

/** What changes from one evaluation to the next. */
struct DlssRayReconstructionFrame {
  float jitter_x = 0.0f;
  float jitter_y = 0.0f;
  /** Start the history over: the first frame, or a camera cut. */
  bool reset = false;
  /** Row-major 4x4, or all zeroes to leave them unset. */
  float world_to_view[16] = {};
  float view_to_clip[16] = {};
  bool have_camera_transforms = false;
};

/**
 * One denoiser, alive for as long as the resolution it was made for.
 *
 * Created and used from the thread that owns the GPU context - the images are Vulkan resources and
 * the evaluation is recorded into a command buffer.
 */
class DlssRayReconstructionSession {
 public:
  virtual ~DlssRayReconstructionSession() = default;

  /**
   * Make the images, or keep the ones already made if the sizes still match.
   *
   * Returns false and leaves `error_get()` worth reading if the set could not be made.
   */
  virtual bool ensure_images(int render_width,
                             int render_height,
                             int output_width,
                             int output_height,
                             int preset) = 0;

  /** The handles Cycles maps. Valid until the next call to `ensure_images` that remakes them. */
  virtual const DlssRayReconstructionImages &images() const = 0;

  /** Whether the last `ensure_images` made a new set, which invalidates every previous handle. */
  virtual bool images_were_remade() const = 0;

  /** Run the model. The inputs are whatever Cycles last wrote into the images. */
  virtual bool evaluate(const DlssRayReconstructionFrame &frame) = 0;

  /** The denoised frame, at output resolution. Owned by the session. */
  virtual Texture *output_texture_get() const = 0;

  virtual const std::string &error_get() const = 0;
};

/**
 * A session on whichever backend can run one, or null.
 *
 * Only the Vulkan backend can: the CUDA path of NGX does not run the model this exists for.
 */
std::unique_ptr<DlssRayReconstructionSession> dlss_ray_reconstruction_session_create();

/**
 * Evaluate once on textures made for the purpose and report what came back.
 *
 * Answers whether a given model runs on this path at all. Presets follow the SDK's numbering: 0
 * leaves the choice to NGX, 4 is D, 5 is E, 6 is F.
 */
std::string dlss_ray_reconstruction_probe(int preset);

}  // namespace blender::gpu
