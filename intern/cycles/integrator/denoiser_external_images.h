/* SPDX-FileCopyrightText: 2026 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/types.h"

CCL_NAMESPACE_BEGIN

/* One image that lives somewhere else, described the way it has to be mapped.
 *
 * `handle` is what the system calls the memory behind it - a HANDLE on Windows, a file descriptor
 * elsewhere - and it is handed over: whoever maps it closes it. */
struct DenoiserExternalImage {
  uint64_t handle = 0;
  uint64_t memory_size = 0;
  uint64_t memory_offset = 0;
  int width = 0;
  int height = 0;
  int channels = 0;
};

/* The whole set, in the order the denoiser fills them. Empty when nothing external is in play.
 *
 * These exist because the model of DLSS 4.5 does not run on the CUDA path of NGX: it refuses every
 * evaluation there, before anything reaches the GPU. The evaluation happens on the display side
 * instead, and the inputs are written straight into memory that side allocated - no copies, and
 * the kernels that write them do not know the difference. */
struct DenoiserExternalImages {
  DenoiserExternalImage color;
  DenoiserExternalImage depth;
  DenoiserExternalImage diffuse_albedo;
  DenoiserExternalImage specular_albedo;
  DenoiserExternalImage normal_roughness;
  DenoiserExternalImage motion;
  DenoiserExternalImage specular_motion;

  /* Where the result lands. Shared as well, so that a final render can have it back in the buffer
   * that gets saved rather than only on the screen. */
  DenoiserExternalImage output;

  bool is_set() const
  {
    return color.handle != 0;
  }
};

CCL_NAMESPACE_END
