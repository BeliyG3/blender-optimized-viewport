/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "vk_frame_generation_preprocess_infos.hh"
#include "gpu_shader_utildefines_lib.glsl"

COMPUTE_SHADER_CREATE_INFO(vk_frame_generation_preprocess)

void main()
{
  int2 texel = int2(gl_GlobalInvocationID.xy);
  int2 size = int2(imageSize(hardware_depth_img));
  if (any(greaterThanEqual(texel, size))) {
    return;
  }

  float linear_depth = imageLoad(linear_depth_img, texel).x;
  float hardware_depth = 1.0f;
  if (linear_depth > 0.0f && isfinite(linear_depth)) {
    if (orthographic) {
      hardware_depth = (linear_depth - near_clip) / max(far_clip - near_clip, 1e-8f);
    }
    else {
      float range = max(far_clip - near_clip, 1e-8f);
      hardware_depth = far_clip / range -
                       (far_clip * near_clip) / (range * linear_depth);
    }
  }
  imageStore(hardware_depth_img, texel, float4(saturate(hardware_depth)));

  /* Cycles' first motion pair is current-to-previous in pixel units. Keep an explicit RG output
   * so the backward pair cannot be consumed accidentally. */
  float2 motion = imageLoad(motion_img, texel).xy;
  imageStore(screen_motion_img, texel, float4(motion, 0.0f, 0.0f));
}
