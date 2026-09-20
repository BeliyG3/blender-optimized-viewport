/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "GPU_dlss_ray_reconstruction.hh"

#include "GPU_context.hh"

#if defined(WITH_DLSS_FRAME_GENERATION) && defined(WITH_VULKAN_BACKEND)
#  include "vulkan/vk_dlss_ray_reconstruction.hh"
#endif

namespace blender::gpu {

std::unique_ptr<DlssRayReconstructionSession> dlss_ray_reconstruction_session_create()
{
#if defined(WITH_DLSS_FRAME_GENERATION) && defined(WITH_VULKAN_BACKEND)
  if (GPU_backend_get_type() == GPU_BACKEND_VULKAN) {
    return vk_dlss_ray_reconstruction_session_create();
  }
#endif
  return nullptr;
}

std::string dlss_ray_reconstruction_probe(const int preset)
{
#if defined(WITH_DLSS_FRAME_GENERATION) && defined(WITH_VULKAN_BACKEND)
  if (GPU_backend_get_type() == GPU_BACKEND_VULKAN) {
    return vk_dlss_ray_reconstruction_probe(preset);
  }
  return "Ray Reconstruction over NGX requires the Vulkan backend";
#else
  UNUSED_VARS(preset);
  return "This Blender build does not include NGX support";
#endif
}

}  // namespace blender::gpu
