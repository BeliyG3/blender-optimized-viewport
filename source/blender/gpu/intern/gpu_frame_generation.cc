/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GPU_frame_generation.hh"

#include "GPU_context.hh"

#if defined(WITH_DLSS_FRAME_GENERATION) && defined(WITH_VULKAN_BACKEND)
#  include "vulkan/vk_frame_generation.hh"
#endif

namespace blender::gpu {

FrameGenerationCapabilities frame_generation_capabilities_get()
{
#if defined(WITH_DLSS_FRAME_GENERATION) && defined(WITH_VULKAN_BACKEND)
  if (GPU_backend_get_type() == GPU_BACKEND_VULKAN) {
    return vk_frame_generation_capabilities_get();
  }
  FrameGenerationCapabilities capabilities;
  capabilities.compiled = true;
  capabilities.backend = "OpenGL";
  capabilities.reason = "DLSS Frame Generation requires the Vulkan backend";
  return capabilities;
#else
  FrameGenerationCapabilities capabilities;
  capabilities.reason = "This Blender build does not include DLSS Frame Generation";
  return capabilities;
#endif
}

std::unique_ptr<FrameGenerationSession> frame_generation_session_create()
{
#if defined(WITH_DLSS_FRAME_GENERATION) && defined(WITH_VULKAN_BACKEND)
  if (GPU_backend_get_type() == GPU_BACKEND_VULKAN) {
    return vk_frame_generation_session_create();
  }
#endif
  return nullptr;
}

}  // namespace blender::gpu
