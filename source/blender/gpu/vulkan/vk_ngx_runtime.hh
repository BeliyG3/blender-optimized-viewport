/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * The NVIDIA NGX runtime, as reached from the Vulkan backend.
 *
 * NGX is initialized once per process and then serves every feature built on it - Frame
 * Generation, Ray Reconstruction - so the loading, the initialization and the wrapper that
 * describes a texture to it live here rather than beside any one of them.
 */

#pragma once

#ifdef WITH_DLSS_FRAME_GENERATION

#  include <mutex>
#  include <string>

#  include <vulkan/vulkan.h>

#  include <nvsdk_ngx_vk.h>

namespace blender::gpu {

class VKTexture;

/* The identifier this application is known by to NGX. Shared by every feature: a driver profile is
 * matched against it, so the two features must not disagree about who is asking. */
constexpr unsigned long long ngx_application_id = 100334311;

std::string ngx_error(const char *operation, NVSDK_NGX_Result result);

class NGXVulkanRuntime {
 public:
  /* _nvngx.dll is the driver NGX core, so use its Core -> Snippet Ext2 signature rather than
   * the SDK wrapper signature declared when NGX_SNIPPET_BUILD is not set.
   *
   * The last argument is the feature info, not a parameter block - the same slot the CUDA side
   * passes one through in `denoiser_dlss.cpp`, and the reason logging and the extra library search
   * path work there. It was declared as `NVSDK_NGX_Parameter *` here and passed as null, which is
   * why frame generation only ever found its library beside the executable. */
  using InitFunction = NVSDK_NGX_Result(NVSDK_CONV *)(unsigned long long,
                                                      const wchar_t *,
                                                      VkInstance,
                                                      VkPhysicalDevice,
                                                      VkDevice,
                                                      PFN_vkGetInstanceProcAddr,
                                                      PFN_vkGetDeviceProcAddr,
                                                      NVSDK_NGX_Version,
                                                      const NVSDK_NGX_FeatureCommonInfo *);
  using GetCapabilitiesFunction = decltype(&NVSDK_NGX_VULKAN_GetCapabilityParameters);
  using AllocateParametersFunction = decltype(&NVSDK_NGX_VULKAN_AllocateParameters);
  using DestroyParametersFunction = decltype(&NVSDK_NGX_VULKAN_DestroyParameters);
  using CreateFeatureFunction = decltype(&NVSDK_NGX_VULKAN_CreateFeature);
  using EvaluateFeatureFunction = decltype(&NVSDK_NGX_VULKAN_EvaluateFeature);
  using ReleaseFeatureFunction = decltype(&NVSDK_NGX_VULKAN_ReleaseFeature);
  using ShutdownFunction = decltype(&NVSDK_NGX_VULKAN_Shutdown1);

  InitFunction init = nullptr;
  GetCapabilitiesFunction get_capabilities = nullptr;
  AllocateParametersFunction allocate_parameters = nullptr;
  DestroyParametersFunction destroy_parameters = nullptr;
  CreateFeatureFunction create_feature = nullptr;
  EvaluateFeatureFunction evaluate_feature = nullptr;
  ReleaseFeatureFunction release_feature = nullptr;
  ShutdownFunction shutdown = nullptr;

  /* Load the core and initialize NGX against the current Vulkan device. Says nothing about whether
   * any particular feature is usable - ask the feature. */
  bool ensure_initialized();

  /* Whether Frame Generation is usable, and with how many generated frames. Asked separately
   * because a machine can carry one feature and not the other, and because Ray Reconstruction must
   * not be gated on the answer for frame generation. */
  bool frame_generation_available();
  int max_generated_frames_get() const;

  void shutdown_runtime(VkDevice device);

  const std::string &error_get() const;

 private:
  std::mutex mutex_;
  bool initialized_ = false;
  bool ngx_initialized_ = false;
  bool frame_generation_checked_ = false;
  bool frame_generation_available_ = false;
  int max_generated_frames_ = 0;
  std::string error_;
};

NGXVulkanRuntime &ngx_runtime();

/* Describe a texture to NGX as an image view, which is the form its Vulkan path takes. */
NVSDK_NGX_Resource_VK ngx_resource(VKTexture &texture, int width, int height, bool read_write);

}  // namespace blender::gpu

#endif /* WITH_DLSS_FRAME_GENERATION */
