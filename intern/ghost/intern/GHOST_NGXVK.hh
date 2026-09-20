/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#ifdef WITH_DLSS_FRAME_GENERATION

#  include <string>

#  include "BLI_vector.hh"

#  include <vulkan/vulkan.h>

namespace blender::ghost {

/**
 * Query the Vulkan extensions required by NVIDIA NGX Frame Generation.
 *
 * The queries intentionally happen before creating the Vulkan instance and device. Failure is
 * non-fatal for Blender: callers keep Vulkan available and expose Frame Generation as unavailable.
 */
bool ngx_vk_instance_extensions_get(Vector<std::string> &r_extensions, std::string &r_error);
bool ngx_vk_device_extensions_get(VkInstance instance,
                                  VkPhysicalDevice physical_device,
                                  Vector<std::string> &r_extensions,
                                  std::string &r_error);

}  // namespace blender::ghost

#endif
