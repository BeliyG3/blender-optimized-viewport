/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "vk_node_info.hh"

namespace blender::gpu::render_graph {

using VKFrameGenerationCommandFn = void (*)(void *user_data, VkCommandBuffer command_buffer);

struct VKFrameGenerationData {
  VKFrameGenerationCommandFn command = nullptr;
  void *user_data = nullptr;
};

struct VKFrameGenerationCreateInfo {
  VKFrameGenerationData node_data;
  VkImage color_image = VK_NULL_HANDLE;
  VkImage depth_image = VK_NULL_HANDLE;
  VkImage motion_image = VK_NULL_HANDLE;
  VkImage generated_image = VK_NULL_HANDLE;
  VkImage real_image = VK_NULL_HANDLE;
};

class VKFrameGenerationNode : public VKNodeInfo<VKNodeType::FRAME_GENERATION,
                                                VKFrameGenerationCreateInfo,
                                                VKFrameGenerationData,
                                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                VKResourceType::IMAGE> {
 public:
  template<typename Node, typename Storage>
  static void set_node_data(Node &node, Storage & /*storage*/, const CreateInfo &create_info)
  {
    node.frame_generation = create_info.node_data;
  }

  void build_links(VKResourceStateTracker &resources,
                   VKRenderGraphLinks &links,
                   const CreateInfo &create_info) override
  {
    const VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    for (const VkImage image :
         {create_info.color_image, create_info.depth_image, create_info.motion_image})
    {
      ResourceWithStamp resource = resources.get_image(image);
      links.images.append(
          {{resource, VK_ACCESS_SHADER_READ_BIT}, VK_IMAGE_LAYOUT_GENERAL, aspect});
    }
    for (const VkImage image : {create_info.generated_image, create_info.real_image}) {
      ResourceWithStamp resource = resources.get_image_and_increase_stamp(image);
      links.images.append(
          {{resource, VK_ACCESS_SHADER_WRITE_BIT}, VK_IMAGE_LAYOUT_GENERAL, aspect});
    }
  }

  void build_commands(VKCommandBufferInterface &command_buffer,
                      Data &data,
                      Span<uint8_t> /*storage_push_constants*/,
                      VKBoundPipelines & /*r_bound_pipelines*/) override
  {
    const VkCommandBuffer native_command_buffer = command_buffer.native_handle_get();
    if (native_command_buffer != VK_NULL_HANDLE && data.command != nullptr) {
      data.command(data.user_data, native_command_buffer);
    }
  }
};

}  // namespace blender::gpu::render_graph
