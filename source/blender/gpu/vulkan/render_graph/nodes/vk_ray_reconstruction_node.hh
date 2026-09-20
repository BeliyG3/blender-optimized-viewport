/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "vk_node_info.hh"

namespace blender::gpu::render_graph {

using VKRayReconstructionCommandFn = void (*)(void *user_data, VkCommandBuffer command_buffer);

/**
 * Hands a command buffer to NGX so it can record Ray Reconstruction into it.
 *
 * The same shape as the Frame Generation node, and separate from it on purpose: that one carries
 * exactly the five images frame generation names, and Ray Reconstruction has nine. Kept apart
 * rather than generalized because frame generation works today and is not worth disturbing.
 */
struct VKRayReconstructionData {
  VKRayReconstructionCommandFn command = nullptr;
  void *user_data = nullptr;
};

struct VKRayReconstructionCreateInfo {
  /* Colour, depth, both albedos, normal, roughness, both motion vectors, and whatever a later
   * model asks for. A fixed array because the node data is copied into the graph, and a heap
   * allocation per evaluation would be paid on every viewport frame. */
  static constexpr int max_inputs = 12;

  VKRayReconstructionData node_data;
  VkImage input_images[max_inputs] = {};
  int input_count = 0;
  VkImage output_image = VK_NULL_HANDLE;
};

class VKRayReconstructionNode : public VKNodeInfo<VKNodeType::RAY_RECONSTRUCTION,
                                                  VKRayReconstructionCreateInfo,
                                                  VKRayReconstructionData,
                                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                  VKResourceType::IMAGE> {
 public:
  template<typename Node, typename Storage>
  static void set_node_data(Node &node, Storage & /*storage*/, const CreateInfo &create_info)
  {
    node.ray_reconstruction = create_info.node_data;
  }

  void build_links(VKResourceStateTracker &resources,
                   VKRenderGraphLinks &links,
                   const CreateInfo &create_info) override
  {
    const VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    for (int index = 0; index < create_info.input_count; index++) {
      const VkImage image = create_info.input_images[index];
      if (image == VK_NULL_HANDLE) {
        continue;
      }
      ResourceWithStamp resource = resources.get_image(image);
      links.images.append(
          {{resource, VK_ACCESS_SHADER_READ_BIT}, VK_IMAGE_LAYOUT_GENERAL, aspect});
    }
    if (create_info.output_image != VK_NULL_HANDLE) {
      ResourceWithStamp resource = resources.get_image_and_increase_stamp(
          create_info.output_image);
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
