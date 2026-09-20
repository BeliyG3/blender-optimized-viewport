/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#ifdef GPU_SHADER
#  pragma once
#endif

#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(vk_frame_generation_preprocess)
LOCAL_GROUP_SIZE(16, 16)
IMAGE(0, SFLOAT_32, read, Float2D, linear_depth_img)
IMAGE(1, SFLOAT_32_32_32_32, read, Float2D, motion_img)
IMAGE(2, SFLOAT_32, write, Float2D, hardware_depth_img)
IMAGE(3, SFLOAT_32_32, write, Float2D, screen_motion_img)
PUSH_CONSTANT(float, near_clip)
PUSH_CONSTANT(float, far_clip)
PUSH_CONSTANT(bool, orthographic)
COMPUTE_SOURCE("vk_frame_generation_preprocess_comp.glsl")
DO_STATIC_COMPILATION()
GPU_SHADER_CREATE_END()
