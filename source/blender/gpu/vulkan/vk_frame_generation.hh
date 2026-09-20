/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <memory>

#include "GPU_frame_generation.hh"

namespace blender::gpu {

FrameGenerationCapabilities vk_frame_generation_capabilities_get();
std::unique_ptr<FrameGenerationSession> vk_frame_generation_session_create();
void vk_frame_generation_shutdown();

}  // namespace blender::gpu
