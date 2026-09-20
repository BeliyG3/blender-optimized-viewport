/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * Ray Reconstruction over the Vulkan path of NGX.
 *
 * Cycles drives its own copy of this feature through the CUDA path, and there the model of DLSS
 * 4.5 - preset F - refuses every evaluation with a platform error, on inputs identical to the ones
 * the previous model accepts. This is the same feature reached the way NVIDIA's own customers
 * reach it.
 */

#pragma once

#include <memory>
#include <string>

#include "GPU_dlss_ray_reconstruction.hh"

namespace blender::gpu {

/** A denoiser on the Vulkan backend, or null with the reason in `r_error`. */
std::unique_ptr<DlssRayReconstructionSession> vk_dlss_ray_reconstruction_session_create();

/**
 * Run one evaluation on textures made for the purpose, and report what NGX made of it.
 *
 * This answers the only question worth answering before a denoiser is moved anywhere: whether the
 * model runs at all on this path. It allocates its own inputs, so it needs nothing from the
 * renderer and can be called from any place that has a Vulkan context.
 */
std::string vk_dlss_ray_reconstruction_probe(int preset);

}  // namespace blender::gpu
