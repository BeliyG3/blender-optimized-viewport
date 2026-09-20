/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace blender::gpu {

class Texture;

struct FrameGenerationCapabilities {
  bool compiled = false;
  bool supported = false;
  bool active = false;
  int max_generated_frames = 0;
  std::string backend;
  std::string reason;
};

struct FrameGenerationCamera {
  float camera_view_to_clip[16] = {};
  float clip_to_camera_view[16] = {};
  float clip_to_lens_clip[16] = {};
  float clip_to_prev_clip[16] = {};
  float prev_clip_to_clip[16] = {};

  float position[3] = {};
  float up[3] = {};
  float right[3] = {};
  float forward[3] = {};

  float near_clip = 0.1f;
  float far_clip = 1000.0f;
  float field_of_view = 0.785398163f;
  float aspect_ratio = 1.0f;
  bool orthographic = false;
};

struct FrameGenerationEvaluation {
  Texture *color = nullptr;
  Texture *depth = nullptr;
  Texture *motion = nullptr;
  int color_width = 0;
  int color_height = 0;
  int color_subrect_x = 0;
  int color_subrect_y = 0;
  int color_subrect_width = 0;
  int color_subrect_height = 0;
  int render_width = 0;
  int render_height = 0;
  uint64_t frame_id = 0;
  bool reset = false;
  bool color_buffers_hdr = false;
  FrameGenerationCamera camera;
};

enum class FrameGenerationPresentation {
  REAL,
  GENERATED,
  PAIRED_REAL,
  RETAINED_REAL,
};

/**
 * Tracks the two presentations produced by one DLSSG evaluation.
 *
 * A newer evaluation intentionally replaces an older unpresented pair. This keeps at most one
 * generated/real pair queued for an interactive viewport.
 */
class FrameGenerationPresentationQueue {
 public:
  void evaluation_succeeded(const bool present_generated)
  {
    next_ = present_generated ? FrameGenerationPresentation::GENERATED :
                                FrameGenerationPresentation::REAL;
  }

  FrameGenerationPresentation consume(const bool hold_last_real = false)
  {
    const FrameGenerationPresentation result = next_;
    if (next_ == FrameGenerationPresentation::GENERATED) {
      next_ = FrameGenerationPresentation::PAIRED_REAL;
    }
    else if (next_ == FrameGenerationPresentation::PAIRED_REAL) {
      next_ = FrameGenerationPresentation::REAL;
    }
    return result == FrameGenerationPresentation::REAL && hold_last_real ?
               FrameGenerationPresentation::RETAINED_REAL :
               result;
  }

  void reset()
  {
    next_ = FrameGenerationPresentation::REAL;
  }

 private:
  FrameGenerationPresentation next_ = FrameGenerationPresentation::REAL;
};

class FrameGenerationSession {
 public:
  virtual ~FrameGenerationSession() = default;

  /**
   * Queue one 2x Frame Generation evaluation. The generated and retained real textures become
   * usable by later GPU commands in the same render graph.
   */
  virtual bool evaluate(const FrameGenerationEvaluation &evaluation) = 0;
  virtual void reset() = 0;

  virtual Texture *generated_texture_get() const = 0;
  virtual Texture *real_texture_get() const = 0;
  virtual const std::string &error_get() const = 0;
};

FrameGenerationCapabilities frame_generation_capabilities_get();
std::unique_ptr<FrameGenerationSession> frame_generation_session_create();

}  // namespace blender::gpu
