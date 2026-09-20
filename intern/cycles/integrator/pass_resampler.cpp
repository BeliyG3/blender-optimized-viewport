/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/pass_resampler.h"

#include <cmath>
#include <limits>

#include "util/math.h"

CCL_NAMESPACE_BEGIN

static bool is_index_pass(const PassType type)
{
  return type == PASS_OBJECT_ID || type == PASS_MATERIAL_ID;
}

static bool is_normal_pass(const PassType type)
{
  return type == PASS_NORMAL || type == PASS_DENOISING_NORMAL;
}

static bool is_vector_pass(const PassType type)
{
  return type == PASS_MOTION || type == PASS_DENOISING_BACKWARD_MOTION ||
         type == PASS_DENOISING_SPECULAR_MOTION;
}

static bool is_depth_pass(const PassType type)
{
  return type == PASS_DEPTH || type == PASS_DENOISING_DEPTH;
}

bool PassResampler::can_resample(const PassType type)
{
  return type != PASS_CRYPTOMATTE && type != PASS_NONE;
}

void PassResampler::resample(const PassType type,
                             const float *source,
                             const int source_width,
                             const int source_height,
                             float *destination,
                             const int destination_width,
                             const int destination_height,
                             const int num_channels)
{
  const float scale_x = float(destination_width) / float(source_width);
  const float scale_y = float(destination_height) / float(source_height);

  for (int y = 0; y < destination_height; ++y) {
    const float source_y = (float(y) + 0.5f) / scale_y - 0.5f;
    const int y0 = clamp(int(floorf(source_y)), 0, source_height - 1);
    const int y1 = min(y0 + 1, source_height - 1);
    const float ty = clamp(source_y - floorf(source_y), 0.0f, 1.0f);

    for (int x = 0; x < destination_width; ++x) {
      const float source_x = (float(x) + 0.5f) / scale_x - 0.5f;
      const int x0 = clamp(int(floorf(source_x)), 0, source_width - 1);
      const int x1 = min(x0 + 1, source_width - 1);
      const float tx = clamp(source_x - floorf(source_x), 0.0f, 1.0f);
      float *output = destination + (x + y * destination_width) * num_channels;

      if (is_index_pass(type)) {
        const int nearest_x = clamp(int(floorf(source_x + 0.5f)), 0, source_width - 1);
        const int nearest_y = clamp(int(floorf(source_y + 0.5f)), 0, source_height - 1);
        const float *input = source +
                             (nearest_x + nearest_y * source_width) * num_channels;
        for (int channel = 0; channel < num_channels; ++channel) {
          output[channel] = input[channel];
        }
        continue;
      }

      const float weights[4] = {
          (1.0f - tx) * (1.0f - ty), tx * (1.0f - ty), (1.0f - tx) * ty, tx * ty};
      const int indices[4] = {
          x0 + y0 * source_width,
          x1 + y0 * source_width,
          x0 + y1 * source_width,
          x1 + y1 * source_width,
      };

      for (int channel = 0; channel < num_channels; ++channel) {
        float value = 0.0f;
        float total_weight = 0.0f;
        for (int sample = 0; sample < 4; ++sample) {
          const float input = source[indices[sample] * num_channels + channel];
          if (is_depth_pass(type) && !std::isfinite(input)) {
            continue;
          }
          value += input * weights[sample];
          total_weight += weights[sample];
        }
        output[channel] = total_weight > 0.0f ?
                              value / total_weight :
                              std::numeric_limits<float>::infinity();
      }

      if (is_normal_pass(type) && num_channels >= 3) {
        const float length = sqrtf(output[0] * output[0] + output[1] * output[1] +
                                   output[2] * output[2]);
        if (length > 1e-20f) {
          output[0] /= length;
          output[1] /= length;
          output[2] /= length;
        }
      }
      else if (is_vector_pass(type) && num_channels >= 2) {
        output[0] *= scale_x;
        output[1] *= scale_y;
        if (num_channels >= 4) {
          output[2] *= scale_x;
          output[3] *= scale_y;
        }
      }
    }
  }
}

CCL_NAMESPACE_END
