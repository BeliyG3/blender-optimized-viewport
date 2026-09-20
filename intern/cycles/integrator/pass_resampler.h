/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/types.h"

CCL_NAMESPACE_BEGIN

class PassResampler {
 public:
  static bool can_resample(PassType type);
  static void resample(PassType type,
                       const float *source,
                       int source_width,
                       int source_height,
                       float *destination,
                       int destination_width,
                       int destination_height,
                       int num_channels);
};

CCL_NAMESPACE_END
