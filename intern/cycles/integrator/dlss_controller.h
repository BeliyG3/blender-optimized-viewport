/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/types.h"

CCL_NAMESPACE_BEGIN

struct DLSSIterationPlan {
  int index = 0;
  float2 jitter = {};
  bool reset_history = false;
  bool zero_motion = false;
};

/* Pure, device-independent policy for bounded offline DLSS evaluations. */
class DLSSRenderController {
 public:
  static DLSSIterationPlan plan_iteration(int index,
                                          bool reset_history,
                                          bool zero_motion_first);

 private:
  static float radical_inverse(int index, int base);
};

CCL_NAMESPACE_END
