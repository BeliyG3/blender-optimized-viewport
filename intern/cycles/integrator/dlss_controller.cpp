/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/dlss_controller.h"

#include "util/math.h"

CCL_NAMESPACE_BEGIN

float DLSSRenderController::radical_inverse(int index, const int base)
{
  float inverse_base = 1.0f / float(base);
  float inverse = inverse_base;
  float result = 0.0f;
  while (index > 0) {
    result += float(index % base) * inverse;
    index /= base;
    inverse *= inverse_base;
  }
  return result;
}

DLSSIterationPlan DLSSRenderController::plan_iteration(const int index,
                                                       const bool reset_history,
                                                       const bool zero_motion_first)
{
  DLSSIterationPlan plan;
  plan.index = max(index, 0);

  /* Start at Halton element one: element zero is the corner (-0.5, -0.5), which is a poor
   * reconstruction input and differs from the viewport sequence. */
  const int halton_index = plan.index + 1;
  plan.jitter = make_float2(radical_inverse(halton_index, 2) - 0.5f,
                            radical_inverse(halton_index, 3) - 0.5f);
  plan.reset_history = reset_history && plan.index == 0;
  plan.zero_motion = zero_motion_first || plan.index > 0;
  return plan;
}

CCL_NAMESPACE_END
