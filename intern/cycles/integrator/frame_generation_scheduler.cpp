/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/frame_generation_scheduler.h"

CCL_NAMESPACE_BEGIN

ViewportFrameGenerationScheduler::Decision ViewportFrameGenerationScheduler::begin_real_frame(
    const uint64_t revision)
{
  if (have_evaluated_revision_ && revision == last_evaluated_revision_) {
    return {};
  }

  Decision decision;
  decision.evaluate = true;
  decision.reset = reset_requested_;
  decision.present_generated = have_previous_real_ && !reset_requested_;
  return decision;
}

void ViewportFrameGenerationScheduler::evaluation_succeeded(const uint64_t revision,
                                                            const Decision &decision)
{
  last_evaluated_revision_ = revision;
  have_evaluated_revision_ = true;
  have_previous_real_ = true;
  reset_requested_ = false;
  generated_pending_ = decision.present_generated;
}

void ViewportFrameGenerationScheduler::evaluation_failed()
{
  generated_pending_ = false;
}

void ViewportFrameGenerationScheduler::request_reset()
{
  have_evaluated_revision_ = false;
  reset_requested_ = true;
  have_previous_real_ = false;
  generated_pending_ = false;
}

bool ViewportFrameGenerationScheduler::consume_generated()
{
  const bool pending = generated_pending_;
  generated_pending_ = false;
  return pending;
}

CCL_NAMESPACE_END
