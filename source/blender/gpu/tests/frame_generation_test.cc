/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GPU_frame_generation.hh"

#include "testing/testing.h"

namespace blender::gpu::tests {

TEST(frame_generation_presentation, generated_is_followed_by_paired_real)
{
  FrameGenerationPresentationQueue queue;
  queue.evaluation_succeeded(true);

  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::GENERATED);
  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::PAIRED_REAL);
  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::REAL);
}

TEST(frame_generation_presentation, warmup_evaluation_presents_real)
{
  FrameGenerationPresentationQueue queue;
  queue.evaluation_succeeded(false);

  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::REAL);
  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::REAL);
}

TEST(frame_generation_presentation, newer_pair_replaces_stale_pair)
{
  FrameGenerationPresentationQueue queue;
  queue.evaluation_succeeded(true);
  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::GENERATED);

  queue.evaluation_succeeded(true);
  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::GENERATED);
  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::PAIRED_REAL);
}

TEST(frame_generation_presentation, reset_discards_pending_pair)
{
  FrameGenerationPresentationQueue queue;
  queue.evaluation_succeeded(true);
  queue.reset();

  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::REAL);
}

TEST(frame_generation_presentation, retains_last_real_while_next_frame_is_prepared)
{
  FrameGenerationPresentationQueue queue;

  EXPECT_EQ(queue.consume(true), FrameGenerationPresentation::RETAINED_REAL);
  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::REAL);
}

TEST(frame_generation_presentation, viewports_have_independent_queues)
{
  FrameGenerationPresentationQueue first_viewport;
  FrameGenerationPresentationQueue second_viewport;
  first_viewport.evaluation_succeeded(true);

  EXPECT_EQ(first_viewport.consume(), FrameGenerationPresentation::GENERATED);
  EXPECT_EQ(second_viewport.consume(), FrameGenerationPresentation::REAL);
}

TEST(frame_generation_presentation, offscreen_reset_discards_interactive_pair)
{
  FrameGenerationPresentationQueue queue;
  queue.evaluation_succeeded(true);
  queue.reset();

  EXPECT_EQ(queue.consume(), FrameGenerationPresentation::REAL);
}

TEST(frame_generation_evaluation, defaults_to_display_referred_sdr)
{
  FrameGenerationEvaluation evaluation;
  EXPECT_FALSE(evaluation.color_buffers_hdr);
}

}  // namespace blender::gpu::tests
