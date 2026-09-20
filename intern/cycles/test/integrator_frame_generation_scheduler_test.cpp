/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/frame_generation_scheduler.h"

#include "testing/testing.h"

CCL_NAMESPACE_BEGIN

TEST(frame_generation_scheduler, static_sampling_is_not_interpolated)
{
  ViewportFrameGenerationScheduler scheduler;
  const auto first = scheduler.begin_real_frame(1);
  EXPECT_TRUE(first.evaluate);
  EXPECT_TRUE(first.reset);
  EXPECT_FALSE(first.present_generated);
  scheduler.evaluation_succeeded(1, first);

  const auto static_sample = scheduler.begin_real_frame(1);
  EXPECT_FALSE(static_sample.evaluate);
  EXPECT_FALSE(scheduler.consume_generated());
}

TEST(frame_generation_scheduler, initial_zero_revision_is_evaluated)
{
  ViewportFrameGenerationScheduler scheduler;
  const auto first = scheduler.begin_real_frame(0);

  EXPECT_TRUE(first.evaluate);
  EXPECT_TRUE(first.reset);
  scheduler.evaluation_succeeded(0, first);
  EXPECT_TRUE(scheduler.has_evaluated_revision());
  EXPECT_FALSE(scheduler.begin_real_frame(0).evaluate);
}

TEST(frame_generation_scheduler, sequential_real_frames_generate_one_intermediate)
{
  ViewportFrameGenerationScheduler scheduler;
  const auto first = scheduler.begin_real_frame(10);
  scheduler.evaluation_succeeded(10, first);

  const auto second = scheduler.begin_real_frame(11);
  EXPECT_TRUE(second.evaluate);
  EXPECT_FALSE(second.reset);
  EXPECT_TRUE(second.present_generated);
  scheduler.evaluation_succeeded(11, second);
  EXPECT_TRUE(scheduler.consume_generated());
  EXPECT_FALSE(scheduler.consume_generated());
}

TEST(frame_generation_scheduler, reset_discards_pending_and_warms_with_real_frame)
{
  ViewportFrameGenerationScheduler scheduler;
  auto decision = scheduler.begin_real_frame(1);
  scheduler.evaluation_succeeded(1, decision);
  decision = scheduler.begin_real_frame(2);
  scheduler.evaluation_succeeded(2, decision);

  scheduler.request_reset();
  EXPECT_FALSE(scheduler.consume_generated());
  decision = scheduler.begin_real_frame(20);
  EXPECT_TRUE(decision.reset);
  EXPECT_FALSE(decision.present_generated);
}

TEST(frame_generation_scheduler, reset_reevaluates_the_same_revision)
{
  ViewportFrameGenerationScheduler scheduler;
  auto decision = scheduler.begin_real_frame(7);
  scheduler.evaluation_succeeded(7, decision);

  scheduler.request_reset();
  decision = scheduler.begin_real_frame(7);
  EXPECT_TRUE(decision.evaluate);
  EXPECT_TRUE(decision.reset);
  EXPECT_FALSE(decision.present_generated);
}

TEST(frame_generation_scheduler, newer_pair_replaces_unpresented_pair)
{
  ViewportFrameGenerationScheduler scheduler;
  auto decision = scheduler.begin_real_frame(1);
  scheduler.evaluation_succeeded(1, decision);
  decision = scheduler.begin_real_frame(2);
  scheduler.evaluation_succeeded(2, decision);
  decision = scheduler.begin_real_frame(3);
  scheduler.evaluation_succeeded(3, decision);

  EXPECT_TRUE(scheduler.consume_generated());
  EXPECT_FALSE(scheduler.consume_generated());
  EXPECT_EQ(scheduler.last_evaluated_revision(), 3);
}

TEST(frame_generation_scheduler, static_sampling_does_not_discard_pending_generated_frame)
{
  ViewportFrameGenerationScheduler scheduler;
  auto decision = scheduler.begin_real_frame(1);
  scheduler.evaluation_succeeded(1, decision);
  decision = scheduler.begin_real_frame(2);
  scheduler.evaluation_succeeded(2, decision);

  const auto static_sample = scheduler.begin_real_frame(2);
  EXPECT_FALSE(static_sample.evaluate);
  EXPECT_TRUE(scheduler.consume_generated());
}

CCL_NAMESPACE_END
