/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/dlss_controller.h"
#include "integrator/pass_resampler.h"

#include <cmath>
#include <limits>

#include "testing/testing.h"

CCL_NAMESPACE_BEGIN

TEST(dlss_controller, iteration_policy)
{
  const DLSSIterationPlan first = DLSSRenderController::plan_iteration(0, true, false);
  EXPECT_TRUE(first.reset_history);
  EXPECT_FALSE(first.zero_motion);
  EXPECT_FLOAT_EQ(first.jitter.x, 0.0f);
  EXPECT_FLOAT_EQ(first.jitter.y, -1.0f / 6.0f);

  const DLSSIterationPlan refinement = DLSSRenderController::plan_iteration(1, true, false);
  EXPECT_FALSE(refinement.reset_history);
  EXPECT_TRUE(refinement.zero_motion);
  EXPECT_NE(first.jitter.x, refinement.jitter.x);
  EXPECT_NE(first.jitter.y, refinement.jitter.y);

  const DLSSIterationPlan warmup = DLSSRenderController::plan_iteration(0, true, true);
  EXPECT_TRUE(warmup.reset_history);
  EXPECT_TRUE(warmup.zero_motion);
}

TEST(dlss_controller, halton_sequence_is_unique)
{
  for (int first_index = 0; first_index < 16; ++first_index) {
    const float2 first = DLSSRenderController::plan_iteration(first_index, false, false).jitter;
    for (int second_index = first_index + 1; second_index < 16; ++second_index) {
      const float2 second = DLSSRenderController::plan_iteration(second_index, false, false).jitter;
      EXPECT_FALSE(first.x == second.x && first.y == second.y);
    }
  }
}

TEST(dlss_pass_resampler, indices_use_nearest)
{
  const float source[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float destination[16] = {};
  PassResampler::resample(PASS_OBJECT_ID, source, 2, 2, destination, 4, 4, 1);
  EXPECT_FLOAT_EQ(destination[0], 1.0f);
  EXPECT_FLOAT_EQ(destination[3], 2.0f);
  EXPECT_FLOAT_EQ(destination[12], 3.0f);
  EXPECT_FLOAT_EQ(destination[15], 4.0f);
}

TEST(dlss_pass_resampler, normals_are_renormalized)
{
  const float source[] = {2.0f, 0.0f, 0.0f};
  float destination[12] = {};
  PassResampler::resample(PASS_NORMAL, source, 1, 1, destination, 2, 2, 3);
  for (int pixel = 0; pixel < 4; ++pixel) {
    EXPECT_FLOAT_EQ(destination[pixel * 3], 1.0f);
    EXPECT_FLOAT_EQ(destination[pixel * 3 + 1], 0.0f);
    EXPECT_FLOAT_EQ(destination[pixel * 3 + 2], 0.0f);
  }
}

TEST(dlss_pass_resampler, vectors_convert_to_output_pixels)
{
  const float source[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float destination[16] = {};
  PassResampler::resample(PASS_MOTION, source, 1, 1, destination, 2, 2, 4);
  EXPECT_FLOAT_EQ(destination[0], 2.0f);
  EXPECT_FLOAT_EQ(destination[1], 4.0f);
  EXPECT_FLOAT_EQ(destination[2], 6.0f);
  EXPECT_FLOAT_EQ(destination[3], 8.0f);
}

TEST(dlss_pass_resampler, depth_does_not_blend_infinity)
{
  const float infinity = std::numeric_limits<float>::infinity();
  const float source[] = {2.0f, infinity, infinity, infinity};
  float destination[9] = {};
  PassResampler::resample(PASS_DEPTH, source, 2, 2, destination, 3, 3, 1);
  EXPECT_FLOAT_EQ(destination[4], 2.0f);
  EXPECT_TRUE(std::isinf(destination[8]));
}

CCL_NAMESPACE_END
