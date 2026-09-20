/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <gtest/gtest.h>

#include "bvh/params.h"
#include "integrator/render_scheduler.h"
#include "scene/integrator.h"
#include "session/session.h"
#include "session/tile.h"

CCL_NAMESPACE_BEGIN

/* Mirrors the instance motion decision in `BVHEmbree::add_instance()`. Object motion arrays are
 * also populated for the interactive motion pass, which the kernel does not trace - instancing
 * them then would place the geometry between two transforms. */
static size_t bvh_embree_instance_motion_steps(const BVHParams &params, const size_t motion_size)
{
  const bool use_object_motion = params.use_object_motion && motion_size > 1;
  return use_object_motion ? motion_size : 1;
}

TEST(BVHEmbreeInstance, motion_steps_ignored_without_object_motion)
{
  BVHParams params;
  EXPECT_FALSE(params.use_object_motion);
  /* The interactive motion pass leaves two keys on a moving object. */
  EXPECT_EQ(bvh_embree_instance_motion_steps(params, 2), 1u);
  EXPECT_EQ(bvh_embree_instance_motion_steps(params, 3), 1u);
}

TEST(BVHEmbreeInstance, motion_steps_used_with_object_motion)
{
  BVHParams params;
  params.use_object_motion = true;
  EXPECT_EQ(bvh_embree_instance_motion_steps(params, 3), 3u);
  /* A single key still means no motion. */
  EXPECT_EQ(bvh_embree_instance_motion_steps(params, 1), 1u);
  EXPECT_EQ(bvh_embree_instance_motion_steps(params, 0), 1u);
}

static DenoiseParams make_dlss_params(const bool offline = false, const int iterations = 1)
{
  DenoiseParams params;
  params.use = true;
  params.type = DENOISER_DLSS;
  params.upscale_factor = 2.0f;
  params.dlss_offline = offline;
  params.dlss_iterations = iterations;
  return params;
}

static SessionParams make_viewport_session_params()
{
  SessionParams params;
  params.background = false;
  params.headless = false;
  params.pixel_size = 1;
  params.use_resolution_divider = false;
  return params;
}

TEST(IntegratorRenderScheduler, calculate_resolution_divider_for_resolution)
{
  EXPECT_EQ(calculate_resolution_divider_for_resolution(1920, 1080, 1920), 1);
  EXPECT_EQ(calculate_resolution_divider_for_resolution(1920, 1080, 960), 2);
  EXPECT_EQ(calculate_resolution_divider_for_resolution(1920, 1080, 480), 4);
}

TEST(IntegratorRenderScheduler, calculate_resolution_for_divider)
{
  EXPECT_EQ(calculate_resolution_for_divider(1920, 1080, 1), 1440);
  EXPECT_EQ(calculate_resolution_for_divider(1920, 1080, 2), 720);
  EXPECT_EQ(calculate_resolution_for_divider(1920, 1080, 4), 360);
}

TEST(IntegratorRenderScheduler, viewport_dlss_honors_sample_limit)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_denoiser_params(make_dlss_params());
  scheduler.set_sample_params(3, false, 0, 0);
  scheduler.reset(BufferParams());

  for (int iteration = 0; iteration < 3; ++iteration) {
    const RenderWork work = scheduler.get_render_work();
    EXPECT_EQ(work.path_trace.start_sample, iteration);
    EXPECT_EQ(work.path_trace.num_samples, 1);
    EXPECT_TRUE(work.init_render_buffers);
    EXPECT_TRUE(work.tile.denoise);
    EXPECT_EQ(scheduler.get_num_rendered_samples(), iteration + 1);
  }

  EXPECT_FALSE(scheduler.get_render_work());
}

TEST(IntegratorRenderScheduler, viewport_dlss_sample_limit_can_increase)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_denoiser_params(make_dlss_params());
  scheduler.set_sample_params(1, false, 0, 0);
  scheduler.reset(BufferParams());

  EXPECT_TRUE(scheduler.get_render_work());
  EXPECT_FALSE(scheduler.get_render_work());

  scheduler.set_sample_params(2, false, 0, 0);
  const RenderWork resumed_work = scheduler.get_render_work();
  EXPECT_EQ(resumed_work.path_trace.start_sample, 1);
  EXPECT_EQ(resumed_work.path_trace.num_samples, 1);
  EXPECT_TRUE(resumed_work.init_render_buffers);
  EXPECT_FALSE(scheduler.get_render_work());
}

TEST(IntegratorRenderScheduler, viewport_dlss_sample_limit_can_decrease)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_denoiser_params(make_dlss_params());
  scheduler.set_sample_params(3, false, 0, 0);
  scheduler.reset(BufferParams());

  EXPECT_EQ(scheduler.get_render_work().path_trace.num_samples, 1);
  EXPECT_EQ(scheduler.get_render_work().path_trace.num_samples, 1);

  scheduler.set_sample_params(1, false, 0, 0);
  const RenderWork final_work = scheduler.get_render_work();
  EXPECT_EQ(final_work.path_trace.num_samples, 0);
  EXPECT_FALSE(scheduler.get_render_work());
}

TEST(IntegratorRenderScheduler, viewport_dlss_accepts_unlimited_sample_limit)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_denoiser_params(make_dlss_params());
  scheduler.set_sample_params(Integrator::MAX_SAMPLES, false, 0, 0);
  scheduler.reset(BufferParams());

  EXPECT_EQ(scheduler.get_num_samples(), Integrator::MAX_SAMPLES);
  for (int iteration = 0; iteration < 4; ++iteration) {
    const RenderWork work = scheduler.get_render_work();
    EXPECT_EQ(work.path_trace.start_sample, iteration);
    EXPECT_EQ(work.path_trace.num_samples, 1);
    EXPECT_TRUE(work.init_render_buffers);
  }
}

TEST(IntegratorRenderScheduler, offline_dlss_uses_iteration_budget)
{
  TileManager tile_manager;
  SessionParams session_params = make_viewport_session_params();
  session_params.background = true;
  RenderScheduler scheduler(tile_manager, session_params);
  scheduler.set_denoiser_params(make_dlss_params(true, 4));
  scheduler.set_sample_params(64, false, 0, 0);
  scheduler.reset(BufferParams());

  EXPECT_EQ(scheduler.get_num_samples(), 4);
  for (int iteration = 0; iteration < 4; ++iteration) {
    const RenderWork work = scheduler.get_render_work();
    EXPECT_EQ(work.path_trace.start_sample, iteration);
    EXPECT_EQ(work.path_trace.num_samples, 1);
    EXPECT_EQ(work.dlss.iteration, iteration);
    EXPECT_TRUE(work.dlss.jitter_from_iteration);
    EXPECT_TRUE(work.init_render_buffers);
  }
  EXPECT_FALSE(scheduler.get_render_work());
}

TEST(IntegratorRenderScheduler, viewport_dlss_zeroes_motion_after_first_iteration)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_denoiser_params(make_dlss_params());
  scheduler.set_sample_params(3, false, 0, 0);
  scheduler.reset(BufferParams());

  for (int iteration = 0; iteration < 3; ++iteration) {
    const RenderWork work = scheduler.get_render_work();
    EXPECT_EQ(work.dlss.iteration, iteration);
    /* Iteration 0 carries the real motion vectors; the rest add samples of an unchanged scene and
     * must not drag the temporal history along again. */
    EXPECT_EQ(work.dlss.zero_motion, iteration > 0);
    /* History reset stays driven by the session, and the jitter by update_camera_resolution(). */
    EXPECT_FALSE(work.dlss.reset_history);
    EXPECT_FALSE(work.dlss.jitter_from_iteration);
  }
}

TEST(IntegratorRenderScheduler, viewport_dlss_interactive_budget_caps_samples)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_denoiser_params(make_dlss_params());
  scheduler.set_sample_params(32, false, 0, 0);
  scheduler.set_interactive_sample_budget(4);
  scheduler.reset(BufferParams());

  EXPECT_EQ(scheduler.get_num_samples(), 4);
  for (int iteration = 0; iteration < 4; ++iteration) {
    EXPECT_EQ(scheduler.get_render_work().path_trace.num_samples, 1);
  }
  EXPECT_FALSE(scheduler.get_render_work());
}

TEST(IntegratorRenderScheduler, viewport_dlss_interactive_budget_resumes_without_reset)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_denoiser_params(make_dlss_params());
  scheduler.set_sample_params(8, false, 0, 0);
  scheduler.set_interactive_sample_budget(2);
  scheduler.reset(BufferParams());

  EXPECT_TRUE(scheduler.get_render_work());
  EXPECT_TRUE(scheduler.get_render_work());
  EXPECT_FALSE(scheduler.get_render_work());

  /* Interaction ended: accumulation continues from where it stopped, with no reset in between, so
   * the DLSS temporal history survives. */
  scheduler.set_interactive_sample_budget(0);
  const RenderWork resumed_work = scheduler.get_render_work();
  EXPECT_EQ(resumed_work.path_trace.start_sample, 2);
  EXPECT_EQ(resumed_work.path_trace.num_samples, 1);
  EXPECT_EQ(scheduler.get_num_samples(), 8);
}

TEST(IntegratorRenderScheduler, viewport_dlss_interactive_budget_clamped_by_max_samples)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_denoiser_params(make_dlss_params());
  scheduler.set_sample_params(2, false, 0, 0);
  scheduler.set_interactive_sample_budget(8);
  scheduler.reset(BufferParams());

  EXPECT_EQ(scheduler.get_num_samples(), 2);
}

TEST(IntegratorRenderScheduler, interactive_budget_ignored_without_dlss)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_sample_params(32, false, 0, 0);
  scheduler.set_interactive_sample_budget(4);
  scheduler.reset(BufferParams());

  EXPECT_EQ(scheduler.get_num_samples(), 32);
}

TEST(IntegratorRenderScheduler, interactive_budget_ignored_in_background)
{
  TileManager tile_manager;
  SessionParams session_params = make_viewport_session_params();
  session_params.background = true;
  RenderScheduler scheduler(tile_manager, session_params);
  scheduler.set_denoiser_params(make_dlss_params());
  scheduler.set_sample_params(32, false, 0, 0);
  scheduler.set_interactive_sample_budget(4);
  scheduler.reset(BufferParams());

  EXPECT_EQ(scheduler.get_num_samples(), 32);
}

TEST(IntegratorRenderScheduler, non_dlss_sampling_is_unchanged)
{
  TileManager tile_manager;
  RenderScheduler scheduler(tile_manager, make_viewport_session_params());
  scheduler.set_sample_params(3, false, 0, 0);
  scheduler.reset(BufferParams());

  const RenderWork first_work = scheduler.get_render_work();
  EXPECT_EQ(first_work.path_trace.start_sample, 0);
  EXPECT_EQ(first_work.path_trace.num_samples, 1);
  EXPECT_TRUE(first_work.init_render_buffers);

  const RenderWork second_work = scheduler.get_render_work();
  EXPECT_EQ(second_work.path_trace.start_sample, 1);
  EXPECT_FALSE(second_work.init_render_buffers);
}

CCL_NAMESPACE_END
