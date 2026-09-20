/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <cstring>

#include "device/cpu/device.h"
#include "device/device.h"
#include "integrator/path_trace.h"
#include "scene/background.h"
#include "scene/camera.h"
#include "scene/image.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader_graph.h"
#include "session/buffers.h"
#include "session/display_driver.h"
#include "session/output_driver.h"
#include "session/session.h"

#include "util/log.h"
#include "util/math.h"
#include "util/task.h"
#include "util/time.h"

CCL_NAMESPACE_BEGIN

Session::Session(const SessionParams &params_, const SceneParams &scene_params)
    : params(params_),
      eviction_manager_(params_.background),
      render_scheduler_(tile_manager_, params)
{
  TaskScheduler::init(params.threads);

  delayed_reset_.do_reset = false;

  pause_ = false;
  new_work_added_ = false;

  device = Device::create(params.device, stats, profiler, params_.headless);

  if (device->have_error()) {
    progress.set_error(device->error_message());
  }

  scene = make_unique<Scene>(scene_params, device.get());

  if (params.device == params.denoise_device) {
    /* Reuse render device. */
  }
  else {
    denoise_device_ = Device::create(params.denoise_device, stats, profiler, params_.headless);

    if (denoise_device_->have_error()) {
      progress.set_error(denoise_device_->error_message());
    }
  }

  /* Configure path tracer. */
  path_trace_ = make_unique<PathTrace>(device.get(),
                                       denoise_device(),
                                       scene->film,
                                       &scene->dscene,
                                       render_scheduler_,
                                       tile_manager_);
  path_trace_->set_progress(&progress);
  path_trace_->progress_update_cb = [&]() { update_status_time(); };

  tile_manager_.full_buffer_written_cb = [&](string_view filename) {
    if (!full_buffer_written_cb) {
      return;
    }
    full_buffer_written_cb(filename);
  };

  /* Create session thread. */
  session_thread_ = make_unique<thread>([this] { thread_run(); });
}

Session::~Session()
{
  /* Cancel any ongoing render operation. */
  cancel();

  /* Signal session thread to end. */
  {
    const thread_scoped_lock session_thread_lock(session_thread_mutex_);
    session_thread_state_ = SESSION_THREAD_END;
  }
  session_thread_cond_.notify_all();

  /* Destroy session thread. */
  session_thread_->join();
  session_thread_.reset();

  /* Destroy path tracer, before the device. This is needed because destruction might need to
   * access device for device memory free.
   * TODO(sergey): Convert device to be unique_ptr, and rely on C++ to destruct objects in the
   * pre-defined order. */
  path_trace_.reset();

  /* Destroy scene and device. */
  scene.reset();
  denoise_device_.reset();
  device.reset();

  /* Stop task scheduler. */
  TaskScheduler::exit();
}

void Session::start()
{
  {
    /* Signal session thread to start rendering. */
    const thread_scoped_lock session_thread_lock(session_thread_mutex_);
    if (session_thread_state_ == SESSION_THREAD_RENDER) {
      /* Already rendering, nothing to do. */
      return;
    }
    session_thread_state_ = SESSION_THREAD_RENDER;
  }

  session_thread_cond_.notify_all();
}

void Session::cancel(bool quick)
{
  /* Cancel any long running device operations (e.g. shader compilations). */
  device->cancel();

  /* Check if session thread is rendering. */
  const bool rendering = is_session_thread_rendering();

  if (rendering) {
    /* Cancel path trace operations. */
    if (quick && path_trace_) {
      path_trace_->cancel();
    }

    /* Cancel other operations. */
    progress.set_cancel("Exiting");

    /* Signal unpause in case the render was paused. */
    {
      const thread_scoped_lock pause_lock(pause_mutex_);
      pause_ = false;
    }
    pause_cond_.notify_all();

    /* Wait for render thread to be cancelled or finished. */
    wait();
  }
}

bool Session::ready_to_reset()
{
  return path_trace_->ready_to_reset();
}

int Session::draws_after_reset()
{
  return path_trace_->draws_after_reset();
}

void Session::request_denoiser_history_reset()
{
  path_trace_->request_denoiser_history_reset();
}

void Session::run_main_render_loop()
{
  path_trace_->zero_display();

  /* Where the session thread's wall clock goes. The measured stages of a playback frame - scene
   * update, mesh copies, path trace, DLSS - add up to about half of the frame period, and the rest
   * had never been attributed. This says how much of it is this thread waiting rather than working:
   * `update` is the scene update under the scene mutex, `wait` is run_wait_for_work, `render` is
   * the trace and display, `idle` is whatever is left between iterations. */
  static const bool report_loop = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  double loop_update_ms = 0.0, loop_wait_ms = 0.0, loop_render_ms = 0.0;
  int loop_iterations = 0;
  double loop_window_start = time_dt();

  while (true) {
    const double iteration_start = time_dt();
    RenderWork render_work = run_update_for_next_iteration();
    loop_update_ms += (time_dt() - iteration_start) * 1000.0;

    const bool did_cancel = progress.get_cancel();

    if (!render_work) {
      /* Everything scheduled for the current scene state is traced and posted. Published here
       * rather than at `RenderScheduler::done()`, which fires while post-processing work with a
       * display update is still outstanding.
       *
       * The extra `set_update()` is not redundant. The last working iteration already called it,
       * but at that moment this flag was still false - it is raised microseconds later, on this
       * turn of the loop. Without a second notification a draw that happened in between would be
       * the last one, and nobody would ever publish the completion. */
      if (!render_complete_.exchange(true, std::memory_order_acq_rel)) {
        if (pose_started_ != 0.0) {
          static const bool report_pose = getenv("CYCLES_DEBUG_POSE_TIME") != nullptr;
          if (report_pose) {
            /* What one accumulation actually costs, end to end, and how often one gets to finish
             * at all. The second number matters more: if poses complete far less often than they
             * start, then during playback the viewport is showing partly accumulated frames, which
             * is exactly the complaint the timeline hold is meant to fix. The first number is what
             * the hold timeout has to be derived from - too short silently disables the feature on
             * normal frames, too long turns a stalled engine into a multi-second pause. */
            static int completions = 0;
            fprintf(stderr,
                    "POSE_TIME ms=%.2f completions=%d starts=%d\n",
                    (time_dt() - pose_started_) * 1000.0,
                    ++completions,
                    pose_starts_);
            fflush(stderr);
          }
          pose_started_ = 0.0;
        }
        progress.set_update();
      }

      if (LOG_IS_ON(LOG_LEVEL_INFO)) {
        if (did_cancel) {
          LOG_INFO << "Rendering was canceled.";
        }
        else {
          double total_time;
          double render_time;
          progress.get_time(total_time, render_time);
          LOG_INFO << "Rendering in main loop is done in " << render_time << " seconds.";
          LOG_INFO << path_trace_->full_report();
        }
      }

      if (params.background) {
        /* if no work left and in background mode, we can stop immediately. */
        progress.set_status("Finished");
        break;
      }
    }

    if (did_cancel) {
      render_scheduler_.render_work_reschedule_on_cancel(render_work);
      if (!render_work) {
        break;
      }
    }
    else {
      const double wait_start = time_dt();
      const bool waited = run_wait_for_work(render_work);
      loop_wait_ms += (time_dt() - wait_start) * 1000.0;
      if (waited) {
        continue;
      }
    }

    /* Stop rendering if error happened during scene update or other step of preparing scene
     * for render. */
    if (device->have_error()) {
      progress.set_error(device->error_message());
      break;
    }

    const double render_start = time_dt();
    {
      /* buffers mutex is locked entirely while rendering each
       * sample, and released/reacquired on each iteration to allow
       * reset and draw in between */
      const thread_scoped_lock buffers_lock(buffers_mutex_);

      /* update status and timing */
      update_status_time();

      /* render */
      path_trace_->render(render_work);

      /* update status and timing */
      update_status_time();

      /* Stop rendering if error happened during path tracing. */
      if (device->have_error()) {
        progress.set_error(device->error_message());
        break;
      }
    }

    loop_render_ms += (time_dt() - render_start) * 1000.0;

    if (report_loop) {
      loop_iterations++;
      const double window_ms = (time_dt() - loop_window_start) * 1000.0;
      if (window_ms >= 2000.0) {
        /* `idle` is the part of the window this thread was neither updating, waiting on the
         * scheduler, nor rendering - lock contention with the sync on the main thread, and the
         * handshake with the viewport draw. */
        /* `lock` is the part of `update` that was spent waiting for the main thread rather than
         * updating anything - it is included in `update`, not additional to it. */
        fprintf(stderr,
                "LOOP window=%.0f iterations=%d update=%.1f lock=%.1f wait=%.1f render=%.1f "
                "idle=%.1f\n",
                window_ms,
                loop_iterations,
                loop_update_ms,
                loop_lock_wait_ms_,
                loop_wait_ms,
                loop_render_ms,
                window_ms - loop_update_ms - loop_wait_ms - loop_render_ms);
        fflush(stderr);
        loop_update_ms = loop_wait_ms = loop_render_ms = 0.0;
        loop_lock_wait_ms_ = 0.0;
        loop_iterations = 0;
        loop_window_start = time_dt();
      }
    }

    progress.set_update();

    if (did_cancel) {
      break;
    }
  }
}

void Session::thread_run()
{
  while (true) {
    {
      thread_scoped_lock session_thread_lock(session_thread_mutex_);

      if (session_thread_state_ == SESSION_THREAD_WAIT) {
        /* Continue waiting for any signal from the main thread. */
        session_thread_cond_.wait(session_thread_lock);
        continue;
      }
      if (session_thread_state_ == SESSION_THREAD_END) {
        /* End thread immediately. */
        break;
      }
    }

    /* Execute a render. */
    thread_render();

    /* Go back from rendering to waiting. */
    {
      const thread_scoped_lock session_thread_lock(session_thread_mutex_);
      if (session_thread_state_ == SESSION_THREAD_RENDER) {
        session_thread_state_ = SESSION_THREAD_WAIT;
      }
    }
    session_thread_cond_.notify_all();
  }

  /* Flush any remaining operations and destroy display driver here. This ensure
   * graphics API resources are created and destroyed all in the session thread,
   * which can avoid problems contexts and multiple threads. */
  path_trace_->flush_display();
  path_trace_->set_display_driver(nullptr);
}

void Session::thread_render()
{
  if (params.use_profiling && (params.device.type == DEVICE_CPU)) {
    profiler.start();
  }

  /* session thread loop */
  progress.set_status("Waiting for render to start");

  /* run */
  if (!progress.get_cancel()) {
    /* reset number of rendered samples */
    progress.reset_sample();

    run_main_render_loop();
  }

  profiler.stop();

  /* progress update */
  if (progress.get_cancel()) {
    progress.set_status(progress.get_cancel_message());
  }
  else {
    progress.set_update();
  }
}

bool Session::is_session_thread_rendering()
{
  const thread_scoped_lock session_thread_lock(session_thread_mutex_);
  return (session_thread_state_ == SESSION_THREAD_RENDER);
}

RenderWork Session::run_update_for_next_iteration()
{
  RenderWork render_work;

  /* The scene lock is taken inside what `LOOP update` measures, so that number has always been the
   * sum of two unrelated things: updating the scene, and waiting for the main thread to finish its
   * own `sync_data` and let go. Reading it as work led to opposite conclusions about where the
   * freed milliseconds went, and neither could be checked. Separated here. */
  static const bool report_lock_wait = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  const double lock_wait_started = report_lock_wait ? time_dt() : 0.0;

  thread_scoped_timed_lock scene_lock(scene->mutex);

  if (report_lock_wait) {
    loop_lock_wait_ms_ += (time_dt() - lock_wait_started) * 1000.0;
  }

  /* Start of a new accumulation. A pending reset means the previous pose is abandoned - whatever
   * it had accumulated describes a scene state that no longer exists - so the clock restarts here
   * rather than only on the first pose. Iterations that continue an existing pose leave it alone.
   *
   * The first version only started the clock once, and the resulting measurement was the length of
   * the whole run: during playback the flag never gets raised, because a new frame arrives before
   * the scheduler runs out of work. That is a finding in its own right, not a broken timer. */
  {
    const thread_scoped_lock reset_lock(delayed_reset_.mutex);
    if (delayed_reset_.do_reset || pose_started_ == 0.0) {
      pose_started_ = time_dt();
      pose_starts_++;
    }
  }

  /* Perform delayed reset if requested. */
  const bool reset_buffers = delayed_reset_buffer_params();

  /* Update scene.
   *
   * Scene preparation is the one phase outside `PathTrace::render_pipeline`, and on a scene whose
   * geometry is driven by modifiers it dominates the frame - so it is timed here, behind
   * CYCLES_DEBUG_VIEWPORT_PHASES, to complete the per-frame attribution. The reason string tells a
   * real depsgraph update apart from the interactive motion pass re-tagging itself. */
  static const bool phases_enabled = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  const bool report_scene_update = phases_enabled && !params.background;
  const string scene_update_reason = report_scene_update ? scene->update_reason() : string();
  const double scene_update_start_time = time_dt();
  const bool reset_scene = update_scene(delayed_reset_.do_reset);
  {
    const double scene_update_ms = (time_dt() - scene_update_start_time) * 1000.0;
    if (report_scene_update) {
      fprintf(stderr,
              "SCENE_UPDATE ms=%.2f reset=%d reason=%s\n",
              scene_update_ms,
              int(reset_scene),
              scene_update_reason.c_str());
      fflush(stderr);
    }
  }

  /* Update buffers for new parameters. After scene update which influences the passes used. */
  bool have_tiles = true;
  bool switched_to_new_tile = false;

  if (reset_buffers) {
    update_buffers_for_params();

    /* After reset make sure the tile manager is at the first big tile. */
    have_tiles = tile_manager_.next();
    switched_to_new_tile = true;

    eviction_manager_.reset();
  }

  /* Update denoiser settings. */
  {
    const DenoiseParams denoise_params = scene->integrator->get_denoise_params();
    path_trace_->set_denoiser_params(denoise_params);
  }

  /* Update adaptive sampling. */
  {
    const AdaptiveSampling adaptive_sampling = scene->integrator->get_adaptive_sampling();
    path_trace_->set_adaptive_sampling(adaptive_sampling);
  }

  /* Update path guiding. */
  {
    const GuidingParams guiding_params = scene->integrator->get_guiding_params(device.get());
    const bool guiding_reset = (guiding_params.use) ? reset_scene : false;
    path_trace_->set_guiding_params(guiding_params, guiding_reset);
  }

  render_scheduler_.set_sample_params(params.samples,
                                      params.use_sample_subset,
                                      params.sample_subset_offset,
                                      params.sample_subset_length);
  render_scheduler_.set_interactive_sample_budget(
      is_interacting_.load(std::memory_order_relaxed) ? params.interactive_samples : 0);
  render_scheduler_.set_time_limit(params.time_limit);

  while (have_tiles) {
    render_work = render_scheduler_.get_render_work();
    if (render_work) {
      break;
    }

    progress.add_finished_tile(false);

    have_tiles = tile_manager_.next();
    if (have_tiles) {
      render_scheduler_.reset_for_next_tile();
      switched_to_new_tile = true;
    }
  }

  /* A scene change normally resets the sample counter, so a viewport DLSS work with a non-zero
   * iteration index means the scene did not move and the motion vectors must be zeroed. If a scene
   * update does land on a later iteration, keep the real motion vectors for it. */
  if (reset_scene && render_work.dlss.iteration > 0) {
    render_work.dlss.zero_motion = false;
  }

  /* Evict unused image tiles periodically. */
  if (eviction_manager_.need_eviction(!render_work, switched_to_new_tile)) {
    scene->image_manager->evict_unused(device.get(), scene.get());
  }

  if (render_work) {
    const scoped_timer update_timer;

    if (switched_to_new_tile) {
      BufferParams tile_params = buffer_params_;

      const Tile &tile = tile_manager_.get_current_tile();

      tile_params.width = tile.width;
      tile_params.height = tile.height;

      tile_params.window_x = tile.window_x;
      tile_params.window_y = tile.window_y;
      tile_params.window_width = tile.window_width;
      tile_params.window_height = tile.window_height;

      tile_params.full_x = tile.x + buffer_params_.full_x;
      tile_params.full_y = tile.y + buffer_params_.full_y;
      tile_params.full_width = buffer_params_.full_width;
      tile_params.full_height = buffer_params_.full_height;

      tile_params.update_offset_stride();

      path_trace_->reset(buffer_params_, tile_params, reset_buffers);
    }

    /* Update camera if dimensions changed for progressive render. the camera
     * knows nothing about progressive or cropped rendering, it just gets the
     * image dimensions passed in. */
    const float resolution = render_work.resolution_divider;
    const int width = max(1, int(buffer_params_.full_width / resolution));
    const int height = max(1, int(buffer_params_.full_height / resolution));

    scene->update_camera_resolution(progress, width, height);

    /* Unlock scene mutex before loading denoiser kernels, since that may attempt to activate
     * graphics interop, which can deadlock when the scene mutex is still being held. */
    scene_lock.unlock();

    path_trace_->load_kernels();
    path_trace_->alloc_work_memory();

    /* Wait for device to be ready (e.g. finish any background compilations). */
    string device_status;
    while (!device->is_ready(device_status)) {
      progress.set_status(device_status);
      if (progress.get_cancel()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    progress.add_skip_time(update_timer, params.background);
  }

  return render_work;
}

bool Session::run_wait_for_work(const RenderWork &render_work)
{
  /* In an offline rendering there is no pause, and no tiles will mean the job is fully done. */
  if (params.background) {
    return false;
  }

  thread_scoped_lock pause_lock(pause_mutex_);

  if (!pause_ && render_work) {
    /* Rendering is not paused and there is work to be done. No need to wait for anything. */
    return false;
  }

  const bool no_work = !render_work;
  update_status_time(pause_, no_work);

  /* Only leave the loop when rendering is not paused. But even if the current render is
   * un-paused but there is nothing to render keep waiting until new work is added. */
  while (!progress.get_cancel()) {
    const scoped_timer pause_timer;

    if (!pause_ && (render_work || new_work_added_ || delayed_reset_.do_reset)) {
      break;
    }

    const std::chrono::milliseconds wait_time = eviction_manager_.wait_time(!render_work);
    if (wait_time == std::chrono::milliseconds::zero()) {
      /* Break out of the loop for cache eviction. */
      break;
    }

    /* Wait for either pause state changed, extra samples added to render, or idle
     * timer before performing eviction. */
    if (wait_time == std::chrono::milliseconds::max()) {
      pause_cond_.wait(pause_lock);
    }
    else {
      pause_cond_.wait_for(pause_lock, wait_time);
    }

    if (pause_) {
      progress.add_skip_time(pause_timer, params.background);
    }

    update_status_time(pause_, no_work);
    progress.set_update();
  }

  new_work_added_ = false;

  return no_work;
}

bool Session::draw()
{
  return path_trace_->draw();
}

int2 Session::get_effective_tile_size() const
{
  const int image_width = buffer_params_.width;
  const int image_height = buffer_params_.height;

  if (!params.use_auto_tile) {
    return make_int2(image_width, image_height);
  }

  const int64_t image_area = static_cast<int64_t>(image_width) * image_height;

  /* TODO(sergey): Take available memory into account, and if there is enough memory do not
   * tile and prefer optimal performance. */

  const int tile_size = tile_manager_.compute_render_tile_size(params.tile_size);
  const int64_t actual_tile_area = static_cast<int64_t>(tile_size) * tile_size;

  if (actual_tile_area >= image_area && image_width <= TileManager::MAX_TILE_SIZE &&
      image_height <= TileManager::MAX_TILE_SIZE)
  {
    return make_int2(image_width, image_height);
  }

  return make_int2(tile_size, tile_size);
}

bool Session::delayed_reset_buffer_params()
{
  /* Reset buffer parameters, delayed from when we got the reset call so we can complete
   * rendering the sample. Otherwise e.g. viewport navigation might reset without ever
   * finishing anything. */
  const thread_scoped_lock reset_lock(delayed_reset_.mutex);
  if (!delayed_reset_.do_reset) {
    return false;
  }

  const thread_scoped_lock buffers_lock(buffers_mutex_);
  delayed_reset_.do_reset = false;

  params = delayed_reset_.session_params;
  buffer_params_ = delayed_reset_.buffer_params;

  /* Store parameters used for buffers access outside of scene graph. */
  buffer_params_.samples = min(params.samples, Integrator::MAX_SAMPLES);
  buffer_params_.exposure = scene->film->get_exposure();
  buffer_params_.use_approximate_shadow_catcher =
      scene->film->get_use_approximate_shadow_catcher();
  buffer_params_.use_transparent_background = scene->background->get_transparent();

  /* Tile and work scheduling. */
  tile_manager_.reset_scheduling(buffer_params_, get_effective_tile_size());

  return true;
}

void Session::update_buffers_for_params()
{
  render_scheduler_.set_sample_params(params.samples,
                                      params.use_sample_subset,
                                      params.sample_subset_offset,
                                      params.sample_subset_length);
  render_scheduler_.reset(buffer_params_);

  /* Update for new state of scene and passes. */
  buffer_params_.update_passes(scene->passes);
  tile_manager_.update(buffer_params_, scene.get());

  /* Update temp directory on reset.
   * This potentially allows to finish the existing rendering with a previously configure
   * temporary
   * directory in the host software and switch to a new temp directory when new render starts. */
  tile_manager_.set_temp_dir(params.temp_dir);

  /* Progress. */
  progress.reset_sample();
  progress.set_total_pixel_samples(static_cast<uint64_t>(buffer_params_.width) *
                                   buffer_params_.height * buffer_params_.samples);

  if (!params.background) {
    progress.set_start_time();
  }
  const double time_limit = params.time_limit * ((double)tile_manager_.get_num_tiles());
  progress.set_render_start_time();
  progress.set_time_limit(time_limit);
}

void Session::reset(const SessionParams &session_params, const BufferParams &buffer_params)
{
  {
    const thread_scoped_lock reset_lock(delayed_reset_.mutex);
    const thread_scoped_lock pause_lock(pause_mutex_);

    delayed_reset_.do_reset = true;
    delayed_reset_.session_params = session_params;
    delayed_reset_.buffer_params = buffer_params;

    /* A new scene state invalidates the previous completion: whatever was accumulated describes a
     * frame that no longer exists. Cleared here rather than where the reset is applied, so that no
     * draw between the two can read a completion belonging to the old state. */
    render_complete_.store(false, std::memory_order_release);

    scene->scene_updated_while_loading_kernels = true;

    /* This is the main thread, and it is holding the scene lock: `BlenderSession::synchronize`
     * takes it, runs `sync_data`, and calls this before letting go. So whatever `cancel()` waits
     * for, the whole of Blender's UI waits for it too, with the lock held.
     *
     * A review pass found that under DLSS in the viewport `cancel()` cannot actually cancel
     * anything: it only raises the request when the session is a background render or when more
     * than one sample sits in the buffer, and the DLSS accessor is clamped to one. So it degrades
     * into waiting out the render iteration that happens to be in flight. That is a hypothesis
     * about a mechanism, worth two numbers before it is worth a change - how often the wait is
     * non-zero, and how long it runs. A two-humped distribution would confirm it. */
    static const bool report_cancel = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
    const double cancel_started = report_cancel ? time_dt() : 0.0;

    path_trace_->cancel();

    if (report_cancel) {
      const double waited_ms = (time_dt() - cancel_started) * 1000.0;
      cancel_calls_++;
      cancel_total_ms_ += waited_ms;
      if (waited_ms > 0.05) {
        cancel_nonzero_++;
      }
      if (waited_ms > cancel_max_ms_) {
        cancel_max_ms_ = waited_ms;
      }
      /* Coarse histogram in whole milliseconds, capped - enough to see two humps, and no allocation
       * on the main thread while it holds the scene lock. */
      const int bucket = std::min(int(waited_ms), int(std::size(cancel_histogram_)) - 1);
      cancel_histogram_[bucket]++;

      if (cancel_calls_ % 100 == 0) {
        fprintf(stderr,
                "CANCEL_WAIT calls=%d nonzero=%d mean=%.2f max=%.2f ms hist=",
                cancel_calls_,
                cancel_nonzero_,
                cancel_total_ms_ / cancel_calls_,
                cancel_max_ms_);
        for (const int count : cancel_histogram_) {
          fprintf(stderr, "%d,", count);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
      }
    }
  }

  pause_cond_.notify_all();
}

void Session::set_samples(const int samples)
{
  if (samples == params.samples) {
    return;
  }

  params.samples = samples;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);
    new_work_added_ = true;
  }

  pause_cond_.notify_all();
}

void Session::set_time_limit(const double time_limit)
{
  if (time_limit == params.time_limit) {
    return;
  }

  params.time_limit = time_limit;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);
    new_work_added_ = true;
  }

  pause_cond_.notify_all();
}

void Session::set_pause(bool pause)
{
  bool notify = false;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);

    if (pause != pause_) {
      pause_ = pause;
      notify = true;
    }
  }

  if (is_session_thread_rendering()) {
    if (notify) {
      pause_cond_.notify_all();
    }
  }
  else if (pause_) {
    update_status_time(pause_);
  }
}

void Session::set_interaction_state(bool interacting)
{
  eviction_manager_.set_navigating(interacting);

  if (is_interacting_.exchange(interacting) == interacting) {
    return;
  }

  /* The scheduler sleeps once the sample budget is reached, so leaving interaction has to wake it
   * up for accumulation to resume. Entering interaction has to wake it too, otherwise a paused
   * viewport would not drop to the reduced budget until something else nudged it. */
  {
    const thread_scoped_lock pause_lock(pause_mutex_);
    new_work_added_ = true;
  }

  pause_cond_.notify_all();
}

void Session::set_volume_grid_hold(const bool hold)
{
  path_trace_->set_volume_grid_hold(hold);
}

void Session::set_interactive_samples(const int samples)
{
  if (samples == params.interactive_samples) {
    return;
  }

  params.interactive_samples = samples;

  {
    const thread_scoped_lock pause_lock(pause_mutex_);
    new_work_added_ = true;
  }

  pause_cond_.notify_all();
}

void Session::set_output_driver(unique_ptr<OutputDriver> driver)
{
  path_trace_->set_output_driver(std::move(driver));
}

void Session::set_denoiser_external_images(const DenoiserExternalImages &images)
{
  path_trace_->set_denoiser_external_images(images);
}

void Session::set_display_driver(unique_ptr<DisplayDriver> driver)
{
  path_trace_->set_display_driver(std::move(driver));
}

double Session::get_estimated_remaining_time() const
{
  const double completed = progress.get_progress();
  if (completed == 0.0) {
    return 0.0;
  }

  double total_time;
  double render_time;
  progress.get_time(total_time, render_time);
  double remaining = (1.0 - (double)completed) * (render_time / (double)completed);

  const double time_limit = render_scheduler_.get_time_limit() *
                            ((double)tile_manager_.get_num_tiles());
  if (time_limit != 0.0) {
    remaining = min(remaining, max(time_limit - render_time, 0.0));
  }

  return remaining;
}

void Session::wait()
{
  /* Wait until session thread either is waiting or ending. */
  while (true) {
    thread_scoped_lock session_thread_lock(session_thread_mutex_);
    if (session_thread_state_ != SESSION_THREAD_RENDER) {
      break;
    }
    session_thread_cond_.wait(session_thread_lock);
  }
}

bool Session::update_scene(const bool reset_samples)
{
  /* Update number of samples in the integrator.
   * Ideally this would need to happen once in `Session::set_samples()`, but the issue there is
   * the initial configuration when Session is created where the `set_samples()` is not used.
   *
   * NOTE: Unless reset was requested only allow increasing number of samples. */
  if (reset_samples || scene->integrator->get_aa_samples() < params.samples) {
    scene->integrator->set_aa_samples(params.samples);
  }

  scene->integrator->set_use_sample_subset(params.use_sample_subset);
  scene->integrator->set_sample_subset_offset(params.sample_subset_offset);
  scene->integrator->set_sample_subset_length(params.sample_subset_length);

  /* When multiple tiles are used SAMPLE_COUNT pass is used to keep track of possible partial
   * tile results. */
  scene->film->set_use_sample_count(tile_manager_.has_multiple_tiles());

  const bool reset = scene->need_reset(false);

  if (scene->update(progress)) {
    profiler.reset(scene->shaders.size(), scene->objects.size());
  }

  return reset;
}

static string status_append(const string &status, const string &suffix)
{
  string prefix = status;
  if (!prefix.empty()) {
    prefix += ", ";
  }
  return prefix + suffix;
}

void Session::update_status_time(bool show_pause, bool show_done)
{
  string status;
  string substatus;

  const int current_tile = progress.get_rendered_tiles();
  const int num_tiles = tile_manager_.get_num_tiles();

  const int current_sample = progress.get_current_sample();
  const int num_samples = render_scheduler_.get_num_samples();

  /* TIle. */
  if (tile_manager_.has_multiple_tiles()) {
    substatus = status_append(substatus,
                              string_printf("Rendered %d/%d Tiles", current_tile, num_tiles));
  }

  /* Sample. */
  if (!params.background && num_samples == Integrator::MAX_SAMPLES) {
    substatus = status_append(substatus, string_printf("Sample %d", current_sample));
  }
  else {
    substatus = status_append(substatus,
                              string_printf("Sample %d/%d", current_sample, num_samples));
  }

  /* Append any device-specific status (such as background kernel optimization) */
  string device_status;
  if (device->is_ready(device_status) && !device_status.empty()) {
    substatus += string_printf(" (%s)", device_status.c_str());
  }

  /* TODO(sergey): Denoising status from the path trace. */

  if (show_pause) {
    status = "Rendering Paused";
  }
  else if (show_done) {
    status = "Rendering Done";
    progress.set_end_time(); /* Save end time so that further calls to get_time are accurate. */
  }
  else {
    status = substatus;
    substatus.clear();
  }

  progress.set_status(status, substatus);
}

void Session::device_free()
{
  scene->device_free();
  path_trace_->device_free();
}

void Session::collect_statistics(RenderStats *render_stats)
{
  scene->collect_statistics(render_stats);
  if (params.use_profiling && (params.device.type == DEVICE_CPU)) {
    render_stats->collect_profiling(scene.get(), profiler);
  }
}

bool Session::dlss_runtime_failed() const
{
  return path_trace_->dlss_runtime_failed();
}

/* --------------------------------------------------------------------
 * Full-frame on-disk
 * storage.
 */

void Session::process_full_buffer_from_disk(string_view filename)
{
  path_trace_->process_full_buffer_from_disk(filename);
}

CCL_NAMESPACE_END
