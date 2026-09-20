/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "integrator/path_trace.h"

#include "device/cpu/device.h"
#include "device/device.h"

#include "integrator/dlss_controller.h"
#ifdef WITH_DLSS
#  include "integrator/denoiser_dlss.h"
#endif
#include "integrator/pass_accessor.h"
#include "integrator/path_trace_display.h"
#include "integrator/path_trace_tile.h"
#include "integrator/render_scheduler.h"

#include "scene/pass.h"
#include "scene/scene.h"

#include "session/display_driver.h"
#include "session/tile.h"

#include "util/log.h"
#include "util/progress.h"
#include "util/scoped_defer.h"
#include "util/tbb.h"
#include "util/time.h"

CCL_NAMESPACE_BEGIN

PathTrace::PathTrace(Device *device,
                     Device *denoise_device,
                     Film *film,
                     DeviceScene *device_scene,
                     RenderScheduler &render_scheduler,
                     TileManager &tile_manager)
    : device_(device),
      denoise_device_(denoise_device),
      film_(film),
      device_scene_(device_scene),
      render_scheduler_(render_scheduler),
      tile_manager_(tile_manager)
{
  DCHECK_NE(device_, nullptr);

  {
    vector<DeviceInfo> cpu_devices;
    device_cpu_info(cpu_devices);

    cpu_device_ = device_cpu_create(
        cpu_devices[0], device->stats, device->profiler, device_->headless);
  }

  /* Create path tracing work in advance, so that it can be reused by incremental sampling as much
   * as possible. */
  device_->foreach_device([&](Device *path_trace_device) {
    unique_ptr<PathTraceWork> work = PathTraceWork::create(
        path_trace_device, film, device_scene, &render_cancel_.is_requested);
    if (work) {
      path_trace_works_.emplace_back(std::move(work));
    }
  });

  work_balance_infos_.resize(path_trace_works_.size());
  work_balance_do_initial(work_balance_infos_);

  render_scheduler.set_need_schedule_rebalance(path_trace_works_.size() > 1);
}

PathTrace::~PathTrace()
{
  destroy_gpu_resources();
}

void PathTrace::load_kernels()
{
  if (denoiser_) {
    /* Activate graphics interop while denoiser device is created, so that it can choose a device
     * that supports interop for faster display updates. */
    if (display_ && path_trace_works_.size() > 1) {
      display_->graphics_interop_activate();
    }

    denoiser_->load_kernels(progress_);

    if (display_ && path_trace_works_.size() > 1) {
      display_->graphics_interop_deactivate();
    }
  }
}

void PathTrace::alloc_work_memory()
{
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->alloc_work_memory();
  }
}

bool PathTrace::ready_to_reset()
{
  /* The logic here is optimized for the best feedback in the viewport, which implies having a GPU
   * display. Of there is no such display, the logic here will break. */
  DCHECK(display_);

  /* The logic here tries to provide behavior which feels the most interactive feel to artists.
   * General idea is to be able to reset as quickly as possible, while still providing interactive
   * feel.
   *
   * If the render result was ever drawn after previous reset, consider that reset is now possible.
   * This way camera navigation gives the quickest feedback of rendered pixels, regardless of
   * whether CPU or GPU drawing pipeline is used.
   *
   * Consider reset happening after redraw "slow" enough to not clog anything. This is a bit
   * arbitrary, but seems to work very well with viewport navigation in Blender. */

  if (did_draw_after_reset_) {
    return true;
  }

  return false;
}

void PathTrace::reset(const BufferParams &full_params,
                      const BufferParams &big_tile_params,
                      const bool reset_rendering)
{
  dlss_runtime_failed_ = false;

  if (big_tile_params_.modified(big_tile_params)) {
    big_tile_params_ = big_tile_params;
    render_state_.need_reset_params = true;
  }

  full_params_ = full_params;

  /* NOTE: GPU display checks for buffer modification and avoids unnecessary re-allocation.
   * It is requires to inform about reset whenever it happens, so that the redraw state tracking is
   * properly updated. */
  if (display_) {
    display_->reset(big_tile_params, reset_rendering);
  }

  render_state_.has_denoised_result = false;
  render_state_.tile_written = false;

  did_draw_after_reset_ = false;
  draws_after_reset_ = 0;
}

void PathTrace::request_denoiser_history_reset()
{
  denoiser_history_reset_requested_.store(true);
}

void PathTrace::device_free()
{
  /* Free render buffers used by the path trace work to reduce memory peak. */
  BufferParams empty_params;
  empty_params.pass_stride = 0;
  empty_params.update_offset_stride();
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->get_render_buffers()->reset(empty_params);
  }
  render_state_.need_reset_params = true;
}

void PathTrace::set_progress(Progress *progress)
{
  progress_ = progress;
}

void PathTrace::render(const RenderWork &render_work)
{
  /* Indicate that rendering has started and that it can be requested to cancel. */
  {
    const thread_scoped_lock lock(render_cancel_.mutex);
    if (render_cancel_.is_requested) {
      return;
    }
    render_cancel_.is_rendering = true;
  }

  render_pipeline(render_work);

  /* Indicate that rendering has finished, making it so thread which requested `cancel()` can carry
   * on. */
  {
    const thread_scoped_lock lock(render_cancel_.mutex);
    render_cancel_.is_rendering = false;
    render_cancel_.condition.notify_one();
  }
}

void PathTrace::render_pipeline(RenderWork render_work)
{
  /* NOTE: Only check for "instant" cancel here. The user-requested cancel via progress is
   * checked in Session and the work in the event of cancel is to be finished here. */

  render_scheduler_.set_need_schedule_cryptomatte(device_scene_->data.film.cryptomatte_passes !=
                                                  0);

  if (render_work.dlss.jitter_from_iteration) {
    /* Offline only. The viewport deliberately does not come here: `plan_iteration()` restarts the
     * Halton sequence on every iteration 0, so driving the viewport from it would give every
     * played-back frame the same sub-pixel positions. There the jitter is owned by
     * `Scene::update_camera_resolution()`, which advances a free-running Halton state once per
     * render work. */
    const DLSSIterationPlan plan = DLSSRenderController::plan_iteration(
        render_work.dlss.iteration, render_work.dlss.reset_history, render_work.dlss.zero_motion);
    render_work.dlss.reset_history = plan.reset_history;
    render_work.dlss.zero_motion = plan.zero_motion;
    device_scene_->data.integrator.pixel_jitter = plan.jitter;

    /* The jitter changes for every independent 1-spp input without a full scene update. */
    device_->const_copy_to("data", &device_scene_->data, sizeof(device_scene_->data));
  }

  render_init_kernel_execution();
  SCOPED_DEFER(render_deinit_kernel_execution());

  render_scheduler_.report_work_begin(render_work);

  init_render_buffers(render_work);

  rebalance(render_work);

  /* Reset sample limit. */
  render_scheduler_.set_limit_samples_per_update(0);

  /* Prepare all per-thread guiding structures before we start with the next rendering
   * iteration/progression. */
  const bool use_guiding = device_scene_->data.integrator.use_guiding;
  if (use_guiding) {
    guiding_prepare_structures();
  }

  const bool has_volume = device_scene_->data.integrator.use_volumes;
  if (has_volume) {
    const uint num_rendered_samples = render_scheduler_.get_num_rendered_samples();
    const uint limit = next_power_of_two(num_rendered_samples) - num_rendered_samples;
    render_scheduler_.set_limit_samples_per_update(limit);
  }

  build_volume_froxel_grid(has_volume);
  if (render_cancel_.is_requested) {
    return;
  }

  /* Per-phase attribution of a render work, behind CYCLES_DEBUG_VIEWPORT_PHASES.
   *
   * Measuring a viewport frame as a whole has produced wrong conclusions twice, because the phases
   * scale completely differently with the scene: on a 64-object scene the trace dominates, on a
   * 742-mesh one the scene update does. */
  static const bool phases_enabled = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  const double phase_start_time = phases_enabled ? time_dt() : 0.0;
  double phase_trace_seconds = 0.0;
  double phase_denoise_seconds = 0.0;
  double phase_display_seconds = 0.0;

  {
    const double start = time_dt();
    path_trace(render_work);
    phase_trace_seconds = time_dt() - start;
  }
  if (render_cancel_.is_requested) {
    return;
  }

  /* Update the guiding field using the training data/samples collected during the rendering
   * iteration/progression. */
  const bool train_guiding = device_scene_->data.integrator.train_guiding;
  if (use_guiding && train_guiding) {
    guiding_update_structures();
  }

  adaptive_sample(render_work);
  if (render_cancel_.is_requested) {
    return;
  }

  cryptomatte_postprocess(render_work);
  if (render_cancel_.is_requested) {
    return;
  }

  {
    const double start = time_dt();
    denoise(render_work);
    phase_denoise_seconds = time_dt() - start;
  }
  if (render_cancel_.is_requested) {
    return;
  }

  denoise_volume_guiding_buffers(render_work, has_volume);
  if (render_cancel_.is_requested) {
    return;
  }

  write_tile_buffer(render_work);
  {
    const double start = time_dt();
    update_display(render_work);
    phase_display_seconds = time_dt() - start;
  }

  progress_update_if_needed(render_work);

  finalize_full_buffer_on_disk(render_work);

  if (phases_enabled) {
    const double pipeline_seconds = time_dt() - phase_start_time;
    const double other_seconds = pipeline_seconds - phase_trace_seconds -
                                 phase_denoise_seconds - phase_display_seconds;
    fprintf(stderr,
            "PHASES pipeline_ms=%.2f trace_ms=%.2f denoise_ms=%.2f display_ms=%.2f other_ms=%.2f\n",
            pipeline_seconds * 1000.0,
            phase_trace_seconds * 1000.0,
            phase_denoise_seconds * 1000.0,
            phase_display_seconds * 1000.0,
            other_seconds * 1000.0);
    fflush(stderr);
  }
}

void PathTrace::render_init_kernel_execution()
{
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->init_execution();
  }
}

void PathTrace::render_deinit_kernel_execution()
{
  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->deinit_execution();
  }
}

/* TODO(sergey): Look into `std::function` rather than using a template. Should not be a
 * measurable performance impact at runtime, but will make compilation faster and binary somewhat
 * smaller. */
template<typename Callback>
static void foreach_sliced_buffer_params(const vector<unique_ptr<PathTraceWork>> &path_trace_works,
                                         const vector<WorkBalanceInfo> &work_balance_infos,
                                         const BufferParams &buffer_params,
                                         const int overscan,
                                         const Callback &callback)
{
  const int num_works = path_trace_works.size();
  const int window_height = buffer_params.window_height;

  int current_y = 0;
  for (int i = 0; i < num_works; ++i) {
    const double weight = work_balance_infos[i].weight;
    const int slice_window_full_y = buffer_params.full_y + buffer_params.window_y + current_y;
    const int slice_window_height = max(lround(window_height * weight), 1);

    /* Disallow negative values to deal with situations when there are more compute devices than
     * scan-lines. */
    const int remaining_window_height = max(0, window_height - current_y);

    BufferParams slice_params = buffer_params;

    slice_params.full_y = max(slice_window_full_y - overscan, buffer_params.full_y);
    slice_params.window_y = slice_window_full_y - slice_params.full_y;

    if (i < num_works - 1) {
      slice_params.window_height = min(slice_window_height, remaining_window_height);
    }
    else {
      slice_params.window_height = remaining_window_height;
    }

    slice_params.height = slice_params.window_y + slice_params.window_height + overscan;
    slice_params.height = min(slice_params.height,
                              buffer_params.height + buffer_params.full_y - slice_params.full_y);

    slice_params.update_offset_stride();

    callback(path_trace_works[i].get(), slice_params);

    current_y += slice_params.window_height;
  }
}

void PathTrace::update_allocated_work_buffer_params()
{
  const int overscan = tile_manager_.get_tile_overscan();
  foreach_sliced_buffer_params(path_trace_works_,
                               work_balance_infos_,
                               big_tile_params_,
                               overscan,
                               [](PathTraceWork *path_trace_work, const BufferParams &params) {
                                 RenderBuffers *buffers = path_trace_work->get_render_buffers();
                                 buffers->reset(params);
                               });
}

static BufferParams scale_buffer_params(const BufferParams &params, const float resolution_divider)
{
  BufferParams scaled_params = params;

  scaled_params.width = max(1, int(params.width / resolution_divider));
  scaled_params.height = max(1, int(params.height / resolution_divider));

  scaled_params.window_x = int(params.window_x / resolution_divider);
  scaled_params.window_y = int(params.window_y / resolution_divider);
  scaled_params.window_width = max(1, int(params.window_width / resolution_divider));
  scaled_params.window_height = max(1, int(params.window_height / resolution_divider));

  scaled_params.full_x = int(params.full_x / resolution_divider);
  scaled_params.full_y = int(params.full_y / resolution_divider);
  scaled_params.full_width = max(1, int(params.full_width / resolution_divider));
  scaled_params.full_height = max(1, int(params.full_height / resolution_divider));

  scaled_params.update_offset_stride();

  return scaled_params;
}

void PathTrace::update_effective_work_buffer_params(const RenderWork &render_work)
{
  const float denoised_resolution_divider = render_work.denoised_resolution_divider;
  const float resolution_divider = render_work.resolution_divider / denoised_resolution_divider;

  const BufferParams denoised_big_tile_params = scale_buffer_params(big_tile_params_,
                                                                    denoised_resolution_divider);
  const BufferParams scaled_big_tile_params = scale_buffer_params(denoised_big_tile_params,
                                                                  resolution_divider);

  const int overscan = tile_manager_.get_tile_overscan();

  foreach_sliced_buffer_params(
      path_trace_works_,
      work_balance_infos_,
      denoised_big_tile_params,
      overscan,
      [&](PathTraceWork *path_trace_work, const BufferParams params) {
        /* Scale down the sliced buffer parameters again that were scaled by denoising upscale
         * factor above. This should match the values that would occur when slicing
         * 'scaled_big_tile_params' directly. */
        const BufferParams scaled_params = scale_buffer_params(params, resolution_divider);
        path_trace_work->set_effective_buffer_params(
            scaled_big_tile_params, scaled_params, denoised_big_tile_params, params);
      });

  render_state_.effective_big_tile_params = scaled_big_tile_params;
  render_state_.effective_denoised_big_tile_params = denoised_big_tile_params;
}

void PathTrace::update_work_buffer_params_if_needed(const RenderWork &render_work)
{
  if (render_state_.need_reset_params) {
    update_allocated_work_buffer_params();
  }

  if (render_state_.need_reset_params ||
      render_state_.resolution_divider != render_work.resolution_divider)
  {
    update_effective_work_buffer_params(render_work);
  }

  render_state_.resolution_divider = render_work.resolution_divider;
  render_state_.need_reset_params = false;
}

void PathTrace::init_render_buffers(const RenderWork &render_work)
{
  update_work_buffer_params_if_needed(render_work);

  /* Handle initialization scheduled by the render scheduler. */
  if (render_work.init_render_buffers) {
    parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
      path_trace_work->zero_render_buffers();
    });

    tile_buffer_read();
  }
}

void PathTrace::path_trace(RenderWork &render_work)
{
  if (!render_work.path_trace.num_samples) {
    return;
  }

  LOG_DEBUG << "Will path trace " << render_work.path_trace.num_samples
            << " samples at the resolution divider " << render_work.resolution_divider;

  const double start_time = time_dt();

  const int num_works = path_trace_works_.size();

  thread_capture_fp_settings();

  parallel_for(0, num_works, [&](int i) {
    const double work_start_time = time_dt();
    const int num_samples = render_work.path_trace.num_samples;

    PathTraceWork *path_trace_work = path_trace_works_[i].get();
    if (path_trace_work->get_device()->have_error()) {
      return;
    }

    PathTraceWork::RenderStatistics statistics;
    path_trace_work->render_samples(statistics,
                                    render_work.path_trace.start_sample,
                                    num_samples,
                                    render_work.path_trace.sample_offset);

    DCHECK(isfinite(statistics.occupancy));

    const double work_time = time_dt() - work_start_time;
    work_balance_infos_[i].time_spent += work_time;
    work_balance_infos_[i].occupancy = statistics.occupancy;

    LOG_INFO << "Rendered " << num_samples << " samples in " << work_time << " seconds ("
             << work_time / num_samples
             << " seconds per sample), occupancy: " << statistics.occupancy;
  });

  float occupancy_accum = 0.0f;
  for (const WorkBalanceInfo &balance_info : work_balance_infos_) {
    occupancy_accum += balance_info.occupancy;
  }
  const float occupancy = occupancy_accum / num_works;
  render_scheduler_.report_path_trace_occupancy(render_work, occupancy);

  render_scheduler_.report_path_trace_time(
      render_work, time_dt() - start_time, is_cancel_requested());
}

void PathTrace::adaptive_sample(RenderWork &render_work)
{
  if (!render_work.adaptive_sampling.filter) {
    return;
  }

  bool did_reschedule_on_idle = false;

  while (true) {
    LOG_DEBUG << "Will filter adaptive stopping buffer, threshold "
              << render_work.adaptive_sampling.threshold;
    if (render_work.adaptive_sampling.reset) {
      LOG_DEBUG << "Will re-calculate convergency flag for currently converged pixels.";
    }

    const double start_time = time_dt();

    uint num_active_pixels = 0;
    parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
      const uint num_active_pixels_in_work =
          path_trace_work->adaptive_sampling_converge_filter_count_active(
              render_work.adaptive_sampling.threshold, render_work.adaptive_sampling.reset);
      if (num_active_pixels_in_work) {
        atomic_add_and_fetch_u(&num_active_pixels, num_active_pixels_in_work);
      }
    });

    render_scheduler_.report_adaptive_filter_time(
        render_work, time_dt() - start_time, is_cancel_requested());

    if (num_active_pixels == 0) {
      LOG_DEBUG << "All pixels converged.";
      if (!render_scheduler_.render_work_reschedule_on_converge(render_work)) {
        break;
      }
      LOG_DEBUG << "Continuing with lower threshold.";
    }
    else if (did_reschedule_on_idle) {
      break;
    }
    else if (num_active_pixels < 128 * 128) {
      /* NOTE: The hardcoded value of 128^2 is more of an empirical value to keep GPU busy so that
       * there is no performance loss from the progressive noise floor feature.
       *
       * A better heuristic is possible here: for example, use maximum of 128^2 and percentage of
       * the final resolution. */
      if (!render_scheduler_.render_work_reschedule_on_idle(render_work)) {
        LOG_DEBUG << "Rescheduling is not possible: final threshold is reached.";
        break;
      }
      LOG_DEBUG << "Rescheduling lower threshold.";
      did_reschedule_on_idle = true;
    }
    else {
      break;
    }
  }
}

void PathTrace::set_denoiser_external_images(const DenoiserExternalImages &images)
{
  if (denoiser_) {
    denoiser_->set_external_images(images);
  }
}

void PathTrace::set_denoiser_params(const DenoiseParams &params)
{
  if (!params.use) {
    denoiser_.reset();
    render_scheduler_.set_denoiser_params(params);
    return;
  }

  GraphicsInteropDevice interop_device;
  if (display_) {
    interop_device = display_->graphics_interop_get_device();
  }

  Device *effective_denoise_device;
  Device *cpu_fallback_device = cpu_device_.get();
  const DenoiseParams effective_denoise_params = get_effective_denoise_params(
      denoise_device_, cpu_fallback_device, params, interop_device, effective_denoise_device);

  bool need_to_recreate_denoiser = false;
  if (denoiser_) {
    const DenoiseParams old_denoiser_params = denoiser_->get_params();

    const bool is_cpu_denoising = old_denoiser_params.type == DENOISER_OPENIMAGEDENOISE &&
                                  old_denoiser_params.use_gpu == false;
    const bool always_gpu_denoising = effective_denoise_params.type == DENOISER_DLSS ||
                                      effective_denoise_params.type == DENOISER_OPTIX;
    const bool requested_gpu_denoising = always_gpu_denoising ||
                                         (effective_denoise_params.type ==
                                              DENOISER_OPENIMAGEDENOISE &&
                                          effective_denoise_params.use_gpu == true);
    if (requested_gpu_denoising && is_cpu_denoising &&
        effective_denoise_device->info.type == DEVICE_CPU)
    {
      /* It won't be possible to use GPU denoising when according to user settings we have
       * only CPU as available denoising device. So we just exiting early to avoid
       * unnecessary denoiser recreation or parameters update. */
      return;
    }

    const bool is_same_denoising_device_type = old_denoiser_params.use_gpu ==
                                               effective_denoise_params.use_gpu;
    /* Optix Denoiser is not supporting CPU devices, so use_gpu option is not
     * shown in the UI and changes in the option value should not be checked. */
    if (old_denoiser_params.type == effective_denoise_params.type &&
        (is_same_denoising_device_type || always_gpu_denoising))
    {
      denoiser_->set_params(effective_denoise_params);
    }
    else {
      need_to_recreate_denoiser = true;
    }
  }
  else {
    /* if there is no denoiser and param.use is true, then we need to create it. */
    need_to_recreate_denoiser = true;
  }

  if (need_to_recreate_denoiser) {
    denoiser_ = Denoiser::create(
        effective_denoise_device, cpu_fallback_device, effective_denoise_params, interop_device);

    if (denoiser_) {
      /* Only take into account the "immediate" cancel to have interactive rendering responding to
       * navigation as quickly as possible, but allow to run denoiser after user hit Escape key
       * while doing offline rendering. */
      denoiser_->is_cancelled_cb = [this]() { return render_cancel_.is_requested; };
    }
  }

  /* Use actual parameters, if available */
  if (denoise_device_ && denoiser_) {
    render_scheduler_.set_denoiser_params(denoiser_->get_params());
  }
  else {
    render_scheduler_.set_denoiser_params(effective_denoise_params);
  }
}

void PathTrace::set_adaptive_sampling(const AdaptiveSampling &adaptive_sampling)
{
  render_scheduler_.set_adaptive_sampling(adaptive_sampling);
}

void PathTrace::cryptomatte_postprocess(const RenderWork &render_work)
{
  if (!render_work.cryptomatte.postprocess) {
    return;
  }
  LOG_DEBUG << "Perform cryptomatte work.";

  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->cryptomatte_postproces();
  });
}

/* Ask the display side for the images the model reads, and hand them to the denoiser.
 *
 * Returns whether the display side took the job on. `CYCLES_DLSS_VULKAN` turns this off, for
 * comparing the two paths on one build. */
bool PathTrace::dlss_external_images_ensure()
{
  static const bool allowed = []() {
    const char *value = getenv("CYCLES_DLSS_VULKAN");
    return (value == nullptr) || atoi(value) != 0;
  }();

  if (!allowed || display_ == nullptr || !denoiser_ ||
      denoiser_->get_params().type != DENOISER_DLSS)
  {
    return false;
  }

  /* A final render takes this path too, and measured on the rig it does: the model runs, the result
   * comes back through the render buffer, and the file that gets saved is the denoised frame.
   *
   * What cannot take it is a render with no window at all. `blender -b` has no GPU context to
   * allocate the shared images from, and there is no display driver either - which the check above
   * already catches, so there is nothing more to say here. */

  const BufferParams &render_params = render_state_.effective_big_tile_params;
  const BufferParams &output_params = render_state_.effective_denoised_big_tile_params;

  DenoiserExternalImages images;
  display_->graphics_interop_activate();
  const bool available = display_->dlss_denoiser_images_ensure(render_params.width,
                                                               render_params.height,
                                                               output_params.width,
                                                               output_params.height,
                                                               denoiser_->get_params().dlss_preset,
                                                               images);
  display_->graphics_interop_deactivate();

  if (!available) {
    if (dlss_external_active_) {
      /* Back to denoising here, which means the denoiser has to let go of images it no longer
       * owns a claim on. */
      denoiser_->set_external_images(DenoiserExternalImages());
      dlss_external_active_ = false;
    }
    return false;
  }

  if (images.is_set()) {
    /* A new set: every previous handle is stale, and the history has nothing to carry over. */
    denoiser_->set_external_images(images);
    dlss_external_reset_ = true;
  }

  /* Run from inside the denoise pass, between the kernels that fill the images and the kernel that
   * reads the result - the only order in which the result exists when it is wanted. */
  denoiser_->set_external_evaluate([this](const bool reset_history) {
    const float2 jitter = device_scene_->data.integrator.pixel_jitter;
    display_->graphics_interop_activate();
    /* Two reasons to start over: the images were remade, or the denoiser was asked to drop its
     * history - a view change, or a capture stepping to its next frame. The second used to reach
     * only the render-device path, so on this path the history survived every reset. */
    const bool evaluated = display_->dlss_denoiser_evaluate(
        jitter.x,
        jitter.y,
        dlss_external_reset_ || reset_history,
        dlss_have_camera_transforms_ ? dlss_world_to_view_ : nullptr,
        dlss_have_camera_transforms_ ? dlss_view_to_clip_ : nullptr);
    display_->graphics_interop_deactivate();
    dlss_external_reset_ = false;
    return evaluated;
  });

  dlss_external_active_ = true;
  return true;
}

void PathTrace::denoise(const RenderWork &render_work)
{
  if (!render_work.tile.denoise) {
    return;
  }

  if (!denoiser_) {
    /* Denoiser was not configured, so nothing to do here. */
    return;
  }

  if (denoiser_history_reset_requested_.exchange(false)) {
    denoiser_->reset_history();
  }

  if (render_work.dlss.iteration >= 0) {
    if (render_work.dlss.reset_history) {
      denoiser_->reset_history();
    }
    denoiser_->set_zero_motion(render_work.dlss.zero_motion);
  }
  else {
    denoiser_->set_zero_motion(false);
  }

  LOG_DEBUG << "Perform denoising work.";

  const double start_time = time_dt();

  RenderBuffers *buffer_to_denoise = nullptr;
  bool allow_inplace_modification = false;

  Device *denoiser_device = denoiser_->get_denoiser_device();
  if (path_trace_works_.size() > 1 && denoiser_device && !big_tile_denoise_work_) {
    big_tile_denoise_work_ = PathTraceWork::create(denoiser_device, film_, device_scene_, nullptr);
  }

  if (big_tile_denoise_work_) {
    big_tile_denoise_work_->set_effective_buffer_params(
        render_state_.effective_big_tile_params,
        render_state_.effective_big_tile_params,
        render_state_.effective_denoised_big_tile_params,
        render_state_.effective_denoised_big_tile_params);

    buffer_to_denoise = big_tile_denoise_work_->get_render_buffers();
    buffer_to_denoise->reset(render_state_.effective_denoised_big_tile_params);

    copy_to_render_buffers(buffer_to_denoise);

    allow_inplace_modification = true;
  }
  else {
    DCHECK_EQ(path_trace_works_.size(), 1);

    buffer_to_denoise = path_trace_works_.front()->get_render_buffers();
  }

  /* Hand the camera over for a denoiser that asks for it. The kernel camera already holds both
   * halves: `worldtocamera` is world to view, and `worldtondc` composed with `cameratoworld` is
   * view to clip. Row-major 4x4, which is what the SDK's own helpers pass. */
  {
    const KernelCamera &cam = device_scene_->data.cam;
    const ProjectionTransform view_to_clip = cam.worldtondc * cam.cameratoworld;

    const auto write_row = [](float *matrix, const int row, const float4 &value) {
      matrix[row * 4 + 0] = value.x;
      matrix[row * 4 + 1] = value.y;
      matrix[row * 4 + 2] = value.z;
      matrix[row * 4 + 3] = value.w;
    };

    float world_to_view_matrix[16];
    write_row(world_to_view_matrix, 0, cam.worldtocamera.x);
    write_row(world_to_view_matrix, 1, cam.worldtocamera.y);
    write_row(world_to_view_matrix, 2, cam.worldtocamera.z);
    write_row(world_to_view_matrix, 3, make_float4(0.0f, 0.0f, 0.0f, 1.0f));

    float view_to_clip_matrix[16];
    write_row(view_to_clip_matrix, 0, view_to_clip.x);
    write_row(view_to_clip_matrix, 1, view_to_clip.y);
    write_row(view_to_clip_matrix, 2, view_to_clip.z);
    write_row(view_to_clip_matrix, 3, view_to_clip.w);

    denoiser_->set_camera_transforms(world_to_view_matrix, view_to_clip_matrix);

    /* Kept for the evaluation below, which happens after this scope closes. */
    std::copy_n(world_to_view_matrix, 16, dlss_world_to_view_);
    std::copy_n(view_to_clip_matrix, 16, dlss_view_to_clip_);
    dlss_have_camera_transforms_ = true;
  }

  /* Whether the display side will run the model this time, on images it allocates and this side
   * fills. Asked every frame because the answer changes with the resolution and with the model
   * chosen, and it is cheap: the images are remade only when one of those changes. */
  dlss_external_images_ensure();

  if (denoiser_->denoise_buffer(render_state_.effective_big_tile_params,
                                render_state_.effective_denoised_big_tile_params,
                                buffer_to_denoise,
                                get_num_samples_in_buffer(),
                                allow_inplace_modification,
                                device_scene_->data.integrator.pixel_jitter))
  {
    render_state_.has_denoised_result = true;
  }
  else if (denoiser_->get_params().type == DENOISER_DLSS && denoiser_->get_params().dlss_offline) {
    dlss_runtime_failed_ = true;
    progress_->set_cancel("DLSS runtime evaluation failed; retrying with OptiX");
  }


  render_scheduler_.report_denoise_time(render_work, time_dt() - start_time);
}

float PathTrace::volume_froxel_far_distance()
{
  const KernelVolumeFroxel &froxel = device_scene_->data.froxel;

  /* `CYCLES_FROXEL_FAR` overrides everything, for measurements driven from a script. */
  static const float froxel_far_forced = []() {
    const char *value = getenv("CYCLES_FROXEL_FAR");
    return (value != nullptr) ? float(atof(value)) : 0.0f;
  }();
  if (froxel_far_forced > 0.0f) {
    return froxel_far_forced;
  }

  if (froxel.user_distance > 0.0f) {
    return froxel.user_distance;
  }

  /* Otherwise reach exactly as far as the media do. The camera's own far plane is no use here -
   * the defaults span ten orders of magnitude, and slices spread over that put the whole of a fog
   * bank inside one of them. The far corner of the volume bounds is the honest answer, and it is
   * what keeps the grid working in a scene nobody tuned it for. */
  if (froxel.bounds_min.w == 0.0f) {
    return 100.0f;
  }

  const float3 camera_P = transform_get_column(&device_scene_->data.cam.cameratoworld, 3);
  const float3 bounds_min = make_float3(
      froxel.bounds_min.x, froxel.bounds_min.y, froxel.bounds_min.z);
  const float3 bounds_max = make_float3(
      froxel.bounds_max.x, froxel.bounds_max.y, froxel.bounds_max.z);

  float far_distance = 0.0f;
  for (int corner = 0; corner < 8; corner++) {
    const float3 P = make_float3((corner & 1) ? bounds_max.x : bounds_min.x,
                                 (corner & 2) ? bounds_max.y : bounds_min.y,
                                 (corner & 4) ? bounds_max.z : bounds_min.z);
    far_distance = max(far_distance, len(P - camera_P));
  }

  return (far_distance > 0.0f) ? far_distance : 100.0f;
}

void PathTrace::set_volume_grid_hold(const bool hold)
{
  volume_grid_hold_.store(hold, std::memory_order_relaxed);
}

float PathTrace::volume_froxel_blend()
{
  if (device_scene_->data.froxel.always) {
    return 1.0f;
  }

  /* A capture of the moving viewport reads its frame a few accumulations in, by which point the
   * fade would already have handed a share of the pixels back to the tracer - one sample each,
   * and the reconstruction has no guide for a medium, so that share came out as grain over the
   * whole of the fog. Measured: the grain was the fade, and at full weight the fog is clean. */
  if (volume_grid_hold_.load(std::memory_order_relaxed)) {
    return 1.0f;
  }

  /* How many frames the grid takes to hand back over to the path tracer once the view settles.
   * Instant would show as a step in the picture; too slow and the approximation is what the user
   * looks at while deciding the shot. */
  static const int fade_frames = []() {
    const char *value = getenv("CYCLES_FROXEL_FADE");
    const int parsed = (value != nullptr) ? atoi(value) : 12;
    return max(parsed, 1);
  }();

  /* Any camera or scene change goes through `RenderScheduler::reset()`, which clears this count,
   * so it is exactly "how many frames since anything moved". */
  const int settled_frames = render_scheduler_.get_num_rendered_samples();
  return clamp(1.0f - float(settled_frames) / float(fade_frames), 0.0f, 1.0f);
}

void PathTrace::build_volume_froxel_grid(const bool has_volume)
{
  KernelVolumeFroxel &froxel = device_scene_->data.froxel;

  /* The scene setting is what asks for the grid. `CYCLES_FROXEL` still forces it on, because every
   * measurement here is driven from a script that has no interface to click. */
  static const bool froxel_forced = []() {
    const char *value = getenv("CYCLES_FROXEL");
    return (value != nullptr) && atoi(value) != 0;
  }();

  /* Once the fade has run out the grid accounts for none of the pixel, so there is nothing to
   * build and the frame is the honest one - which is what "While Moving" has to mean to be worth
   * having. */
  const float blend = volume_froxel_blend();
  const bool froxel_requested = froxel.requested || froxel_forced;

  /* A final render is background, and the grid is a viewport thing, so it does not run there. But
   * that also means the grid can only ever be seen next to the traced result through a screen
   * capture, and comparing a capture against a rendered file measures the display path as much as
   * the grid. Forcing it on lifts this too, so that both pictures come out of the same F12 and
   * differ in nothing but the medium. Measurement only - the setting alone never does this. */
  const bool background = render_scheduler_.is_background() && !froxel_forced;
  const bool enabled = froxel_requested && has_volume && !background && blend > 0.0f;

  /* Said once, whatever the answer: "why is the grid not on" is otherwise guesswork, and each guess
   * costs a build. */
  static bool reported_gate = false;
  if (!reported_gate) {
    reported_gate = true;
    LOG_INFO << "Volume froxel gate: requested=" << froxel_requested
                << " has_volume=" << has_volume
                << " background=" << render_scheduler_.is_background()
                << " forced=" << froxel_forced
                << " width=" << render_state_.effective_big_tile_params.width
                << " height=" << render_state_.effective_big_tile_params.height;
  }

  if (!enabled) {
    if (froxel.enabled) {
      froxel.enabled = 0;
      device_->const_copy_to("data", &device_scene_->data, sizeof(device_scene_->data));
    }
    return;
  }

  const BufferParams &params = render_state_.effective_big_tile_params;
  if (params.width <= 0 || params.height <= 0) {
    return;
  }

  const int tile = 8;
  froxel.tile = tile;
  froxel.res_x = divide_up(params.width, tile);
  froxel.res_y = divide_up(params.height, tile);
  froxel.res_z = 64;
  froxel.t_near = max(device_scene_->data.cam.nearclip, 1e-3f);
  froxel.t_far = max(volume_froxel_far_distance(), froxel.t_near * 2.0f);
  froxel.blend = blend;
  froxel.enabled = 1;

  /* Which part of the injection to run, for attributing a fault to a step rather than to the grid
   * as a whole. Zero, the default, runs all of it. */
  static const int froxel_stage = []() {
    const char *value = getenv("CYCLES_FROXEL_STAGE");
    return (value) ? atoi(value) : 0;
  }();
  froxel.stage = froxel_stage;

  /* The count comes from the scene; the variable overrides it for measurements driven from a
   * script. Eight is the default because the answer is the same every frame for a given cell
   * either way - the seed is a function of the cell - so more of them buys accuracy rather than
   * stability: with one sample, neighbouring cells pick different lights and the fog comes out in
   * soft patches. */
  static const int froxel_light_samples_forced = []() {
    const char *value = getenv("CYCLES_FROXEL_LIGHT_SAMPLES");
    return (value) ? max(atoi(value), 0) : -1;
  }();
  if (froxel_light_samples_forced >= 0) {
    froxel.light_samples = froxel_light_samples_forced;
  }

  /* Which of the lighting corrections to apply, as a bit per fix - see `VolumeFroxelFix`. All of
   * the ones that have been measured are on; the variable is what turns them off again, one at a
   * time, so a picture can be attributed to a correction rather than to the set of them.
   *
   * Measured on a cube of noise-driven smoke against a converged path trace, as a fraction of the
   * energy of the traced single scattering: the grid gave 0.00 without them - a light outside the
   * hull reached the fog not at all - 0.73 with the hull no longer counted as a wall, 0.74 with
   * the slices weighed by coverage, 0.82 with the shadow walked through the density octree, and
   * 1.00 once that walk read the density a third of the way up a node's extrema. Against the full
   * path, with multiple scattering standing in, 0.94. All of it together costs 3 to 6 percent of
   * the frame. */
  static const int froxel_fixes = []() {
    const char *value = getenv("CYCLES_FROXEL_FIX");
    return (value != nullptr) ? int(strtol(value, nullptr, 0)) :
                                (VOLUME_FROXEL_FIX_EMITTER_ESTIMATE |
                                 VOLUME_FROXEL_FIX_UNBOUNDED_SHADOW |
                                 VOLUME_FROXEL_FIX_MARCHED_SHADOW |
                                 VOLUME_FROXEL_FIX_SLICE_OVERLAP |
                                 VOLUME_FROXEL_FIX_WORLD_VOLUME |
                                 VOLUME_FROXEL_FIX_MULTI_SCATTER |
                                 VOLUME_FROXEL_FIX_HULL_SHADOW);
  }();
  froxel.fixes = froxel_fixes;

  /* Two numbers the corrections are fitted with. Both are read from the environment so that they
   * can be swept against the traced result without a rebuild; the defaults are what that sweep
   * settled on. */
  static const float froxel_sigma_mix = []() {
    const char *value = getenv("CYCLES_FROXEL_SIGMA_MIX");
    return (value != nullptr) ? float(atof(value)) : 0.33f;
  }();
  froxel.shadow_sigma_mix = froxel_sigma_mix;

  static const float froxel_multi_scatter = []() {
    const char *value = getenv("CYCLES_FROXEL_MULTI_SCATTER");
    return (value != nullptr) ? float(atof(value)) : 1.0f;
  }();
  froxel.multi_scatter_return = froxel_multi_scatter;

  const size_t num_cells = size_t(froxel.res_x) * froxel.res_y * froxel.res_z;
  if (device_scene_->volume_froxel_scatter.size() != num_cells) {
    /* `copy_to_device` is what binds the pointer a global array is read through - `zero_to_device`
     * alone leaves the kernel dereferencing null, which shows up as an illegal address inside the
     * froxel kernels rather than as anything resembling its cause. */
    float4 *scatter = device_scene_->volume_froxel_scatter.alloc(num_cells);
    memset(scatter, 0, num_cells * sizeof(float4));
    device_scene_->volume_froxel_scatter.copy_to_device();
  }

  device_->const_copy_to("data", &device_scene_->data, sizeof(device_scene_->data));

  const double start_time = time_dt();

  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->build_volume_froxel_grid();
  });

  /* Said once, because "did the grid actually turn on" is the first question of every measurement
   * and reading it off the picture is guesswork. */
  static bool announced = false;
  if (!announced) {
    announced = true;
    LOG_INFO << "Volume froxel grid active: " << froxel.res_x << "x" << froxel.res_y << "x"
             << froxel.res_z << " cells, depth " << froxel.t_near << " to " << froxel.t_far
             << ", built in " << time_dt() - start_time << " seconds.";
  }

  LOG_DEBUG << "Volume froxel grid built in " << time_dt() - start_time << " seconds.";
}

void PathTrace::denoise_volume_guiding_buffers(const RenderWork &render_work,
                                               const bool has_volume)
{
  if (!has_volume || !render_work.volume_guiding_denoise) {
    return;
  }

  LOG_DEBUG << "Denoise volume guiding buffers.";

  const double start_time = time_dt();

  /* TODO: in the multi-GPU case, we can denoise on one device and copy to the rest, instead of
   * denoising on each device separately. */
  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->denoise_volume_guiding_buffers();
  });

  render_scheduler_.report_volume_guiding_denoise_time(render_work, time_dt() - start_time);
}

void PathTrace::set_output_driver(unique_ptr<OutputDriver> driver)
{
  output_driver_ = std::move(driver);
}

void PathTrace::set_display_driver(unique_ptr<DisplayDriver> driver)
{
  /* The display driver is the source of the drawing context which might be used by
   * path trace works. Make sure there is no graphics interop using resources from
   * the old display, as it might no longer be available after this call. */
  destroy_gpu_resources();

  if (driver) {
    display_ = make_unique<PathTraceDisplay>(std::move(driver));
  }
  else {
    display_ = nullptr;
  }
}

void PathTrace::zero_display()
{
  if (display_) {
    display_->zero();
  }
}

bool PathTrace::draw()
{
  if (!display_) {
    return false;
  }

  /* Counted before the result is folded in: the point is to know the UI thread got here at all,
   * separately from whether it found fresh pixels waiting when it did. */
  draws_after_reset_++;
  const bool fresh = display_->draw();
  did_draw_after_reset_ |= fresh;
  return fresh;
}

void PathTrace::flush_display()
{
  if (!display_) {
    return;
  }

  display_->flush();
}

void PathTrace::update_display(const RenderWork &render_work)
{
  if (!render_work.display.update) {
    return;
  }

  if (!display_ && !output_driver_) {
    LOG_DEBUG << "Ignore display update.";
    return;
  }

  if (full_params_.width == 0 || full_params_.height == 0) {
    LOG_DEBUG << "Skipping PathTraceDisplay update due to 0 size of the render buffer.";
    return;
  }

  const double start_time = time_dt();

  if (output_driver_) {
    LOG_DEBUG << "Invoke buffer update callback.";

    const PathTraceTile tile(*this);
    output_driver_->update_render_tile(tile);
  }

  if (display_) {
    LOG_DEBUG << "Perform copy to GPUDisplay work.";

    const PassType pass_type = film_->get_display_pass();
    const bool show_denoised =
        ((render_work.display.use_denoised_result && has_denoised_result() &&
          big_tile_params_.get_pass_offset(pass_type, PassMode::DENOISED) != PASS_UNUSED) ||
         is_volume_guiding_pass(pass_type));

    const int texture_width = show_denoised ?
                                  render_state_.effective_denoised_big_tile_params.window_width :
                                  render_state_.effective_big_tile_params.window_width;
    const int texture_height = show_denoised ?
                                   render_state_.effective_denoised_big_tile_params.window_height :
                                   render_state_.effective_big_tile_params.window_height;
    if (!display_->update_begin(texture_width, texture_height)) {
      LOG_ERROR << "Error beginning GPUDisplay update.";
      return;
    }

    const PassMode pass_mode = show_denoised ? PassMode::DENOISED : PassMode::NOISY;

    /* TODO(sergey): When using multi-device rendering map the GPUDisplay once and copy data from
     * all works in parallel. */
    const int num_samples = get_num_samples_in_buffer();
    if (big_tile_denoise_work_ && render_state_.has_denoised_result) {
      big_tile_denoise_work_->copy_to_display(display_.get(), pass_mode, num_samples);
    }
    else {
      for (auto &&path_trace_work : path_trace_works_) {
        path_trace_work->copy_to_display(display_.get(), pass_mode, num_samples);
      }
    }

    display_->update_end();
  }

  render_scheduler_.report_display_update_time(render_work, time_dt() - start_time);
}

void PathTrace::rebalance(const RenderWork &render_work)
{
  if (!render_work.rebalance) {
    return;
  }

  const int num_works = path_trace_works_.size();

  if (num_works == 1) {
    LOG_DEBUG << "Ignoring rebalance work due to single device render.";
    return;
  }

  const double start_time = time_dt();

  if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    LOG_DEBUG << "Perform rebalance work.";
    LOG_DEBUG << "Per-device path tracing time (seconds):";
    for (int i = 0; i < num_works; ++i) {
      LOG_DEBUG << path_trace_works_[i]->get_device()->info.description << ": "
                << work_balance_infos_[i].time_spent;
    }
  }

  const bool did_rebalance = work_balance_do_rebalance(work_balance_infos_);

  if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    LOG_DEBUG << "Calculated per-device weights for works:";
    for (int i = 0; i < num_works; ++i) {
      LOG_DEBUG << path_trace_works_[i]->get_device()->info.description << ": "
                << work_balance_infos_[i].weight;
    }
  }

  if (!did_rebalance) {
    LOG_DEBUG << "Balance in path trace works did not change.";
    render_scheduler_.report_rebalance_time(render_work, time_dt() - start_time, false);
    return;
  }

  RenderBuffers big_tile_cpu_buffers(cpu_device_.get());
  big_tile_cpu_buffers.reset(render_state_.effective_big_tile_params);

  copy_to_render_buffers(&big_tile_cpu_buffers);

  render_state_.need_reset_params = true;
  update_work_buffer_params_if_needed(render_work);

  copy_from_render_buffers(&big_tile_cpu_buffers);

  render_scheduler_.report_rebalance_time(render_work, time_dt() - start_time, true);
}

void PathTrace::write_tile_buffer(const RenderWork &render_work)
{
  if (!render_work.tile.write) {
    return;
  }

  LOG_DEBUG << "Write tile result.";

  render_state_.tile_written = true;

  const bool has_multiple_tiles = tile_manager_.has_multiple_tiles();

  /* Write render tile result, but only if not using tiled rendering.
   *
   * Tiles are written to a file during rendering, and written to the software at the end
   * of rendering (wither when all tiles are finished, or when rendering was requested to be
   * canceled).
   *
   * Important thing is: tile should be written to the software via callback only once. */
  if (!has_multiple_tiles) {
    LOG_DEBUG << "Write tile result via buffer write callback.";
    tile_buffer_write();
  }
  /* Write tile to disk, so that the render work's render buffer can be re-used for the next tile.
   */
  else {
    LOG_DEBUG << "Write tile result to disk.";
    tile_buffer_write_to_disk();
  }
}

void PathTrace::finalize_full_buffer_on_disk(const RenderWork &render_work)
{
  if (!render_work.full.write) {
    return;
  }

  LOG_DEBUG << "Handle full-frame render buffer work.";

  if (!tile_manager_.has_written_tiles()) {
    LOG_DEBUG << "No tiles on disk.";
    return;
  }

  /* Make sure writing to the file is fully finished.
   * This will include writing all possible missing tiles, ensuring validness of the file. */
  tile_manager_.finish_write_tiles();

  /* NOTE: The rest of full-frame post-processing (such as full-frame denoising) will be done after
   * all scenes and layers are rendered by the Session (which happens after freeing Session memory,
   * so that we never hold scene and full-frame buffer in memory at the same time). */
}

void PathTrace::cancel()
{
  thread_scoped_lock lock(render_cancel_.mutex);

  /* Only cancel in the middle of rendering when there is at least one sample in the output.
   * Otherwise interactivity becomes bad. */
  if (render_scheduler_.is_background() || get_num_samples_in_buffer() > 1) {
    render_cancel_.is_requested = true;
  }

  while (render_cancel_.is_rendering) {
    render_cancel_.condition.wait(lock);
  }

  render_cancel_.is_requested = false;
}

int PathTrace::get_num_samples_in_buffer() const
{
  if (denoiser_ && denoiser_->get_params().type == DENOISER_DLSS) {
    /* The DLSS path normally clears the buffer before every iteration, so what it holds is one
     * iteration's worth of samples - not the running total. That is the number the denoiser and the
     * display have to scale by.
     *
     * While the buffer is accumulating - camera still, `CYCLES_DLSS_ACCUMULATE` on - it holds the
     * running total instead, and everything that divides by this number has to see the real one or
     * the frame comes out scaled by however many iterations have gone by. */
    if (render_scheduler_.is_accumulating_in_buffer()) {
      return render_scheduler_.get_num_rendered_samples();
    }
    return render_scheduler_.get_num_rendered_samples() > 0 ?
               RenderScheduler::get_dlss_samples_per_iteration() :
               0;
  }
  return render_scheduler_.get_num_rendered_samples();
}

bool PathTrace::is_cancel_requested()
{
  if (render_cancel_.is_requested) {
    return true;
  }

  if (progress_ != nullptr) {
    if (progress_->get_cancel()) {
      return true;
    }
  }

  return false;
}

void PathTrace::tile_buffer_write()
{
  if (!output_driver_) {
    return;
  }

  const PathTraceTile tile(*this);
  output_driver_->write_render_tile(tile);
}

void PathTrace::tile_buffer_read()
{
  if (!device_scene_->data.bake.use) {
    return;
  }

  if (!output_driver_) {
    return;
  }

  /* Read buffers back from device. */
  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    path_trace_work->copy_render_buffers_from_device();
  });

  /* Read (subset of) passes from output driver. */
  const PathTraceTile tile(*this);
  if (output_driver_->read_render_tile(tile)) {
    /* Copy buffers to device again. */
    parallel_for_each(path_trace_works_, [](unique_ptr<PathTraceWork> &path_trace_work) {
      path_trace_work->copy_render_buffers_to_device();
    });
  }
}

void PathTrace::tile_buffer_write_to_disk()
{
  /* Sample count pass is required to support per-tile partial results stored in the file. */
  DCHECK_NE(big_tile_params_.get_pass_offset(PASS_SAMPLE_COUNT), PASS_UNUSED);

  const int num_rendered_samples = render_scheduler_.get_num_rendered_samples();

  if (num_rendered_samples == 0) {
    /* The tile has zero samples, no need to write it. */
    return;
  }

  /* Get access to the CPU-side render buffers of the current big tile. */
  RenderBuffers *buffers;
  RenderBuffers big_tile_cpu_buffers(cpu_device_.get());

  if (path_trace_works_.size() == 1) {
    path_trace_works_[0]->copy_render_buffers_from_device();
    buffers = path_trace_works_[0]->get_render_buffers();
  }
  else {
    big_tile_cpu_buffers.reset(render_state_.effective_big_tile_params);
    copy_to_render_buffers(&big_tile_cpu_buffers);

    buffers = &big_tile_cpu_buffers;
  }

  if (!tile_manager_.write_tile(*buffers)) {
    device_->set_error("Error writing tile to file");
  }
}

void PathTrace::progress_update_if_needed(const RenderWork &render_work)
{
  if (progress_ != nullptr) {
    const int2 tile_size = get_render_tile_size();
    const uint64_t num_samples_added = uint64_t(tile_size.x) * tile_size.y *
                                       render_work.path_trace.num_samples;
    const int current_sample = render_work.path_trace.start_sample +
                               render_work.path_trace.num_samples -
                               render_work.path_trace.sample_offset;
    progress_->add_samples(num_samples_added, current_sample);
  }

  if (progress_update_cb) {
    progress_update_cb();
  }
}

void PathTrace::progress_set_status(const string &status, const string &substatus)
{
  if (progress_ != nullptr) {
    progress_->set_status(status, substatus);
  }
}

void PathTrace::copy_to_render_buffers(RenderBuffers *render_buffers)
{
  parallel_for_each(path_trace_works_,
                    [&render_buffers](unique_ptr<PathTraceWork> &path_trace_work) {
                      path_trace_work->copy_to_render_buffers(render_buffers);
                    });
  render_buffers->copy_to_device();
}

void PathTrace::copy_from_render_buffers(RenderBuffers *render_buffers)
{
  render_buffers->copy_from_device();
  parallel_for_each(path_trace_works_,
                    [&render_buffers](unique_ptr<PathTraceWork> &path_trace_work) {
                      path_trace_work->copy_from_render_buffers(render_buffers);
                    });
}

bool PathTrace::copy_render_tile_from_device()
{
  if (full_frame_state_.render_buffers) {
    /* Full-frame buffer is always allocated on CPU. */
    return true;
  }

  if (big_tile_denoise_work_ && render_state_.has_denoised_result) {
    return big_tile_denoise_work_->copy_render_buffers_from_device();
  }

  bool success = true;

  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    if (!success) {
      return;
    }
    if (!path_trace_work->copy_render_buffers_from_device()) {
      success = false;
    }
  });

  return success;
}

static string get_layer_view_name(const RenderBuffers &buffers)
{
  string result;

  if (!buffers.params.layer.empty()) {
    result += string(buffers.params.layer);
  }

  if (!buffers.params.view.empty()) {
    if (!result.empty()) {
      result += ", ";
    }
    result += string(buffers.params.view);
  }

  return result;
}

void PathTrace::process_full_buffer_from_disk(string_view filename)
{
  LOG_DEBUG << "Processing full frame buffer file " << filename;

  progress_set_status("Reading full buffer from disk");

  RenderBuffers full_frame_buffers(cpu_device_.get());

  DenoiseParams denoise_params;
  if (!tile_manager_.read_full_buffer_from_disk(filename, &full_frame_buffers, &denoise_params)) {
    const string error_message = "Error reading tiles from file";
    if (progress_) {
      progress_->set_error(error_message);
      progress_->set_cancel(error_message);
    }
    else {
      LOG_ERROR << error_message;
    }
    return;
  }

  const string layer_view_name = get_layer_view_name(full_frame_buffers);

  render_state_.has_denoised_result = false;

  if (denoise_params.use && denoiser_ && !progress_->get_cancel()) {
    progress_set_status(layer_view_name, "Denoising");

    /* If GPU should be used is not based on file metadata. */
    denoise_params.use_gpu = render_scheduler_.is_denoiser_gpu_used();

    /* Re-use the denoiser as much as possible, avoiding possible device re-initialization.
     *
     * It will not conflict with the regular rendering as:
     *  - Rendering is supposed to be finished here.
     *  - The next rendering will go via Session's `run_update_for_next_iteration` which will
     *    ensure proper denoiser is used. */
    set_denoiser_params(denoise_params);

    /* Number of samples doesn't matter too much, since the samples count pass will be used. */
    denoiser_->denoise_buffer(
        full_frame_buffers.params, full_frame_buffers.params, &full_frame_buffers, 0, false);

    render_state_.has_denoised_result = true;
  }

  full_frame_state_.render_buffers = &full_frame_buffers;

  progress_set_status(layer_view_name, "Finishing");

  /* Write the full result pretending that there is a single tile.
   * Requires some state change, but allows to use same communication API with the software. */
  tile_buffer_write();

  full_frame_state_.render_buffers = nullptr;
}

int PathTrace::get_num_render_tile_samples() const
{
  if (full_frame_state_.render_buffers) {
    return full_frame_state_.render_buffers->params.samples;
  }

  return get_num_samples_in_buffer();
}

bool PathTrace::get_render_tile_pixels(const PassAccessor &pass_accessor,
                                       const PassAccessor::Destination &destination)
{
  if (full_frame_state_.render_buffers) {
    return pass_accessor.get_render_tile_pixels(full_frame_state_.render_buffers, destination);
  }

  if (big_tile_denoise_work_ && render_state_.has_denoised_result) {
    /* Only use the big tile denoised buffer to access the denoised passes.
     * The guiding passes are allowed to be modified in-place for the needs of the denoiser,
     * so copy those from the original devices buffers. */
    if (pass_accessor.get_pass_access_info().mode == PassMode::DENOISED) {
      return big_tile_denoise_work_->get_render_tile_pixels(pass_accessor, destination);
    }
  }

  bool success = true;

  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    if (!success) {
      return;
    }
    if (!path_trace_work->get_render_tile_pixels(pass_accessor, destination)) {
      success = false;
    }
  });

  return success;
}

bool PathTrace::set_render_tile_pixels(PassAccessor &pass_accessor,
                                       const PassAccessor::Source &source)
{
  bool success = true;

  parallel_for_each(path_trace_works_, [&](unique_ptr<PathTraceWork> &path_trace_work) {
    if (!success) {
      return;
    }
    if (!path_trace_work->set_render_tile_pixels(pass_accessor, source)) {
      success = false;
    }
  });

  return success;
}

int2 PathTrace::get_render_tile_size() const
{
  if (full_frame_state_.render_buffers) {
    return make_int2(full_frame_state_.render_buffers->params.window_width,
                     full_frame_state_.render_buffers->params.window_height);
  }

  const Tile &tile = tile_manager_.get_current_tile();
  return make_int2(tile.window_width, tile.window_height);
}

int2 PathTrace::get_render_tile_input_size() const
{
  if (full_frame_state_.render_buffers) {
    return make_int2(full_frame_state_.render_buffers->params.window_width,
                     full_frame_state_.render_buffers->params.window_height);
  }
  return make_int2(render_state_.effective_big_tile_params.window_width,
                   render_state_.effective_big_tile_params.window_height);
}

int2 PathTrace::get_render_tile_offset() const
{
  if (full_frame_state_.render_buffers) {
    return make_int2(0, 0);
  }

  const Tile &tile = tile_manager_.get_current_tile();
  return make_int2(tile.x + tile.window_x, tile.y + tile.window_y);
}

int2 PathTrace::get_render_size() const
{
  return tile_manager_.get_size();
}

const BufferParams &PathTrace::get_render_tile_params() const
{
  if (full_frame_state_.render_buffers) {
    return full_frame_state_.render_buffers->params;
  }

  return big_tile_params_;
}

bool PathTrace::has_denoised_result() const
{
  return render_state_.has_denoised_result;
}

bool PathTrace::is_dlss_denoising() const
{
  return denoiser_ && denoiser_->get_params().type == DENOISER_DLSS;
}

bool PathTrace::dlss_runtime_failed() const
{
  return dlss_runtime_failed_;
}

void PathTrace::destroy_gpu_resources()
{
  /* Destroy any GPU resource which was used for graphics interop.
   * Need to have access to the PathTraceDisplay as it is the only source of drawing context which
   * is used for interop. */
  if (display_) {
    for (auto &&path_trace_work : path_trace_works_) {
      path_trace_work->destroy_gpu_resources(display_.get());
    }

    if (big_tile_denoise_work_) {
      big_tile_denoise_work_->destroy_gpu_resources(display_.get());
    }
  }
}

/* --------------------------------------------------------------------
 * Report generation.
 */

static const char *device_type_for_description(const DeviceType type)
{
  switch (type) {
    case DEVICE_NONE:
      return "None";

    case DEVICE_CPU:
      return "CPU";
    case DEVICE_CUDA:
      return "CUDA";
    case DEVICE_OPTIX:
      return "OptiX";
    case DEVICE_HIP:
      return "HIP";
    case DEVICE_HIPRT:
      return "HIPRT";
    case DEVICE_ONEAPI:
      return "oneAPI";
    case DEVICE_DUMMY:
      return "Dummy";
    case DEVICE_MULTI:
      return "Multi";
    case DEVICE_METAL:
      return "Metal";
  }

  return "UNKNOWN";
}

/* Construct description of the device which will appear in the full report. */
/* TODO(sergey): Consider making it more reusable utility. */
static string full_device_info_description(const DeviceInfo &device_info)
{
  string full_description = device_info.description;

  full_description += " (" + string(device_type_for_description(device_info.type)) + ")";

  if (device_info.display_device) {
    full_description += " (display)";
  }

  if (device_info.type == DEVICE_CPU) {
    full_description += " (" + to_string(device_info.cpu_threads) + " threads)";
  }

  full_description += " [" + device_info.id + "]";

  return full_description;
}

/* Construct string which will contain information about devices, possibly multiple of the devices.
 *
 * In the simple case the result looks like:
 *
 *   Message: Full Device Description
 *
 * If there are multiple devices then the result looks like:
 *
 *   Message: Full First Device Description
 *            Full Second Device Description
 *
 * Note that the newlines are placed in a way so that the result can be easily concatenated to the
 * full report. */
static string device_info_list_report(const string &message, const DeviceInfo &device_info)
{
  string result = "\n" + message + ": ";
  const string pad(message.length() + 2, ' ');

  if (device_info.multi_devices.empty()) {
    result += full_device_info_description(device_info) + "\n";
    result += pad +
              "    Hardware Ray-Tracing: " + (device_info.use_hardware_raytracing ? "On" : "Off") +
              "\n";
    return result;
  }

  bool is_first = true;
  for (const DeviceInfo &sub_device_info : device_info.multi_devices) {
    if (!is_first) {
      result += pad;
    }

    result += full_device_info_description(sub_device_info) + "\n";
    result += pad + "    Hardware Ray-Tracing: " +
              (sub_device_info.use_hardware_raytracing ? "On" : "Off") + "\n";

    is_first = false;
  }

  return result;
}

static string path_trace_devices_report(const vector<unique_ptr<PathTraceWork>> &path_trace_works)
{
  DeviceInfo device_info;
  device_info.type = DEVICE_MULTI;

  for (auto &&path_trace_work : path_trace_works) {
    device_info.multi_devices.push_back(path_trace_work->get_device()->info);
  }

  return device_info_list_report("Path tracing on", device_info);
}

static string denoiser_device_report(const Denoiser *denoiser)
{
  if (!denoiser) {
    return "";
  }

  if (!denoiser->get_params().use) {
    return "";
  }

  const Device *denoiser_device = denoiser->get_denoiser_device();
  if (!denoiser_device) {
    return "";
  }

  return device_info_list_report("Denoising on", denoiser_device->info);
}

string PathTrace::full_report() const
{
  string result = "Full path tracing report:\n";

  result += path_trace_devices_report(path_trace_works_);
  result += denoiser_device_report(denoiser_.get());

  /* Report from the render scheduler, which includes:
   * - Render mode (interactive, offline, headless)
   * - Adaptive sampling and denoiser parameters
   * - Breakdown of timing. */
  result += render_scheduler_.full_report();

  return result;
}

void PathTrace::set_guiding_params(const GuidingParams &guiding_params, const bool reset)
{
#if defined(WITH_PATH_GUIDING)
  if (guiding_params_.modified(guiding_params)) {
    guiding_params_ = guiding_params;

#  if !(OPENPGL_VERSION_MAJOR == 0 && OPENPGL_VERSION_MINOR <= 5)
#    define OPENPGL_USE_FIELD_CONFIG
#  endif

    if (guiding_params_.use) {
#  ifdef OPENPGL_USE_FIELD_CONFIG
      openpgl::cpp::FieldConfig field_config;
#  else
      PGLFieldArguments field_args;
#  endif
      switch (guiding_params_.type) {
        default:
        /* Parallax-aware von Mises-Fisher mixture models. */
        case GUIDING_TYPE_PARALLAX_AWARE_VMM: {
#  ifdef OPENPGL_USE_FIELD_CONFIG
          field_config.Init(
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM,
              guiding_params.deterministic);
#  else
          pglFieldArgumentsSetDefaults(
              field_args,
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);
#  endif
          break;
        }
        /* Directional quad-trees. */
        case GUIDING_TYPE_DIRECTIONAL_QUAD_TREE: {
#  ifdef OPENPGL_USE_FIELD_CONFIG
          field_config.Init(
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE,
              guiding_params.deterministic);
#  else
          pglFieldArgumentsSetDefaults(
              field_args,
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE);
#  endif
          break;
        }
        /* von Mises-Fisher mixture models. */
        case GUIDING_TYPE_VMM: {
#  ifdef OPENPGL_USE_FIELD_CONFIG
          field_config.Init(PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
                            PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_VMM,
                            guiding_params.deterministic);
#  else
          pglFieldArgumentsSetDefaults(
              field_args,
              PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE,
              PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_VMM);
#  endif
          break;
        }
      }
#  ifdef OPENPGL_USE_FIELD_CONFIG
      field_config.SetSpatialStructureArgMaxDepth(16);
#  else
      field_args.deterministic = guiding_params.deterministic;
      reinterpret_cast<PGLKDTreeArguments *>(field_args.spatialSturctureArguments)->maxDepth = 16;
#  endif
      openpgl::cpp::Device *guiding_device = static_cast<openpgl::cpp::Device *>(
          device_->get_guiding_device());
      if (guiding_device) {
        guiding_sample_data_storage_ = make_unique<openpgl::cpp::SampleStorage>();
#  ifdef OPENPGL_USE_FIELD_CONFIG
        guiding_field_ = make_unique<openpgl::cpp::Field>(guiding_device, field_config);
#  else
        guiding_field_ = make_unique<openpgl::cpp::Field>(guiding_device, field_args);
#  endif
      }
      else {
        guiding_sample_data_storage_ = nullptr;
        guiding_field_ = nullptr;
      }
    }
    else {
      guiding_sample_data_storage_ = nullptr;
      guiding_field_ = nullptr;
    }
  }
  else if (reset) {
    if (guiding_field_) {
      guiding_field_->Reset();
    }
  }
#else
  (void)guiding_params;
  (void)reset;
#endif
}

void PathTrace::guiding_prepare_structures()
{
#if defined(WITH_PATH_GUIDING)
  const bool train = (guiding_params_.training_samples == 0) ||
                     (guiding_field_->GetIteration() < guiding_params_.training_samples);

  for (auto &&path_trace_work : path_trace_works_) {
    path_trace_work->guiding_init_kernel_globals(
        guiding_field_.get(), guiding_sample_data_storage_.get(), train);
  }

  if (train) {
    /* For training the guiding distribution we need to force the number of samples
     * per update to be limited, for reproducible results and reasonable training size.
     *
     * Idea: we could stochastically discard samples with a probability of 1/num_samples_per_update
     * we can then update only after the num_samples_per_update iterations are rendered. */
    render_scheduler_.set_limit_samples_per_update(4);
  }
  else {
    render_scheduler_.set_limit_samples_per_update(0);
  }
#endif
}

void PathTrace::guiding_update_structures()
{
#if defined(WITH_PATH_GUIDING)
  LOG_DEBUG << "Update path guiding structures";

  LOG_TRACE << "Number of surface samples: " << guiding_sample_data_storage_->GetSizeSurface();
  LOG_TRACE << "Number of volume samples: " << guiding_sample_data_storage_->GetSizeVolume();

  const size_t num_valid_samples = guiding_sample_data_storage_->GetSizeSurface() +
                                   guiding_sample_data_storage_->GetSizeVolume();

  /* we wait until we have at least 1024 samples */
  if (num_valid_samples >= 1024) {
    guiding_field_->Update(*guiding_sample_data_storage_);
    guiding_update_count++;

    LOG_TRACE << "Path guiding field valid: " << guiding_field_->Validate();

    guiding_sample_data_storage_->Clear();
  }
#endif
}

CCL_NAMESPACE_END
