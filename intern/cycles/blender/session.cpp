/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "BLI_listbase.h"
#include "BLI_math_matrix.h"
#include "DEG_depsgraph_query.hh"
#include "DNA_scene_types.h"
#include "DNA_screen_types.h"
#include "DNA_space_types.h"
#include "RNA_prototypes.hh"

#include "GPU_frame_generation.hh"
#include "IMB_colormanagement.hh"

#include "device/device.h"

#include "scene/background.h"
#include "scene/bake.h"
#include "scene/camera.h"
#include "scene/film.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/stats.h"

#include "session/buffers.h"
#include "session/session.h"

#include "util/hash.h"
#include "util/log.h"
#include "util/murmurhash.h"
#include "util/path.h"
#include "util/progress.h"
#include "util/time.h"

#include "blender/display_driver.h"
#include "blender/output_driver.h"
#include "blender/session.h"
#include "blender/sync.h"
#include "blender/util.h"

CCL_NAMESPACE_BEGIN

DeviceTypeMask BlenderSession::device_override = DEVICE_MASK_ALL;
bool BlenderSession::headless = false;
bool BlenderSession::print_render_stats = false;

static bool get_show_viewport_fps(blender::Scene &b_scene)
{
  blender::PointerRNA scene_rna_ptr = RNA_id_pointer_create(&b_scene.id);
  blender::PointerRNA cscene = RNA_pointer_get(&scene_rna_ptr, "cycles");
  return get_boolean(cscene, "use_dlss_viewport_fps");
}

void BlenderSession::update_frame_generation_state()
{
  if (display_driver_ == nullptr || b_scene == nullptr || b_rv3d == nullptr || b_v3d == nullptr) {
    return;
  }

  blender::PointerRNA scene_rna_ptr = RNA_id_pointer_create(&b_scene->id);
  blender::PointerRNA cscene = RNA_pointer_get(&scene_rna_ptr, "cycles");
  const bool hdr_output = IMB_colormanagement_display_is_hdr(
      &b_scene->display_settings, b_scene->view_settings.view_transform);
  const bool enabled = get_boolean(cscene, "use_preview_denoising") &&
                       get_boolean(cscene, "use_dlss_preview") &&
                       get_boolean(cscene, "use_dlss_frame_generation") && !hdr_output;
  display_driver_->set_frame_generation_enabled(enabled);
  if (!enabled) {
    frame_generation_history_.enabled = false;
    return;
  }

  const blender::ColorManagedViewSettings &view_settings = b_scene->view_settings;
  const bool color_management_changed = frame_generation_history_.valid &&
                                        (frame_generation_history_.view_transform !=
                                             view_settings.view_transform ||
                                         frame_generation_history_.look != view_settings.look ||
                                         frame_generation_history_.exposure !=
                                             view_settings.exposure ||
                                         frame_generation_history_.gamma != view_settings.gamma);
  const bool view_changed = !frame_generation_history_.valid ||
                            !blender::compare_m4m4(
                                frame_generation_history_.persmat, b_rv3d->persmat, 1e-6f);
  const bool output_frame_changed = !frame_generation_history_.valid ||
                                    b_scene->r.cfra != frame_generation_history_.frame;
  const bool camera_changed = !frame_generation_history_.valid ||
                              b_scene->camera != frame_generation_history_.camera;
  if (frame_generation_history_.enabled && !view_changed && !output_frame_changed &&
      !camera_changed && !color_management_changed)
  {
    return;
  }

  const bool timeline_discontinuous = frame_generation_history_.valid &&
                                      b_scene->r.cfra != frame_generation_history_.frame &&
                                      b_scene->r.cfra != frame_generation_history_.frame + 1;
  if (!frame_generation_history_.enabled || color_management_changed || timeline_discontinuous ||
      camera_changed || has_dlss_reset_marker())
  {
    display_driver_->request_frame_generation_reset();
  }

  blender::gpu::FrameGenerationCamera camera;
  float camera_view_to_clip[4][4];
  blender::transpose_m4_m4(camera_view_to_clip, b_rv3d->winmat);
  std::memcpy(camera.camera_view_to_clip, camera_view_to_clip, sizeof(camera.camera_view_to_clip));
  float inverse_projection[4][4];
  blender::invert_m4_m4(inverse_projection, b_rv3d->winmat);
  float clip_to_camera_view[4][4];
  blender::transpose_m4_m4(clip_to_camera_view, inverse_projection);
  std::memcpy(camera.clip_to_camera_view, clip_to_camera_view, sizeof(camera.clip_to_camera_view));
  blender::unit_m4(reinterpret_cast<float (*)[4]>(camera.clip_to_lens_clip));

  if (frame_generation_history_.valid) {
    float clip_to_previous[4][4];
    float previous_to_clip[4][4];
    blender::mul_m4_m4m4(clip_to_previous, frame_generation_history_.persmat, b_rv3d->persinv);
    blender::mul_m4_m4m4(previous_to_clip, b_rv3d->persmat, frame_generation_history_.persinv);
    float clip_to_previous_row_major[4][4];
    float previous_to_clip_row_major[4][4];
    blender::transpose_m4_m4(clip_to_previous_row_major, clip_to_previous);
    blender::transpose_m4_m4(previous_to_clip_row_major, previous_to_clip);
    std::memcpy(
        camera.clip_to_prev_clip, clip_to_previous_row_major, sizeof(camera.clip_to_prev_clip));
    std::memcpy(
        camera.prev_clip_to_clip, previous_to_clip_row_major, sizeof(camera.prev_clip_to_clip));
  }
  else {
    blender::unit_m4(reinterpret_cast<float (*)[4]>(camera.clip_to_prev_clip));
    blender::unit_m4(reinterpret_cast<float (*)[4]>(camera.prev_clip_to_clip));
  }

  for (int axis = 0; axis < 3; axis++) {
    camera.position[axis] = b_rv3d->viewinv[3][axis];
    camera.right[axis] = b_rv3d->viewinv[0][axis];
    camera.up[axis] = b_rv3d->viewinv[1][axis];
    camera.forward[axis] = -b_rv3d->viewinv[2][axis];
  }
  camera.near_clip = b_v3d->clip_start;
  camera.far_clip = b_v3d->clip_end;
  camera.orthographic = b_rv3d->is_persp == 0;
  camera.field_of_view = 2.0f * std::atan(1.0f / std::max(fabsf(b_rv3d->winmat[1][1]), 1e-8f));
  camera.aspect_ratio = fabsf(b_rv3d->winmat[1][1] / b_rv3d->winmat[0][0]);
  display_driver_->set_frame_generation_camera(camera);

  frame_generation_history_.valid = true;
  frame_generation_history_.enabled = true;
  frame_generation_history_.frame = b_scene->r.cfra;
  frame_generation_history_.camera = b_scene->camera;
  std::memcpy(frame_generation_history_.persmat,
              b_rv3d->persmat,
              sizeof(frame_generation_history_.persmat));
  std::memcpy(frame_generation_history_.persinv,
              b_rv3d->persinv,
              sizeof(frame_generation_history_.persinv));
  frame_generation_history_.view_transform = view_settings.view_transform;
  frame_generation_history_.look = view_settings.look;
  frame_generation_history_.exposure = view_settings.exposure;
  frame_generation_history_.gamma = view_settings.gamma;
}

bool BlenderSession::use_dlss_animation_persistence() const
{
  if (!background || (b_engine.flag & blender::RE_ENGINE_FRAME_SEQUENCE) == 0 ||
      b_scene == nullptr)
  {
    return false;
  }

  blender::PointerRNA scene_rna_ptr = RNA_id_pointer_create(&b_scene->id);
  blender::PointerRNA cscene = RNA_pointer_get(&scene_rna_ptr, "cycles");
  return RNA_boolean_get(&cscene, "use_denoising") && RNA_boolean_get(&cscene, "use_dlss_render");
}

bool BlenderSession::has_dlss_reset_marker() const
{
  if (b_scene == nullptr) {
    return false;
  }

  for (const blender::TimeMarker &marker : b_scene->markers) {
    if (marker.frame == b_scene->r.cfra && STREQ(marker.name, "DLSS_RESET")) {
      return true;
    }
  }
  return false;
}

bool BlenderSession::dlss_history_needs_reset(const BufferParams &buffer_params,
                                              const string &layer,
                                              const string &view) const
{
  if (!dlss_history_.valid || has_dlss_reset_marker()) {
    return true;
  }

  return b_scene->r.cfra != dlss_history_.frame + 1 || b_scene->camera != dlss_history_.camera ||
         layer != dlss_history_.layer || view != dlss_history_.view ||
         buffer_params.width != dlss_history_.width ||
         buffer_params.height != dlss_history_.height ||
         buffer_params.full_x != dlss_history_.full_x ||
         buffer_params.full_y != dlss_history_.full_y ||
         buffer_params.window_x != dlss_history_.window_x ||
         buffer_params.window_y != dlss_history_.window_y ||
         buffer_params.window_width != dlss_history_.window_width ||
         buffer_params.window_height != dlss_history_.window_height ||
         scene->integrator->get_denoiser_upscale_factor() != dlss_history_.upscale_factor ||
         session->params.device.id != dlss_history_.device_id;
}

void BlenderSession::dlss_history_commit(const BufferParams &buffer_params,
                                         const string &layer,
                                         const string &view)
{
  dlss_history_.valid = true;
  dlss_history_.frame = b_scene->r.cfra;
  dlss_history_.camera = b_scene->camera;
  dlss_history_.layer = layer;
  dlss_history_.view = view;
  dlss_history_.width = buffer_params.width;
  dlss_history_.height = buffer_params.height;
  dlss_history_.full_x = buffer_params.full_x;
  dlss_history_.full_y = buffer_params.full_y;
  dlss_history_.window_x = buffer_params.window_x;
  dlss_history_.window_y = buffer_params.window_y;
  dlss_history_.window_width = buffer_params.window_width;
  dlss_history_.window_height = buffer_params.window_height;
  dlss_history_.upscale_factor = scene->integrator->get_denoiser_upscale_factor();
  dlss_history_.device_id = session->params.device.id;
}

BlenderSession::BlenderSession(blender::RenderEngine &b_engine,
                               blender::UserDef &b_userpref,
                               blender::Main &b_data,
                               bool preview_osl)
    : session(nullptr),
      scene(nullptr),
      sync(nullptr),
      b_engine(b_engine),
      b_userpref(b_userpref),
      b_data(&b_data),
      b_render(RE_engine_get_render_data(b_engine.re)),
      b_depsgraph(nullptr),
      b_scene(nullptr),
      b_screen(nullptr),
      b_v3d(nullptr),
      b_rv3d(nullptr),
      width(0),
      height(0),
      pixelsize(1.0f),
      preview_osl(preview_osl),
      python_thread_state(nullptr),
      use_developer_ui(b_userpref.experimental.use_cycles_debug &&
                       (b_userpref.flag & blender::USER_DEVELOPER_UI) != 0)
{
  /* offline render */
  background = true;
  last_redraw_time = 0.0;
  start_resize_time = 0.0;
  last_status_time = 0.0;
}

BlenderSession::BlenderSession(blender::RenderEngine &b_engine,
                               blender::UserDef &b_userpref,
                               blender::Main &b_data,
                               blender::bScreen *b_screen,
                               blender::View3D *b_v3d,
                               blender::RegionView3D *b_rv3d,
                               const int width,
                               const int height)
    : session(nullptr),
      scene(nullptr),
      sync(nullptr),
      b_engine(b_engine),
      b_userpref(b_userpref),
      b_data(&b_data),
      b_render(nullptr),
      b_depsgraph(nullptr),
      b_scene(nullptr),
      b_screen(b_screen),
      b_v3d(b_v3d),
      b_rv3d(b_rv3d),
      width(width),
      height(height),
      pixelsize(blender::U.pixelsize),
      preview_osl(false),
      python_thread_state(nullptr),
      use_developer_ui(b_userpref.experimental.use_cycles_debug &&
                       (b_userpref.flag & blender::USER_DEVELOPER_UI) != 0)
{
  /* 3d view render */
  background = false;
  last_redraw_time = 0.0;
  start_resize_time = 0.0;
  last_status_time = 0.0;
}

BlenderSession::~BlenderSession()
{
  free_session();
}

void BlenderSession::create_session()
{
  SessionParams session_params = BlenderSync::get_session_params(
      b_engine, b_userpref, *b_scene, background, pixelsize);
  if (use_dlss_animation_persistence()) {
    /* Offline DLSS disables tiling so every temporal input covers the whole frame. Keep the
     * reset-session parameters consistent with render(), otherwise the auto-tile difference
     * recreates the Session and NGX feature between animation frames. */
    session_params.use_auto_tile = false;
  }
  const SceneParams scene_params = BlenderSync::get_scene_params(
      b_userpref, *b_data, *b_scene, background, use_developer_ui);
  const bool session_pause = BlenderSync::get_session_pause(*b_scene, background);

  /* reset status/progress */
  last_status = "";
  last_error = "";
  last_progress = -1.0;
  start_resize_time = 0.0;

  /* create session */
  session = make_unique<Session>(session_params, scene_params);
  session->progress.set_update_callback([this] { tag_redraw(); });
  session->progress.set_cancel_callback([this] { test_cancel(); });
  session->set_pause(session_pause);

  /* create scene */
  scene = session->scene.get();
  scene->name = BKE_id_name(b_scene->id);

  /* Per-manager scene update timings are otherwise only available for final renders, which never
   * take the interactive motion pass, so viewport update cost cannot be attributed at all. Behind
   * an environment variable because the report is printed on every `Scene::device_update` and would
   * flood a playback log. */
  if (!background && getenv("CYCLES_DEBUG_VIEWPORT_UPDATE_STATS")) {
    scene->enable_update_stats();
  }

  /* create sync */
  sync = make_unique<BlenderSync>(
      b_engine, *b_data, *b_scene, scene, !background, use_developer_ui, session->progress);
  if (b_v3d) {
    sync->sync_view(b_v3d, b_rv3d, width, height);
  }
  else {
    sync->sync_camera(*b_render, width, height, "");
  }

  /* set buffer parameters */
  const BufferParams buffer_params = BlenderSync::get_buffer_params(
      b_v3d, b_rv3d, scene->camera, width, height);
  session->reset(session_params, buffer_params);

  /* Viewport and preview (as in, material preview) does not do tiled rendering, so can inform
   * engine that no tracking of the tiles state is needed.
   * The offline rendering will make a decision when tile is being written. The penalty of asking
   * the engine to keep track of tiles state is minimal, so there is nothing to worry about here
   * about possible single-tiled final render. */
  if ((b_engine.flag & blender::RE_ENGINE_PREVIEW) == 0 && !b_v3d) {
    b_engine.flag |= blender::RE_ENGINE_HIGHLIGHT_TILES;
  }
}

void BlenderSession::reset_session(blender::Main &b_data, blender::Depsgraph &b_depsgraph)
{
  /* Update data, scene and depsgraph pointers. These can change after undo. */
  this->b_data = &b_data;
  this->b_depsgraph = &b_depsgraph;
  this->b_scene = DEG_get_evaluated_scene(&b_depsgraph);
  if (sync) {
    sync->reset(*this->b_data, *this->b_scene);
  }

  if (preview_osl) {
    blender::PointerRNA scene_rna_ptr = RNA_id_pointer_create(&b_scene->id);
    blender::PointerRNA cscene = RNA_pointer_get(&scene_rna_ptr, "cycles");
    RNA_boolean_set(&cscene, "shading_system", preview_osl);
  }

  if (b_v3d) {
    this->b_render = &b_scene->r;
  }
  else {
    this->b_render = RE_engine_get_render_data(b_engine.re);
    width = render_resolution_x(*b_render);
    height = render_resolution_y(*b_render);
  }

  const bool is_new_session = (session == nullptr);
  if (is_new_session) {
    /* Initialize session and remember it was just created so not to
     * re-create it below.
     */
    create_session();
  }

  if (b_v3d) {
    /* NOTE: We need to create session, but all the code from below
     * will make viewport render to stuck on initialization.
     */
    return;
  }

  SessionParams session_params = BlenderSync::get_session_params(
      b_engine, b_userpref, *b_scene, background, pixelsize);
  if (use_dlss_animation_persistence()) {
    session_params.use_auto_tile = false;
  }
  const SceneParams scene_params = BlenderSync::get_scene_params(
      b_userpref, b_data, *b_scene, background, use_developer_ui);

  const bool effective_persistent_data = (this->b_render->mode & blender::R_PERSISTENT_DATA) !=
                                             0 ||
                                         use_dlss_animation_persistence();
  if (scene->params.modified(scene_params) || session->params.modified(session_params) ||
      !effective_persistent_data)
  {
    /* if scene or session parameters changed, it's easier to simply re-create
     * them rather than trying to distinguish which settings need to be updated
     */
    if (!is_new_session) {
      free_session();
      create_session();
    }
    return;
  }

  session->progress.reset();

  /* peak memory usage should show current render peak, not peak for all renders
   * made by this render session
   */
  session->stats.mem_peak = session->stats.mem_used;

  if (is_new_session) {
    /* Sync object should be re-created for new scene. */
    sync = make_unique<BlenderSync>(
        b_engine, b_data, *b_scene, scene, !background, use_developer_ui, session->progress);
  }
  else {
    /* Sync recalculations to do just the required updates. */
    sync->sync_recalc(b_depsgraph, b_screen, b_v3d, b_rv3d);
  }

  sync->sync_camera(*b_render, width, height, "");

  const BufferParams buffer_params = BlenderSync::get_buffer_params(
      nullptr, nullptr, scene->camera, width, height);
  session->reset(session_params, buffer_params);

  /* reset time */
  start_resize_time = 0.0;

  {
    const thread_scoped_lock lock(draw_state_.mutex);
    draw_state_.last_pass_index = -1;
  }
}

void BlenderSession::free_session()
{
  if (session) {
    session->cancel(true);
  }

  sync.reset();
  session.reset();
  dlss_history_.valid = false;
  dlss_runtime_failed_ = false;
  dlss_animation_persistence_active_ = false;
  frame_generation_history_ = {};

  display_driver_ = nullptr;
}

void BlenderSession::full_buffer_written(string_view filename)
{
  full_buffer_files_.emplace_back(filename);
}

static void add_cryptomatte_layer(blender::RenderResult &b_rr, string name, string manifest)
{
  const string identifier = string_printf("%08x",
                                          util_murmur_hash3(name.c_str(), name.length(), 0));
  const string prefix = "cryptomatte/" + identifier.substr(0, 7) + "/";

  render_add_metadata(b_rr, prefix + "name", name);
  render_add_metadata(b_rr, prefix + "hash", "MurmurHash3_32");
  render_add_metadata(b_rr, prefix + "conversion", "uint32_to_float32");
  render_add_metadata(b_rr, prefix + "manifest", manifest);
}

void BlenderSession::stamp_view_layer_metadata(Scene *scene,
                                               const string &view_layer_name,
                                               const bool dlss_requested)
{
  blender::RenderResult *b_rr = RE_engine_get_result(&b_engine);
  const string prefix = "cycles." + view_layer_name + ".";

  /* Configured number of samples for the view layer. */
  BKE_render_result_stamp_data(
      b_rr, (prefix + "samples").c_str(), to_string(session->params.samples).c_str());

  const bool dlss_effective = scene->integrator->get_use_denoise() &&
                              scene->integrator->get_denoiser_type() == DENOISER_DLSS;
  BKE_render_result_stamp_data(
      b_rr, (prefix + "dlss_requested").c_str(), dlss_requested ? "true" : "false");
  const char *effective_denoiser = dlss_effective                        ? "DLSS" :
                                   !scene->integrator->get_use_denoise() ? "Disabled" :
                                   scene->integrator->get_denoiser_type() == DENOISER_OPTIX ?
                                                                           "OptiX" :
                                                                           "Other";
  BKE_render_result_stamp_data(b_rr, (prefix + "dlss_effective").c_str(), effective_denoiser);
  if (dlss_requested && !dlss_effective) {
    const char *reason = !scene->integrator->get_use_denoise() ?
                             "Denoising disabled for this View Layer" :
                         scene->film->get_cryptomatte_passes() != CRYPT_NONE ?
                             "Cryptomatte requires full-resolution unfiltered passes" :
                             "DLSS unavailable on the selected device, driver, or runtime";
    BKE_render_result_stamp_data(b_rr, (prefix + "dlss_fallback_reason").c_str(), reason);
  }

  /* Store ranged samples information. */
  /* TODO(sergey): Need to bring this information back. */
#if 0
  if (session->tile_manager.range_num_samples != -1) {
    b_rr.stamp_data_add_field((prefix + "range_start_sample").c_str(),
                              to_string(session->tile_manager.range_start_sample).c_str());
    b_rr.stamp_data_add_field((prefix + "range_num_samples").c_str(),
                              to_string(session->tile_manager.range_num_samples).c_str());
  }
#endif

  /* Write cryptomatte metadata. */
  if (scene->film->get_cryptomatte_passes() & CRYPT_OBJECT) {
    add_cryptomatte_layer(*b_rr,
                          view_layer_name + ".CryptoObject",
                          scene->object_manager->get_cryptomatte_objects(scene));
  }
  if (scene->film->get_cryptomatte_passes() & CRYPT_MATERIAL) {
    add_cryptomatte_layer(*b_rr,
                          view_layer_name + ".CryptoMaterial",
                          scene->shader_manager->get_cryptomatte_materials(scene));
  }
  if (scene->film->get_cryptomatte_passes() & CRYPT_ASSET) {
    add_cryptomatte_layer(*b_rr,
                          view_layer_name + ".CryptoAsset",
                          scene->object_manager->get_cryptomatte_assets(scene));
  }

  /* Store synchronization and bare-render times. */
  double total_time;
  double render_time;
  session->progress.get_time(total_time, render_time);
  BKE_render_result_stamp_data(
      b_rr, (prefix + "total_time").c_str(), time_human_readable_from_seconds(total_time).c_str());
  BKE_render_result_stamp_data(b_rr,
                               (prefix + "render_time").c_str(),
                               time_human_readable_from_seconds(render_time).c_str());
  BKE_render_result_stamp_data(b_rr,
                               (prefix + "synchronization_time").c_str(),
                               time_human_readable_from_seconds(total_time - render_time).c_str());
}

void BlenderSession::render(blender::Depsgraph &b_depsgraph_)
{
  b_depsgraph = &b_depsgraph_;

  if (session->progress.get_cancel()) {
    update_status_progress();
    return;
  }

  /* Create driver to write out render results. */
  ensure_display_driver_if_needed();
  session->set_output_driver(make_unique<BlenderOutputDriver>(b_engine));

  session->full_buffer_written_cb = [&](string_view filename) { full_buffer_written(filename); };

  blender::ViewLayer &b_view_layer = *DEG_get_evaluated_view_layer(b_depsgraph);

  /* get buffer parameters */
  const SessionParams session_params = BlenderSync::get_session_params(
      b_engine, b_userpref, *b_scene, background, pixelsize);
  BufferParams buffer_params = BlenderSync::get_buffer_params(
      b_v3d, b_rv3d, scene->camera, width, height);

  /* temporary render result to find needed passes and views */
  blender::RenderResult *b_rr = RE_engine_begin_result(
      &b_engine, 0, 0, 1, 1, b_view_layer.name, nullptr);
  blender::RenderLayer *b_rlay = static_cast<blender::RenderLayer *>(b_rr->layers.first);

  {
    const thread_scoped_lock lock(draw_state_.mutex);
    b_rlay_name = b_view_layer.name;

    /* Signal that the display pass is to be updated. */
    draw_state_.last_pass_index = -1;
  }

  /* Compute render passes and film settings. */
  sync->sync_render_passes(*b_rlay, b_view_layer);

  const int num_views = b_rr->views.count();
  blender::PointerRNA scene_rna_ptr = RNA_id_pointer_create(&b_scene->id);
  blender::PointerRNA cscene = RNA_pointer_get(&scene_rna_ptr, "cycles");
  const bool dlss_requested = RNA_boolean_get(&cscene, "use_dlss_render");
  dlss_animation_persistence_active_ = false;

  for (const auto [view_index, b_view] : b_rr->views.enumerate()) {
    b_rview_name = b_view.name;

    buffer_params.layer = b_view_layer.name;
    buffer_params.view = b_rview_name;

    /* set the current view */
    RE_engine_active_view_set(&b_engine, b_rview_name.c_str());

    /* Force update in this case, since the camera transform on each frame changes
     * in different views. This could be optimized by somehow storing the animated
     * camera transforms separate from the fixed stereo transform. */
    if ((scene->need_motion() != Scene::MOTION_NONE) && view_index > 0) {
      sync->tag_update();
    }

    /* update scene */
    sync->sync_camera(*b_render, width, height, b_rview_name.c_str());
    sync->sync_data(*b_render,
                    *b_depsgraph,
                    b_screen,
                    b_v3d,
                    b_rv3d,
                    width,
                    height,
                    &python_thread_state,
                    session_params.denoise_device);

    const bool offline_dlss = scene->integrator->get_use_denoise() &&
                              scene->integrator->get_denoiser_type() == DENOISER_DLSS &&
                              scene->integrator->get_dlss_offline();
    dlss_animation_persistence_active_ |= offline_dlss &&
                                          (b_engine.flag & blender::RE_ENGINE_FRAME_SEQUENCE) != 0;
    if (dlss_animation_persistence_active_) {
      b_engine.flag |= blender::RE_ENGINE_FORCE_PERSISTENT_DATA;
    }
    else {
      b_engine.flag &= ~blender::RE_ENGINE_FORCE_PERSISTENT_DATA;
    }

    /* At the moment we only free if we are not doing multi-view
     * (or if we are rendering the last view). See #58142/D4239 for discussion.
     */
    const bool can_free_cache = (view_index == num_views - 1);
    if (can_free_cache) {
      sync->free_data_after_sync(*b_depsgraph);
    }

    builtin_images_load();

    /* Attempt to free all data which is held by Blender side, since at this
     * point we know that we've got everything to render current view layer.
     */
    if (can_free_cache && !offline_dlss) {
      free_blender_memory_if_possible();
    }

    /* Make sure all views have different noise patterns. - hardcoded value just to make it
     * random
     */
    if (view_index != 0) {
      int seed = scene->integrator->get_seed();
      seed += hash_uint2(seed, hash_uint2(view_index * 0xdeadbeef, 0));
      scene->integrator->set_seed(seed);
    }

    if (dlss_runtime_failed_ && scene->integrator->get_denoiser_type() == DENOISER_DLSS) {
      scene->integrator->set_denoiser_type(DENOISER_OPTIX);
      scene->integrator->set_denoiser_upscale_factor(1.0f);
      scene->integrator->set_dlss_offline(false);
      LOG_WARNING << "DLSS disabled for the remainder of this render session after a runtime "
                     "failure. Using OptiX.";
    }

    /* Update number of samples per layer. */
    const int samples = sync->get_layer_samples();
    const bool bound_samples = sync->get_layer_bound_samples();

    SessionParams effective_session_params = session_params;
    if (samples != 0 && (!bound_samples || (samples < session_params.samples))) {
      effective_session_params.samples = samples;
    }

    /* Every independent DLSS input must cover the entire frame before the next temporal
     *
     * iteration. Keep normal auto-tiling when DLSS was requested but fell back to OptiX. */
    effective_session_params.use_auto_tile = !offline_dlss;

    /* Update session itself. */
    session->reset(effective_session_params, buffer_params);

    bool commit_dlss_history = false;
    if (offline_dlss) {
      blender::PointerRNA scene_rna_ptr = RNA_id_pointer_create(&b_scene->id);
      blender::PointerRNA cscene = RNA_pointer_get(&scene_rna_ptr, "cycles");

      const bool animation = (b_engine.flag & blender::RE_ENGINE_FRAME_SEQUENCE) != 0;
      const bool reset_history = !animation || dlss_history_needs_reset(
                                                   buffer_params, b_rlay_name, b_rview_name);
      const int iterations = animation ? (reset_history ?
                                              RNA_int_get(&cscene, "dlss_reset_iterations") :
                                              RNA_int_get(&cscene, "dlss_animation_iterations")) :
                                         RNA_int_get(&cscene, "dlss_still_iterations");

      scene->integrator->set_dlss_animation(animation);
      scene->integrator->set_dlss_reset_history(reset_history);
      scene->integrator->set_dlss_zero_motion_first(!animation || reset_history);
      scene->integrator->set_dlss_iterations(max(iterations, 1));
      commit_dlss_history = animation;
    }

    /* render */
    if ((b_engine.flag & blender::RE_ENGINE_PREVIEW) == 0 && background && print_render_stats) {
      scene->enable_update_stats();
    }

    session->start();
    session->wait();

    if (offline_dlss && session->dlss_runtime_failed()) {
      dlss_runtime_failed_ = true;
      dlss_history_.valid = false;
      commit_dlss_history = false;
      dlss_animation_persistence_active_ = false;
      b_engine.flag &= ~blender::RE_ENGINE_FORCE_PERSISTENT_DATA;

      LOG_WARNING << "DLSS runtime evaluation failed. Restarting this frame at full resolution "
                     "with OptiX.";
      scene->integrator->set_denoiser_type(DENOISER_OPTIX);
      scene->integrator->set_denoiser_upscale_factor(1.0f);
      scene->integrator->set_dlss_offline(false);
      scene->integrator->set_dlss_animation(false);
      scene->integrator->set_dlss_reset_history(true);
      scene->integrator->set_dlss_zero_motion_first(true);
      scene->integrator->set_dlss_iterations(1);

      SessionParams fallback_session_params = effective_session_params;
      fallback_session_params.use_auto_tile = true;
      session->progress.reset();
      session->reset(fallback_session_params, buffer_params);
      session->start();
      session->wait();
    }

    if (commit_dlss_history) {
      if (session->progress.get_cancel() || session->progress.get_error()) {
        dlss_history_.valid = false;
      }
      else {
        dlss_history_commit(buffer_params, b_rlay_name, b_rview_name);
      }
    }

    if ((b_engine.flag & blender::RE_ENGINE_PREVIEW) == 0 && background && print_render_stats) {
      RenderStats stats;
      session->collect_statistics(&stats);
      printf("Render statistics:\n%s\n", stats.full_report().c_str());
    }

    if (session->progress.get_cancel()) {
      break;
    }
  }

  /* add metadata */
  stamp_view_layer_metadata(scene, b_rlay_name, dlss_requested);

  /* free result without merging */
  RE_engine_end_result(&b_engine, b_rr, true, false, false);

  /* When tiled rendering is used there will be no "write" done for the tile. Forcefully clear
   * highlighted tiles now, so that the highlight will be removed while processing full frame
   * from file. */
  RE_engine_tile_highlight_clear_all(&b_engine);

  double total_time;
  double render_time;
  session->progress.get_time(total_time, render_time);
  if (print_render_stats) {
    printf("CYCLES_RENDER_CORE_SECONDS=%.9f\n", render_time);
    printf("CYCLES_RENDER_EFFECTIVE_DENOISER=%s\n",
           scene->integrator->get_denoiser_type() == DENOISER_DLSS ? "DLSS" : "OPTIX_OR_OTHER");
  }
  LOG_INFO << "Total render time: " << total_time;
  LOG_INFO << "Render time (without synchronization): " << render_time;
}

void BlenderSession::render_frame_finish()
{
  /* Processing of all layers and views is done. Clear the strings so that we can communicate
   * progress about reading files and denoising them. */
  b_rlay_name = "";
  b_rview_name = "";

  const bool effective_persistent_data = (b_render->mode & blender::R_PERSISTENT_DATA) != 0 ||
                                         dlss_animation_persistence_active_;
  if (!effective_persistent_data) {
    /* Free the sync object so that it can properly dereference nodes from the scene graph before
     * the graph is freed. */
    sync.reset();

    session->device_free();
  }

  for (const string_view filename : full_buffer_files_) {
    session->process_full_buffer_from_disk(filename);
    if (check_and_report_session_error()) {
      break;
    }
  }

  for (const string_view filename : full_buffer_files_) {
    path_remove(filename);
  }

  /* Clear output driver. */
  session->set_output_driver(nullptr);
  session->full_buffer_written_cb = nullptr;

  /* The display driver is the source of drawing context for both drawing and possible graphics
   * interoperability objects in the path trace. Once the frame is finished the OpenGL context
   * might be freed form Blender side. Need to ensure that all GPU resources are freed prior to
   * that point.
   * Ideally would only do this when OpenGL context is actually destroyed, but there is no way to
   * know when this happens (at least in the code at the time when this comment was written).
   * The penalty of re-creating resources on every frame is unlikely to be noticed. */
  display_driver_ = nullptr;
  session->set_display_driver(nullptr);

  /* All the files are handled.
   * Clear the list so that this session can be re-used by Persistent Data. */
  full_buffer_files_.clear();
}

static bool bake_setup_pass(Scene *scene, const string &bake_type, const int bake_filter)
{
  Integrator *integrator = scene->integrator;
  Film *film = scene->film;

  const bool filter_direct = (bake_filter & blender::R_BAKE_PASS_FILTER_DIRECT) != 0;
  const bool filter_indirect = (bake_filter & blender::R_BAKE_PASS_FILTER_INDIRECT) != 0;
  const bool filter_color = (bake_filter & blender::R_BAKE_PASS_FILTER_COLOR) != 0;

  PassType type = PASS_NONE;
  bool use_direct_light = false;
  bool use_indirect_light = false;
  bool include_albedo = false;

  /* Data passes. */
  if (bake_type == "POSITION") {
    type = PASS_POSITION;
  }
  else if (bake_type == "NORMAL") {
    type = PASS_NORMAL;
  }
  else if (bake_type == "UV") {
    type = PASS_UV;
  }
  else if (bake_type == "ROUGHNESS") {
    type = PASS_ROUGHNESS;
  }
  else if (bake_type == "EMIT") {
    type = PASS_EMISSION;
  }
  /* Environment pass. */
  else if (bake_type == "ENVIRONMENT") {
    type = PASS_BACKGROUND;
  }
  /* AO pass. */
  else if (bake_type == "AO") {
    type = PASS_AO;
  }
  /* Shadow pass. */
  else if (bake_type == "SHADOW") {
    /* Bake as combined pass, together with marking the object as a shadow catcher. */
    type = PASS_SHADOW_CATCHER;
    film->set_use_approximate_shadow_catcher(true);

    use_direct_light = true;
    use_indirect_light = true;
    include_albedo = true;

    integrator->set_use_diffuse(true);
    integrator->set_use_glossy(true);
    integrator->set_use_transmission(true);
    integrator->set_use_emission(true);
  }
  /* Combined pass. */
  else if (bake_type == "COMBINED") {
    type = PASS_COMBINED;
    film->set_use_approximate_shadow_catcher(true);

    use_direct_light = filter_direct;
    use_indirect_light = filter_indirect;
    include_albedo = filter_color;

    integrator->set_use_diffuse((bake_filter & blender::R_BAKE_PASS_FILTER_DIFFUSE) != 0);
    integrator->set_use_glossy((bake_filter & blender::R_BAKE_PASS_FILTER_GLOSSY) != 0);
    integrator->set_use_transmission((bake_filter & blender::R_BAKE_PASS_FILTER_TRANSM) != 0);
    integrator->set_use_emission((bake_filter & blender::R_BAKE_PASS_FILTER_EMIT) != 0);
  }
  /* Light component passes. */
  else if ((bake_type == "DIFFUSE") || (bake_type == "GLOSSY") || (bake_type == "TRANSMISSION")) {
    use_direct_light = filter_direct;
    use_indirect_light = filter_indirect;
    include_albedo = filter_color;

    integrator->set_use_diffuse(bake_type == "DIFFUSE");
    integrator->set_use_glossy(bake_type == "GLOSSY");
    integrator->set_use_transmission(bake_type == "TRANSMISSION");

    if (bake_type == "DIFFUSE") {
      if (filter_direct && filter_indirect) {
        type = PASS_DIFFUSE;
      }
      else if (filter_direct) {
        type = PASS_DIFFUSE_DIRECT;
      }
      else if (filter_indirect) {
        type = PASS_DIFFUSE_INDIRECT;
      }
      else {
        type = PASS_DIFFUSE_COLOR;
      }
    }
    else if (bake_type == "GLOSSY") {
      if (filter_direct && filter_indirect) {
        type = PASS_GLOSSY;
      }
      else if (filter_direct) {
        type = PASS_GLOSSY_DIRECT;
      }
      else if (filter_indirect) {
        type = PASS_GLOSSY_INDIRECT;
      }
      else {
        type = PASS_GLOSSY_COLOR;
      }
    }
    else if (bake_type == "TRANSMISSION") {
      if (filter_direct && filter_indirect) {
        type = PASS_TRANSMISSION;
      }
      else if (filter_direct) {
        type = PASS_TRANSMISSION_DIRECT;
      }
      else if (filter_indirect) {
        type = PASS_TRANSMISSION_INDIRECT;
      }
      else {
        type = PASS_TRANSMISSION_COLOR;
      }
    }
  }

  if (type == PASS_NONE) {
    return false;
  }

  /* Create pass. */
  Pass *pass = scene->create_node<Pass>();
  pass->set_name(ustring("Combined"));
  pass->set_type(type);
  pass->set_include_albedo(include_albedo);

  /* Disable direct indirect light for performance when not needed. */
  integrator->set_use_direct_light(use_direct_light);
  integrator->set_use_indirect_light(use_indirect_light);

  /* Disable denoiser if the pass does not support it.
   * For the passes which support denoising follow the user configuration. */
  const PassInfo pass_info = Pass::get_info(type);
  if (integrator->get_use_denoise() && !pass_info.support_denoise) {
    integrator->set_use_denoise(false);
  }

  return true;
}

void BlenderSession::bake(blender::Depsgraph &b_depsgraph_,
                          blender::Object &b_object,
                          const string &bake_type,
                          const int bake_filter,
                          const int bake_width,
                          const int bake_height)
{
  b_depsgraph = &b_depsgraph_;

  /* Get session parameters. */
  const SessionParams session_params = BlenderSync::get_session_params(
      b_engine, b_userpref, *b_scene, background, pixelsize);

  /* Initialize bake manager, before we load the baking kernels. */
  scene->bake_manager->set_baking(scene, true);

  session->set_display_driver(nullptr);
  session->set_output_driver(make_unique<BlenderOutputDriver>(b_engine));
  session->full_buffer_written_cb = [&](string_view filename) { full_buffer_written(filename); };

  /* Sync scene. */
  sync->set_bake_target(b_object);
  sync->sync_camera(*b_render, width, height, "");
  sync->sync_data(*b_render,
                  *b_depsgraph,
                  b_screen,
                  b_v3d,
                  b_rv3d,
                  width,
                  height,
                  &python_thread_state,
                  session_params.denoise_device);

  /* Save the current state of the denoiser, as it might be disabled by the pass configuration
   * (for passed which do not support denoising). */
  Integrator *integrator = scene->integrator;
  const bool was_denoiser_enabled = integrator->get_use_denoise();

  /* Add render pass that we want to bake, and name it Combined so that it is
   * used as that on the Blender side. */
  if (!bake_setup_pass(scene, bake_type, bake_filter)) {
    session->cancel(true);
  }

  /* Always use transparent background for baking. */
  scene->background->set_transparent(true);

  if (!session->progress.get_cancel()) {
    /* Load built-in images from Blender. */
    builtin_images_load();
  }

  /* Object might have been disabled for rendering or excluded in some
   * other way, in that case Blender will report a warning afterwards. */
  Object *bake_object = nullptr;
  if (!session->progress.get_cancel()) {
    for (Object *ob : scene->objects) {
      if (ob->get_is_bake_target()) {
        bake_object = ob;
        break;
      }
    }
  }

  /* For the shadow pass, temporarily mark the object as a shadow catcher. */
  const bool was_shadow_catcher = (bake_object) ? bake_object->get_is_shadow_catcher() : false;
  if (bake_object && bake_type == "SHADOW") {
    bake_object->set_is_shadow_catcher(true);
  }

  if (bake_object && !session->progress.get_cancel()) {
    /* Get buffer parameters. */
    BufferParams buffer_params;
    buffer_params.width = bake_width;
    buffer_params.height = bake_height;
    buffer_params.window_width = bake_width;
    buffer_params.window_height = bake_height;
    /* Unique layer name for multi-image baking. */
    buffer_params.layer = string_printf("bake_%d\n", bake_id++);

    /* Update session. */
    session->reset(session_params, buffer_params);

    session->progress.set_update_callback([this] { update_bake_progress(); });
  }

  /* Perform bake. Check cancel to avoid crash with incomplete scene data. */
  if (bake_object && !session->progress.get_cancel()) {
    session->start();
    session->wait();
  }

  /* Restore object state. */
  if (bake_object) {
    bake_object->set_is_shadow_catcher(was_shadow_catcher);
  }

  /* Restore the state of denoiser to before it was possibly disabled by the pass, so that the
   * next baking pass can use the original value. */
  integrator->set_use_denoise(was_denoiser_enabled);
}

void BlenderSession::synchronize(blender::Depsgraph &b_depsgraph_)
{
  /* only used for viewport render */
  if (!b_v3d) {
    return;
  }

  /* Splits the per-frame cost that sits outside `Scene::device_update`: how much of it is Cycles
   * reading the evaluated scene back out of Blender, and how much is everything else. Wall-clock
   * only, four numbers per frame - anything finer distorts what it measures. */
  static const bool report_sync = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  const double sync_start_time = report_sync ? time_dt() : 0.0;
  double recalc_ms = 0.0;
  double sync_data_ms = 0.0;
  bool sync_deferred = true;

  const auto report_sync_timing = [&]() {
    if (!report_sync) {
      return;
    }
    fprintf(stderr,
            "VIEWPORT_SYNC total=%.2f recalc=%.2f data=%.2f deferred=%d\n",
            (time_dt() - sync_start_time) * 1000.0,
            recalc_ms,
            sync_data_ms,
            int(sync_deferred));
    fflush(stderr);
  };

  /* on session/scene parameter changes, we recreate session entirely */
  const SessionParams session_params = BlenderSync::get_session_params(
      b_engine, b_userpref, *b_scene, background, pixelsize);
  const SceneParams scene_params = BlenderSync::get_scene_params(
      b_userpref, *b_data, *b_scene, background, use_developer_ui);
  const bool session_pause = BlenderSync::get_session_pause(*b_scene, background);

  if (session->params.modified(session_params) || scene->params.modified(scene_params)) {
    free_session();
    create_session();
  }

  ensure_display_driver_if_needed();

  /* increase samples and render time, but never decrease */
  session->set_samples(session_params.samples);
  session->set_interactive_samples(session_params.interactive_samples);
  session->set_time_limit(session_params.time_limit);
  session->set_pause(session_pause);

  /* The capture switch, read once per sync rather than on every redraw: `view_draw` runs several
   * times per frame and an RNA lookup there would be pure overhead. An enum, so `get_enum` - an
   * enum read with `get_int` comes back as its index and never reaches the code that wants it. */
  {
    blender::PointerRNA scene_rna_ptr = RNA_id_pointer_create(&b_scene->id);
    blender::PointerRNA cscene = RNA_pointer_get(&scene_rna_ptr, "cycles");
    playblast_mode_ = get_enum(cscene, "dlss_playblast_mode", 2, 0);
  }

  /* Second, independent source of the interaction state. `view_draw` is the primary one, but this
   * runs on every depsgraph update and keeps the viewport from getting stuck on the reduced budget
   * if a draw-side transition is ever missed.
   *
   * During an offscreen capture the state is the switch, not the mouse: the fast mode is the
   * viewport playing back - moving, on the grid, one input per frame with the history carried
   * across - and the converging one is the viewport once it has settled. */
  const bool fast_capture = b_engine.viewport_offscreen_capture && (playblast_mode_ == 0);
  session->set_interaction_state(b_engine.viewport_offscreen_capture ?
                                     (playblast_mode_ == 0) :
                                     viewport_interaction_active(b_screen, b_rv3d));
  session->set_volume_grid_hold(fast_capture);

  /* The fast capture reads the first input of each frame, and nothing rendered after it reaches
   * the frame - so a budget of one: the render thread is idle by the time the capture has read
   * the frame, the next frame's reset has nothing to cancel and wait for, and the GPU is free for
   * the next frame's input rather than for accumulation no one will see. */
  if (fast_capture) {
    session->set_interactive_samples(1);
  }

  show_viewport_fps = get_show_viewport_fps(*b_scene);

  /* copy recalc flags, outside of mutex so we can decide to do the real
   * synchronization at a later time to not block on running updates */
  {
    const double recalc_start_time = report_sync ? time_dt() : 0.0;
    sync->sync_recalc(b_depsgraph_, b_screen, b_v3d, b_rv3d);
    recalc_ms = report_sync ? (time_dt() - recalc_start_time) * 1000.0 : 0.0;
  }

  /* don't do synchronization if on pause */
  if (session_pause) {
    tag_update();
    report_sync_timing();
    return;
  }

  /* try to acquire mutex. if we don't want to or can't, come back later */
  {
    /* Which of the two refusals dominates decides whether shortening the sync pays off at all.
     * `!ready_to_reset` means the previous result has not been drawn yet - the loop is bounded by
     * the render thread, and a faster sync only waits longer. `!try_lock` means the scene is held,
     * and there every millisecond taken off the sync is a millisecond the next one starts earlier.
     * The two were indistinguishable in the counters, so half the deferred frames were being read
     * as contention that may not exist. */
    const bool not_ready = !session->ready_to_reset();

    /* Waited on rather than tried once. The session thread takes this mutex at the start of every
     * accumulation iteration, and with DLSS there are four of them per pose - on all but the first
     * it holds the lock for microseconds. Giving up on a lock that is about to be free cost a whole
     * frame, and the counters said that was four refusals out of five.
     *
     * The wait is bounded so the UI cannot be dragged down by a genuinely long update: past it the
     * frame is deferred exactly as before, and Blender comes back on the next event loop pass. */
    const std::chrono::milliseconds lock_wait{3};
    const bool not_locked = !not_ready && !session->scene->mutex.try_lock_for(lock_wait);

    if (not_ready || not_locked) {
      static const bool report_defer = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
      if (report_defer) {
        static int no_redraw_count = 0;
        static int stale_texture_count = 0;
        static int not_locked_count = 0;
        static int would_also_block_count = 0;

        if (not_ready) {
          /* `ready_to_reset()` is false for two unrelated reasons, and telling them apart decides
           * where the remaining work is.
           *
           * It reports `did_draw_after_reset_`, which needs BOTH the UI thread to have finished a
           * viewport redraw since the reset AND the texture to have been fresh when it got there.
           * The redraw is where the overlays run; the texture is what the render thread posts. So
           * zero redraws means the frame is waiting on the UI thread - that is an overlay problem.
           * Redraws that happened but found the texture outdated means it is waiting on tracing. */
          if (session->draws_after_reset() == 0) {
            no_redraw_count++;
          }
          else {
            stale_texture_count++;
          }

          /* `not_locked` is only evaluated when the frame was otherwise ready, so the two counters
           * never described competing causes. This asks the question the other branch never got
           * to - untimed, so it cannot lengthen the refusal path the way a 3 ms wait would. */
          if (session->scene->mutex.try_lock()) {
            session->scene->mutex.unlock();
          }
          else {
            would_also_block_count++;
          }
        }
        not_locked_count += not_locked;

        if ((no_redraw_count + stale_texture_count + not_locked_count) % 100 == 0) {
          fprintf(stderr,
                  "SYNC_DEFER no_redraw=%d stale_texture=%d not_locked=%d would_also_block=%d\n",
                  no_redraw_count,
                  stale_texture_count,
                  not_locked_count,
                  would_also_block_count);
          fflush(stderr);
        }
      }
      tag_update();
      report_sync_timing();
      return;
    }
  }

  /* data and camera synchronize */
  b_depsgraph = &b_depsgraph_;

  sync_deferred = false;
  /* Labels the delivery diagnostic: which pose the pixels about to be produced belong to. Set
   * before `sync_data` rather than after, so a delivery that races the end of the sync is labelled
   * with the pose being worked on rather than the previous one. */
  note_accepted_scene_frame(b_scene->r.cfra);
  {
    const double sync_data_start_time = report_sync ? time_dt() : 0.0;
    sync->sync_data(*b_render,
                    *b_depsgraph,
                    b_screen,
                    b_v3d,
                    b_rv3d,
                    width,
                    height,
                    &python_thread_state,
                    session_params.denoise_device);
    sync_data_ms = report_sync ? (time_dt() - sync_data_start_time) * 1000.0 : 0.0;
  }

  /* While a capture owns the view, only the capture's own draw may set the camera: a sync raised
   * from the region on the main thread carries the region's matrices, and taking them would read
   * as a camera move between the capture's frames. */
  const bool region_sync_during_capture = b_engine.viewport_offscreen_capture &&
                                          !b_engine.viewport_offscreen;
  if (b_rv3d && !region_sync_during_capture) {
    const bool viewport_mapping_changed = sync->sync_view(b_v3d, b_rv3d, width, height);
    if (viewport_mapping_changed) {
      session->request_denoiser_history_reset();
      if (display_driver_) {
        display_driver_->request_frame_generation_reset();
      }
    }
    update_frame_generation_state();
  }
  else if (!b_rv3d) {
    sync->sync_camera(*b_render, width, height, "");
  }

  /* get buffer parameters */
  const BufferParams buffer_params = BlenderSync::get_buffer_params(
      b_v3d, b_rv3d, scene->camera, width, height);

  /* reset if needed */
  if (scene->need_reset()) {
    session->reset(session_params, buffer_params);

    /* After session reset, so device is not accessing image data anymore. */
    builtin_images_load();

    /* reset time */
    start_resize_time = 0.0;
  }

  /* This frame's data is now in the session. Recorded only on the path that got here: the early
   * returns above leave the previous value, so a refused sync cannot be mistaken for a synced
   * frame. The published readiness is retired at the same time - a newly synced frame makes any
   * previous completion stale, and this also covers a capture that revisits a frame number. */
  playblast_synced_frame_ = b_scene->r.cfra;
  b_engine.viewport_offscreen_frame = INT_MIN;
  /* And the draw manager need not raise another update for this frame's offscreen draws: the
   * capture clears this when it steps to the next frame. */
  b_engine.viewport_offscreen_synced = true;

  /* unlock */
  session->scene->mutex.unlock();

  report_sync_timing();

  /* Start rendering thread, if it's not running already. Do this
   * after all scene data has been synced at least once. */
  session->start();
}

void BlenderSession::draw(blender::bScreen &b_screen, blender::SpaceImage &space_image)
{
  if (!session || !session->scene) {
    /* Offline render drawing does not force the render engine update, which means it's possible
     * that the Session is not created yet. */
    return;
  }

  const thread_scoped_lock lock(draw_state_.mutex);

  const int pass_index = space_image.iuser.pass;
  if (pass_index != draw_state_.last_pass_index) {
    blender::RenderPass *b_display_pass = RE_engine_pass_by_index_get(
        &b_engine, b_rlay_name.c_str(), pass_index);
    if (!b_display_pass) {
      return;
    }

    Scene *scene = session->scene.get();

    const thread_scoped_timed_lock lock(scene->mutex);

    const Pass *pass = Pass::find(scene->passes, b_display_pass->name);
    if (!pass) {
      return;
    }

    scene->film->set_display_pass(pass->get_type());

    draw_state_.last_pass_index = pass_index;
  }

  if (display_driver_) {
    blender::PointerRNA space_image_rna_ptr = RNA_pointer_create_id_subdata(
        b_screen.id, blender::RNA_SpaceImageEditor, &space_image);
    float zoom[2];
    RNA_float_get_array(&space_image_rna_ptr, "zoom", zoom);
    display_driver_->set_zoom(zoom[0], zoom[1]);
  }

  session->draw();
}

void BlenderSession::view_draw(const int w, const int h)
{
  /* While a capture owns this view, a draw from the region does nothing. The region and the
   * offscreen buffer differ in size, and the session treats every size change as a resize and
   * resets: letting both draw would reset the session on every alternation, and the capture would
   * sample the emptied display between them. The region keeps showing what its viewport retained. */
  const bool capture = b_engine.viewport_offscreen_capture;
  if (capture && !b_engine.viewport_offscreen) {
    return;
  }

  /* pause in redraw in case update is not being called due to final render */
  session->set_pause(BlenderSync::get_session_pause(*b_scene, background));

  /* Update navigating state. */
  const bool dimensions_changed = (width != w || height != h || pixelsize != blender::U.pixelsize);
  const bool is_interacting = capture ? (playblast_mode_ == 0) :
                                        (viewport_interaction_active(b_screen, b_rv3d) ||
                                         dimensions_changed);
  session->set_interaction_state(is_interacting);
  session->set_volume_grid_hold(capture && playblast_mode_ == 0);

  const bool show_fps = get_show_viewport_fps(*b_scene);
  if (show_fps != show_viewport_fps) {
    show_viewport_fps = show_fps;
    /* The status text is only rebuilt when the session reports progress, and a finished render
     * reports none - so toggling this would otherwise leave the previous text on screen. Clearing
     * the last status makes the next build push its result through. */
    last_status.clear();
    update_status_progress();
  }

  /* before drawing, we verify camera and viewport size changes, because
   * we do not get update callbacks for those, we must detect them here */
  if (session->ready_to_reset()) {
    bool reset = false;

    /* If dimensions changed, reset. We need to check pixel size here because
     * it's only valid during drawing, as it can change per window. */
    if (dimensions_changed && b_engine.viewport_offscreen) {
      /* A capture's size is settled by construction, and the debounce below would only make it
       * wait out 0.2 s on every frame. */
      width = w;
      height = h;
      pixelsize = blender::U.pixelsize;
      reset = true;
    }
    else if (dimensions_changed) {
      if (start_resize_time == 0.0) {
        /* don't react immediately to resizes to avoid flickery resizing
         * of the viewport, and some window managers changing the window
         * size temporarily on unminimize */
        start_resize_time = time_dt();
        tag_redraw();
      }
      else if (time_dt() - start_resize_time < 0.2) {
        tag_redraw();
      }
      else {
        width = w;
        height = h;
        pixelsize = blender::U.pixelsize;
        reset = true;
      }
    }

    /* try to acquire mutex. if we can't, come back later */
    if (!session->scene->mutex.try_lock()) {
      tag_update();
    }
    else {
      /* update camera from 3d view */

      const bool viewport_mapping_changed = sync->sync_view(b_v3d, b_rv3d, width, height);
      const bool camera_modified = scene->camera->is_modified();
      if (viewport_mapping_changed) {
        session->request_denoiser_history_reset();
        if (display_driver_) {
          display_driver_->request_frame_generation_reset();
        }
      }
      if (viewport_mapping_changed || camera_modified) {
        update_frame_generation_state();
      }

      if (camera_modified) {
        reset = true;
      }

      session->scene->mutex.unlock();
    }

    /* reset if requested */
    if (reset) {
      const SessionParams session_params = BlenderSync::get_session_params(
          b_engine, b_userpref, *b_scene, background, pixelsize);
      const BufferParams buffer_params = BlenderSync::get_buffer_params(
          b_v3d, b_rv3d, scene->camera, width, height);
      const bool session_pause = BlenderSync::get_session_pause(*b_scene, background);

      if (session_pause == false) {
        session->reset(session_params, buffer_params);
        start_resize_time = 0.0;
      }
    }
  }
  else {
    tag_update();
  }

  /* update status and progress for 3d view draw */
  update_status_progress();

  /* Completion is sampled before the draw, deliberately: the final update can land between the
   * draw and the check, and then a frame whose last pixels are not on the texture yet would be
   * declared complete and read out one draw early. Sampled first, "complete" always refers to
   * data the draw that follows had a chance to show. */
  const bool complete = session->is_render_complete();

  /* draw */
  const bool fresh = session->draw();

  /* Publish, for an offscreen capture, whether the frame it is about to read is there. Both facts
   * are needed: `playblast_synced_frame_` says this frame's data reached the session at all -
   * a refused sync leaves it behind, and then the completion belongs to an older frame - and
   * `fresh` says the draw that just happened put the current pixels on the texture. The fast mode
   * stops there - one input per frame, reconstructed over the history of the frames before, the
   * way the viewport plays back - and so runs at the viewport's own rate; the converging one also
   * wants the accumulation finished. */
  b_engine.viewport_offscreen_sync = playblast_participates();
  if (b_engine.viewport_offscreen_sync && playblast_synced_frame_ == b_scene->r.cfra && fresh &&
      (complete || playblast_mode_ == 0))
  {
    b_engine.viewport_offscreen_frame = b_scene->r.cfra;
  }
}

bool BlenderSession::playblast_participates() const
{
  /* Every branch here is a way for the engine to be unable to report completion. Returning false
   * removes the wait entirely, so the capture reads what is there rather than waiting out a cap on
   * every frame - the failure mode has to be "no wait", never "stuck". */
  if (!b_engine.viewport_offscreen_capture || background || b_v3d == nullptr) {
    return false;
  }
  if (!session || !scene || session->progress.get_error()) {
    return false;
  }

  /* Paused renders never finish, by definition. */
  if (BlenderSync::get_session_pause(*b_scene, background)) {
    return false;
  }

  /* Both modes wait for the accumulation to finish - fast within the moving viewport's budget,
   * converging within the sample count - and with Viewport Samples at zero the converging one
   * never does: the scheduler runs to Max Samples. */
  if (playblast_mode_ != 0 && session->params.samples == INT_MAX) {
    return false;
  }

  return true;
}

void BlenderSession::get_status(string &status, string &substatus)
{
  session->progress.get_status(status, substatus);
}

void BlenderSession::get_progress(double &progress, double &total_time, double &render_time)
{
  session->progress.get_time(total_time, render_time);
  progress = session->progress.get_progress();
}

void BlenderSession::update_bake_progress()
{
  const double progress = session->progress.get_progress();

  if (progress != last_progress) {
    RE_engine_update_progress(&b_engine, (float)progress);
    last_progress = progress;
  }
}

void BlenderSession::update_status_progress()
{
  string timestatus;
  string status;
  string substatus;
  get_status(status, substatus);
  if (background && !substatus.empty()) {
    status += " | " + substatus;
  }

  double progress;
  double total_time;
  double render_time;
  get_progress(progress, total_time, render_time);

  const float mem_used = (float)session->stats.mem_used / 1024.0f / 1024.0f;
  const float mem_peak = (float)session->stats.mem_peak / 1024.0f / 1024.0f;
  if (background) {

    if (progress > 0) {
      const double remaining_time = session->get_estimated_remaining_time();
      if (remaining_time > 0) {
        timestatus = "Remaining: " + time_human_readable_from_seconds(remaining_time) + " | ";
      }
    }

    timestatus += string_printf("Mem: %dM | ", (int)ceilf(mem_used));
  }
  else if (display_driver_ && show_viewport_fps) {
    /* How often the viewport actually receives new rendered pixels. Blender's own playback counter
     * measures the timeline instead, which during a Rendered viewport says nothing about the
     * renderer - the timeline can run ahead while the same image is redrawn. Shown whether or not an
     * animation is playing, since the question "how many fps does full shading give me" applies
     * just as much to orbiting a still frame.
     *
     * Reported through the engine status rather than drawn separately, so that Blender lays it out
     * with the rest of the viewport text and it cannot land on top of another overlay line. */
    const double delivered_fps = display_driver_->get_delivered_fps();
    if (delivered_fps > 0.0) {
      timestatus += string_printf("Viewport: %.1f fps | ", delivered_fps);
    }
  }

  const double current_time = time_dt();
  /* When rendering in a window, redraw the status at least once per second to keep things
   * up to date. For headless rendering, only report when something significant changes to
   * keep the console output readable. */
  if (status != last_status || (!headless && (current_time - last_status_time) > 1.0)) {
    RE_engine_update_stats(&b_engine, "", (timestatus + status).c_str());
    RE_engine_update_memory_stats(&b_engine, mem_used, mem_peak);
    last_status = status;
    last_status_time = current_time;
  }
  if (progress != last_progress) {
    RE_engine_update_progress(&b_engine, (float)progress);
    last_progress = progress;
  }

  check_and_report_session_error();
}

bool BlenderSession::check_and_report_session_error()
{
  if (!session->progress.get_error()) {
    return false;
  }

  const string error = session->progress.get_error_message();
  if (error != last_error) {
    /* TODO(sergey): Currently C++ RNA API doesn't let us to use mnemonic name for the variable.
     * Would be nice to have this figured out.
     *
     * For until then, 1 << 5 means RPT_ERROR. */
    RE_engine_report(&b_engine, 1 << 5, error.c_str());
    RE_engine_set_error_message(&b_engine, error.c_str());
    last_error = error;
  }

  return true;
}

void BlenderSession::tag_update()
{
  /* tell blender that we want to get another update callback */
  b_engine.flag |= blender::RE_ENGINE_DO_UPDATE;
}

void BlenderSession::tag_redraw()
{
  if (background) {
    /* update stats and progress, only for background here because
     * in 3d view we do it in draw for thread safety reasons */
    update_status_progress();

    /* offline render, redraw if timeout passed */
    if (time_dt() - last_redraw_time > 1.0) {
      b_engine.flag |= blender::RE_ENGINE_DO_DRAW;
      last_redraw_time = time_dt();
    }
  }
  else {
    /* tell blender that we want to redraw */
    b_engine.flag |= blender::RE_ENGINE_DO_DRAW;
  }
}

void BlenderSession::test_cancel()
{
  /* test if we need to cancel rendering */
  if (background) {
    if (RE_engine_test_break(&b_engine)) {
      session->progress.set_cancel("Cancelled");
    }
  }
}

void BlenderSession::free_blender_memory_if_possible()
{
  if (!background) {
    /* During interactive render we can not free anything: attempts to save
     * memory would cause things to be allocated and evaluated for every
     * updated sample.
     */
    return;
  }
  RE_engine_free_blender_memory(&b_engine);
  /* The evaluated scene is owned by the render engine and becomes invalid above. Keep the
   * pointer state truthful so late frame-finish hooks cannot query RNA through freed memory. */
  b_scene = nullptr;
}

void BlenderSession::ensure_display_driver_if_needed()
{
  if (display_driver_) {
    /* Driver is already created. */
    return;
  }

  if (headless) {
    /* No display needed for headless. */
    return;
  }

  if ((b_engine.flag & blender::RE_ENGINE_PREVIEW) != 0) {
    /* TODO(sergey): Investigate whether DisplayDriver can be used for the preview as well. */
    return;
  }

  unique_ptr<BlenderDisplayDriver> display_driver = make_unique<BlenderDisplayDriver>(
      b_engine, *b_scene, b_rv3d, background);
  display_driver_ = display_driver.get();
  session->set_display_driver(std::move(display_driver));
}

CCL_NAMESPACE_END
