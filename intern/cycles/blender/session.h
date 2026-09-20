/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <climits>

#include "device/device.h"

#include "scene/scene.h"
#include "session/session.h"

#include "util/unique_ptr.h"
#include "util/vector.h"

namespace blender {
struct bScreen;
struct Depsgraph;
struct Main;
struct Object;
struct RegionView3D;
struct RenderData;
struct RenderEngine;
struct Scene;
struct SpaceImage;
struct UserDef;
struct View3D;
}  // namespace blender

CCL_NAMESPACE_BEGIN

class BlenderDisplayDriver;
class BlenderSync;
class ImageMetaData;
class Scene;
class Session;

class BlenderSession {
 public:
  BlenderSession(blender::RenderEngine &b_engine,
                 blender::UserDef &b_userpref,
                 blender::Main &b_data,
                 bool preview_osl);

  BlenderSession(blender::RenderEngine &b_engine,
                 blender::UserDef &b_userpref,
                 blender::Main &b_data,
                 blender::bScreen *b_screen,
                 blender::View3D *b_v3d,
                 blender::RegionView3D *b_rv3d,
                 const int width,
                 int height);

  ~BlenderSession();

  /* session */
  void create_session();
  void free_session();

  void reset_session(blender::Main &b_data, blender::Depsgraph &b_depsgraph);

  /* offline render */
  void render(blender::Depsgraph &b_depsgraph);

  void render_frame_finish();

  void bake(blender::Depsgraph &b_depsgraph_,
            blender::Object &b_object,
            const string &bake_type,
            const int bake_filter,
            const int bake_width,
            const int bake_height);

  void full_buffer_written(string_view filename);
  /* interactive updates */
  void synchronize(blender::Depsgraph &b_depsgraph);

  /* drawing */
  void draw(blender::bScreen &b_screen, blender::SpaceImage &space_image);
  void view_draw(const int w, const int h);
  void tag_redraw();
  void tag_update();
  void get_status(string &status, string &substatus);
  void get_progress(double &progress, double &total_time, double &render_time);
  void test_cancel();
  void update_status_progress();
  void update_bake_progress();

  bool background;
  unique_ptr<Session> session;
  Scene *scene;
  unique_ptr<BlenderSync> sync;
  double last_redraw_time;

  blender::RenderEngine &b_engine;
  blender::UserDef &b_userpref;
  blender::Main *b_data;
  blender::RenderData *b_render;
  blender::Depsgraph *b_depsgraph;
  /* NOTE: Blender's scene might become invalid after call
   * #free_blender_memory_if_possible(). */
  blender::Scene *b_scene;
  blender::bScreen *b_screen;
  blender::View3D *b_v3d;
  blender::RegionView3D *b_rv3d;
  string b_rlay_name;
  string b_rview_name;

  /* Offscreen capture - a viewport render or playblast reading this session's view back from an
   * offscreen buffer. The capture waits for the engine to say the frame is there, and these are
   * what saying so takes.
   *
   * `playblast_synced_frame_` is the scene frame whose data actually reached the session, which
   * is not the frame Blender is on: `synchronize()` is allowed to refuse, and then the completion
   * in hand belongs to an older frame. `playblast_mode_` is the switch: 0 waits for the first
   * result after the sync - one accumulation through the reconstruction, the picture the viewport
   * shows while the camera moves; 1 waits for the accumulation to finish, the picture it shows
   * once the camera stops. */
  int playblast_synced_frame_ = INT_MIN;
  int playblast_mode_ = 0;


  /* Whether waiting for this session is meaningful right now. False removes the wait entirely,
   * and every condition in it is a way the engine could fail to report completion - the failure
   * mode has to be "no wait", never "stuck". */
  bool playblast_participates() const;

  string last_status;
  string last_error;
  double last_progress;
  double last_status_time;

  /* Whether to report the viewport delivery rate in the status line. Read from the scene on the
   * main thread, since the status is also built from the render thread. */
  bool show_viewport_fps = false;

  int width, height;
  float pixelsize;
  bool preview_osl;
  double start_resize_time;

  void *python_thread_state;

  bool use_developer_ui;

  /* Global state which is common for all render sessions created from Blender.
   * Usually denotes command line arguments.
   */
  static DeviceTypeMask device_override;

  /* Blender is running from the command line, no windows are shown and some
   * extra render optimization is possible (possible to free draw-only data and
   * so on.
   */
  static bool headless;

  static bool print_render_stats;

 protected:
  void stamp_view_layer_metadata(Scene *scene, const string &view_layer_name, bool dlss_requested);

  /* Check whether session error happened.
   * If so, it is reported to the render engine and true is returned.
   * Otherwise false is returned. */
  bool check_and_report_session_error();

  void builtin_images_load();

  /* Is used after each render layer synchronization is done with the goal
   * of freeing render engine data which is held from Blender side (for
   * example, dependency graph).
   */
  void free_blender_memory_if_possible();

  void ensure_display_driver_if_needed();
  void update_frame_generation_state();

  bool use_dlss_animation_persistence() const;
  bool dlss_history_needs_reset(const BufferParams &buffer_params,
                                const string &layer,
                                const string &view) const;
  void dlss_history_commit(const BufferParams &buffer_params,
                           const string &layer,
                           const string &view);
  bool has_dlss_reset_marker() const;
  bool dlss_runtime_failed_ = false;
  bool dlss_animation_persistence_active_ = false;

  struct {
    thread_mutex mutex;
    int last_pass_index = -1;
  } draw_state_;

  /* NOTE: The BlenderSession references the display driver. */
  BlenderDisplayDriver *display_driver_ = nullptr;

  struct {
    bool valid = false;
    bool enabled = false;
    int frame = 0;
    blender::Object *camera = nullptr;
    float persmat[4][4] = {};
    float persinv[4][4] = {};
    string view_transform;
    string look;
    float exposure = 0.0f;
    float gamma = 1.0f;
  } frame_generation_history_;

  vector<string> full_buffer_files_;

  int bake_id = 0;

  struct {
    bool valid = false;
    int frame = 0;
    blender::Object *camera = nullptr;
    string layer;
    string view;
    int width = 0;
    int height = 0;
    int full_x = 0;
    int full_y = 0;
    int window_x = 0;
    int window_y = 0;
    int window_width = 0;
    int window_height = 0;
    float upscale_factor = 0.0f;
    string device_id;
  } dlss_history_;
};

CCL_NAMESPACE_END
