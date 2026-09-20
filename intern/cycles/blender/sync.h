/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "DNA_userdef_types.h"
#include "RNA_types.hh"

#include "blender/id_map.h"
#include "blender/util.h"
#include "blender/viewport.h"

#include "scene/scene.h"
#include "session/session.h"

#include "util/map.h"
#include "util/set.h"
#include "util/vector.h"

#include <utility>

namespace blender {
struct DEGObjectIterData;
struct DupliObject;
struct MeshSequenceCacheModifier;
}  // namespace blender

CCL_NAMESPACE_BEGIN

class Background;
class BlenderObjectCulling;
class BlenderViewportParameters;
class Camera;
class Film;
class Hair;
class Light;
class Mesh;
class Object;
class ParticleSystem;
class Scene;
class Shader;
class ShaderGraph;
class TaskPool;

class BlenderSync {
 public:
  BlenderSync(blender::RenderEngine &b_engine,
              blender::Main &b_data,
              blender::Scene &b_scene,
              Scene *scene,
              bool preview,
              bool use_developer_ui,
              Progress &progress);
  ~BlenderSync();

  void reset(blender::Main &b_data, blender::Scene &b_scene);

  void tag_update();

  void set_bake_target(blender::Object &b_object);

  /* sync */
  void sync_recalc(blender::Depsgraph &b_depsgraph,
                   blender::bScreen *b_screen,
                   blender::View3D *b_v3d,
                   blender::RegionView3D *b_rv3d);
  void sync_data(blender::RenderData &b_render,
                 blender::Depsgraph &b_depsgraph,
                 blender::bScreen *b_screen,
                 blender::View3D *b_v3d,
                 blender::RegionView3D *b_rv3d,
                 const int width,
                 const int height,
                 void **python_thread_state,
                 const DeviceInfo &denoise_device_info);
  void sync_view_layer(blender::ViewLayer &b_view_layer);
  void sync_render_passes(blender::RenderLayer &b_rlay, blender::ViewLayer &b_view_layer);
  void sync_integrator(blender::ViewLayer &b_view_layer,
                       bool background,
                       const DeviceInfo &denoise_device_info);
  void sync_scene_attributes();
  void sync_camera(const blender::RenderData &b_render,
                   const int width,
                   const int height,
                   const char *viewname);
  bool sync_view(blender::View3D *b_v3d,
                 blender::RegionView3D *b_rv3d,
                 const int width,
                 const int height);
  int get_layer_samples()
  {
    return view_layer.samples;
  }
  int get_layer_bound_samples()
  {
    return view_layer.bound_samples;
  }

  /* Early data free. */
  void free_data_after_sync(blender::Depsgraph &b_depsgraph);

  /* get parameters */
  static SceneParams get_scene_params(blender::UserDef &b_preferences,
                                      blender::Main &b_data,
                                      blender::Scene &b_scene,
                                      const bool background,
                                      const bool use_developer_ui);
  static SessionParams get_session_params(blender::RenderEngine &b_engine,
                                          blender::UserDef &b_preferences,
                                          blender::Scene &b_scene,
                                          bool background,
                                          float pixelsize);
  static bool get_session_pause(blender::Scene &b_scene, bool background);
  static BufferParams get_buffer_params(blender::View3D *b_v3d,
                                        blender::RegionView3D *b_rv3d,
                                        Camera *cam,
                                        const int width,
                                        const int height);

  static DenoiseParams get_denoise_params(blender::Scene &b_scene,
                                          blender::ViewLayer *b_view_layer,
                                          bool background,
                                          const DeviceInfo &denoise_device,
                                          bool is_animation = false);

 private:
  /* sync */
  void sync_lights(blender::Depsgraph &b_depsgraph, bool update_all, bool update_time);
  void sync_materials(blender::Depsgraph &b_depsgraph, bool update_all, bool update_time);
  void sync_objects(blender::Depsgraph &b_depsgraph,
                    blender::bScreen *b_screen,
                    blender::View3D *b_v3d,
                    const float motion_time = 0.0f);
  void sync_objects_and_motion(blender::RenderData &b_render,
                               blender::Depsgraph &b_depsgraph,
                               blender::bScreen *b_screen,
                               blender::View3D *b_v3d,
                               blender::RegionView3D *b_rv3d,
                               const int width,
                               const int height,
                               void **python_thread_state);
  void sync_film(blender::ViewLayer &b_view_layer,
                 blender::bScreen *b_screen,
                 blender::View3D *b_v3d);
  void sync_view();

  /* Shader */
  array<Node *> find_used_shaders(blender::Object &b_ob);
  void sync_world(blender::Depsgraph &b_depsgraph,
                  blender::bScreen *b_screen,
                  blender::View3D *b_v3d,
                  bool update_all,
                  bool update_time);
  void sync_shaders(blender::Depsgraph &b_depsgraph,
                    blender::bScreen *b_screen,
                    blender::View3D *b_v3d,
                    bool update_all,
                    bool update_time);
  void sync_nodes(Shader *shader, blender::bNodeTree &b_ntree);

  bool scene_attr_needs_recalc(Shader *shader, blender::Depsgraph &b_depsgraph);
  void resolve_view_layer_attributes(Shader *shader,
                                     ShaderGraph *graph,
                                     blender::Depsgraph &b_depsgraph);

  /* Object */
  Object *sync_object(blender::ViewLayer &b_view_layer,
                      blender::Object &b_ob,
                      blender::DEGObjectIterData &b_deg_iter_data,
                      const float motion_time,
                      bool use_particle_hair,
                      bool show_lights,
                      BlenderObjectCulling &culling,
                      TaskPool *geom_task_pool);
  void sync_object_motion_init(blender::Object &b_parent, blender::Object &b_ob, Object *object);

  void sync_procedural(blender::Object &b_ob,
                       blender::MeshSequenceCacheModifier &b_mesh_cache,
                       bool has_subdivision);

  bool sync_object_attributes(blender::Object &b_ob,
                              blender::DEGObjectIterData &b_deg_iter_data,
                              Object *object);

  /* Volume */
  void sync_volume(BObjectInfo &b_ob_info, Volume *volume);

  /* Mesh */
  void sync_mesh(BObjectInfo &b_ob_info, Mesh *mesh);
  void sync_mesh_motion(BObjectInfo &b_ob_info, Mesh *mesh, int motion_step);

  /* Hair */
  void sync_hair(BObjectInfo &b_ob_info, Hair *hair);
  void sync_hair_motion(BObjectInfo &b_ob_info, Hair *hair, int motion_step);
  void sync_hair(Hair *hair, BObjectInfo &b_ob_info, bool motion, const int motion_step = 0);
  void sync_particle_hair(Hair *hair,
                          const blender::Mesh &b_mesh,
                          BObjectInfo &b_ob_info,
                          bool motion,
                          const int motion_step = 0);
  bool object_has_particle_hair(blender::Object *b_ob);
  bool compute_object_has_particle_hair(blender::Object *b_ob);

  /* Point Cloud */
  void sync_pointcloud(PointCloud *pointcloud, BObjectInfo &b_ob_info);
  void sync_pointcloud_motion(PointCloud *pointcloud,
                              BObjectInfo &b_ob_info,
                              const int motion_step = 0);

  /* Camera */
  void sync_camera_motion(const blender::RenderData &b_render,
                          blender::Object *b_ob,
                          const int width,
                          const int height,
                          const float motion_time);

  /* Geometry */
  Geometry *sync_geometry(BObjectInfo &b_ob_info,
                          bool object_updated,
                          bool use_particle_hair,
                          TaskPool *task_pool);

  void sync_geometry_motion(BObjectInfo &b_ob_info,
                            Object *object,
                            const float motion_time,
                            bool use_particle_hair,
                            TaskPool *task_pool);

  /* Light */
  Geometry *create_light(BObjectInfo &b_ob_info);
  void sync_light(BObjectInfo &b_ob_info, Light *light);
  void sync_background_light(blender::bScreen *b_screen, blender::View3D *b_v3d);

  /* Particles */
  bool sync_dupli_particle(blender::Object &b_parent,
                           blender::DEGObjectIterData &b_deg_iter_data,
                           blender::Object &b_ob,
                           Object *object);

  /* Images. */
  void sync_images();

  /* util */
  void find_shader(const blender::ID *id, array<Node *> &used_shaders, Shader *default_shader);
  bool BKE_object_is_modified(blender::Object &b_ob);
  bool compute_object_is_modified(blender::Object &b_ob);
  bool object_is_geometry(BObjectInfo &b_ob_info);
  bool object_can_have_geometry(blender::Object &b_ob);
  bool object_is_light(blender::Object &b_ob);
  bool object_is_camera(blender::Object &b_ob);

  blender::Object *get_camera_object(blender::View3D *b_v3d, blender::RegionView3D *b_rv3d);
  blender::Object *get_dicing_camera_object(blender::View3D *b_v3d, blender::RegionView3D *b_rv3d);

  /* variables */
  blender::RenderEngine *b_engine;
  blender::Main *b_data;
  blender::Scene *b_scene;
  blender::Object *b_bake_target;

  enum ShaderFlags { SHADER_WITH_LAYER_ATTRS };

  id_map<const void *, Shader, ShaderFlags> shader_map;
  /* To keep track of the AOVs in consecutive view layers that are rendered, this is the old data
   * for comparing. */
  blender::Vector<std::pair<std::string, int>> shader_view_layer_aovs;
  id_map<ObjectKey, Object> object_map;
  id_map<void *, Procedural> procedural_map;
  id_map<GeometryKey, Geometry> geometry_map;
  id_map<ParticleSystemKey, ParticleSystem> particle_system_map;
  set<Geometry *> geometry_synced;
  set<Geometry *> geometry_motion_synced;
  set<Geometry *> geometry_motion_attribute_synced;
  /** Remember which geometries come from which objects to be able to sync them after changes. */
  map<void *, set<blender::ID *>> instance_geometries_by_object;

  /* Per-object values that cost a string lookup through RNA, or a walk up the parent chain, to
   * read. Instanced geometry hands the same source object back thousands of times in one pass -
   * this scene evaluates 33858 objects out of 837 real ones - so they are read once per object and
   * reused for its instances. Cleared at the start of every object sync, so a change to any of them
   * is picked up on the next frame like any other. */
  struct CachedObjectProperties {
    float ao_distance;
    bool is_caustics_caster;
    bool is_caustics_receiver;
    ustring asset_name;
  };
  /* Hashed, not ordered: `map` in Cycles is `std::map`, and a tree lookup among a few hundred
   * entries costs about ten dependent memory hops - times 33858 instances that measured 11 ms,
   * more than the RNA reads the cache was meant to save. */
  unordered_map<const blender::Object *, CachedObjectProperties> object_properties_cache;

  /* The effective AO distance falls back to the parent's when the object's own is zero, so it
   * belongs to the (object, parent) pair rather than to the object - the same geometry instanced
   * under two parents would otherwise inherit whichever one was synced first. */
  unordered_map<const blender::Object *, float> parent_ao_distance_cache;

  /* Answers that come from walking an object's modifier stack. They describe the source object, but
   * an instanced one reaches these checks once per instance - 33858 stack walks per frame on this
   * scene where 837 would do. Both depend only on the object and on `preview`, which does not
   * change within a pass. */
  /* Diagnostics only: how much of the instance list actually moves between frames. */
  int sync_stats_objects_seen = 0;
  int sync_stats_transform_changed = 0;
  int sync_stats_geometry_dirty = 0;
  /* Corners across the dirty geometries, and how many of them were repacked on the main thread
   * rather than on the pool. Together they say whether the pool wait is mesh repacking at all. */
  int sync_stats_geometry_corners = 0;
  int sync_stats_geometry_synced_inline = 0;
  /* Diagnostics only: how the instances divide among their instancers, and how much of that list a
   * per-instancer skip could reach. A group counts as moved when any one of its instances moved -
   * so `instances_in_moved` is the work such a skip would still have to do. */
  int sync_stats_groups = 0;
  int sync_stats_groups_moved = 0;
  int sync_stats_instances_in_moved_groups = 0;
  int sync_stats_largest_group = 0;
  int sync_stats_instances_in_groups = 0;
  int sync_stats_groups_clean = 0;
  int sync_stats_instances_in_clean_groups = 0;
  /* How many meshes the warm-up below actually took. Reported so the gates can be seen to be doing
   * something: a zero here on a frame where geometry is dirty means every candidate was filtered
   * out, and the wait after the walk stays what it was. */
  int sync_stats_normals_warmed = 0;
  /* And why the rest of the candidates were not taken. A warm-up that never fires is otherwise
   * indistinguishable from one that fires and does not help. */
  int sync_stats_warm_candidates = 0;
  int sync_stats_warm_gone = 0;
  int sync_stats_warm_unevaluated = 0;
  int sync_stats_warm_invisible = 0;
  int sync_stats_warm_subdivided = 0;
  int sync_stats_warm_not_mesh = 0;
  /* The part of the walk spent on instances the clean-group fast path did not claim, and how many
   * of them there were. Splits the walk into "syncing the few" and "walking past the many". */
  double sync_stats_full_path_ms = 0.0;
  int sync_stats_full_path_count = 0;

  /* Starts the expensive half of copying a dirty mesh before the walk rather than after it. See the
   * definition for why this is a scheduling change and not a behaviour one. */
  void warm_dirty_geometry_normals(blender::Depsgraph &b_depsgraph,
                                   TaskPool &geom_task_pool,
                                   bool motion);

  /* Whether the depsgraph reported no change for this instancer, so every instance it produces
   * would be identical to the previous frame. Reads Cycles' own recalc sets rather than
   * `ID.recalc`. */
  bool instancer_looks_clean(blender::Object &b_parent);

  /* Skipping unchanged instances
   *
   * The object walk visits every instance every frame - 33858 of them on this scene, of which 74
   * move. What the walk costs is almost entirely the body: the iterator alone measures 6.4 ms
   * against 37.6 for the whole loop. So the work worth removing is per-instance work in Cycles,
   * and the unit it can be removed in is the instancer: instances arrive grouped by the object
   * that produced them, and a group whose instancer the depsgraph did not touch produces exactly
   * what it produced last frame.
   *
   * Two things are checked before a group is skipped. The instancer must be clean by Cycles' own
   * recalc sets, and so must every source object the group instances - a material or visibility
   * change on the source never reaches the instancer's key. Then each instance is still compared
   * against what it was, field by field, which costs a 136-byte memcmp instead of a sync_object.
   * That comparison is what makes the skip safe against a change the recalc sets did not report:
   * the moment one instance disagrees, the rest of the group takes the full path and the entry is
   * dropped, so the next frame rebuilds it. */
  /* Everything about an instance that decides what it syncs to. Compared as bytes, so it is filled
   * through memset first - a padding hole would compare as a spurious change. */
  struct DupliFingerprint {
    const blender::Object *ob;
    const blender::ID *ob_data;
    const void *particle_system;
    float mat[4][4];
    float orco[3];
    float uv[2];
    unsigned int random_id;
    int persistent_id[8];
    short type;
    char no_draw;
  };

  struct CachedDupli {
    DupliFingerprint fingerprint;
    /* What this step produced, kept with the step rather than in one flat list: when a group turns
     * out to have changed halfway through, the steps already skipped still have to be marked used,
     * and that needs to know which objects belong to which step. Two of them where particle hair
     * adds a second object. */
    Object *objects[2];
    int num_objects;
  };

  struct InstancerCacheEntry {
    /* One per iterator step, in walk order, to compare this frame's instances against. */
    vector<CachedDupli> duplis;
    /* Deduplicated, so a skip can mark them used without walking the objects again. */
    vector<Geometry *> geometries;
    vector<ParticleSystem *> particle_systems;
    /* The source objects, to test for changes the instancer's own key never sees. */
    vector<blender::Object *> sources;
    /* Which object data each source contributed, replayed into `instance_geometries_by_object`.
     * Kept as edges rather than a slice of that map: it is keyed by source object, and the same
     * source can be instanced by more than one instancer. */
    vector<std::pair<void *, blender::ID *>> geometry_edges;
    /* Digest of everything `sync_object` reads off the instancer and its sources before the sync
     * gate: holdout, indirect-only, ray visibility, shadow catcher, display type. None of it leaves
     * a trace in the recalc sets - the RNA callbacks behind those properties raise
     * ID_RECALC_SYNC_TO_EVAL, which the depsgraph also raises whenever it refreshes an evaluated
     * copy, so the flag cannot tell a toggle from routine work. The values can. */
    size_t visibility_signature = 0;
    bool seen_this_frame = false;
  };

  unordered_map<const blender::Object *, InstancerCacheEntry> instancer_cache;
  /* Set while a group is being walked in full, so the collectors know where to record. */
  InstancerCacheEntry *instancer_recording = nullptr;
  /* Everything outside the instancers that would change what an instance syncs to. When this
   * differs from last frame no group can be trusted, whatever the depsgraph reported. */
  size_t instancer_cache_globals = 0;
  bool use_instancer_cache = false;

  size_t instancer_cache_signature(blender::View3D *b_v3d, bool show_lights);
  size_t instancer_visibility_signature(blender::ViewLayer &b_view_layer,
                                        blender::Object &b_parent,
                                        const vector<blender::Object *> &sources);
  bool instancer_group_is_clean(blender::ViewLayer &b_view_layer,
                                blender::Object &b_parent,
                                const InstancerCacheEntry &entry);
  void instancer_replay_shared(const InstancerCacheEntry &entry);
  static void instancer_fill_fingerprint(const blender::DupliObject &dupli,
                                         DupliFingerprint &r_fingerprint);

  unordered_map<const blender::Object *, bool> object_is_modified_cache;
  unordered_map<const blender::Object *, bool> object_has_particle_hair_cache;

  /* Objects whose geometry `sync_recalc` found dirty, so that the expensive part of copying them
   * can be started before the walk instead of after it.
   *
   * Session identifiers rather than pointers, and accumulated rather than replaced. Both follow
   * from the same measurement: Cycles' own recalc flags survive a refused frame, so the geometry
   * repacked on an accepted frame was usually marked dirty during an earlier refused one. A list
   * that only held the last pass therefore fired on about one accepted frame in four. Carrying
   * pointers across those frames is not an option - the depsgraph evaluation in between frees the
   * evaluated data - whereas an identifier resolves to whatever is current, or to nothing if the
   * object is gone. */
  set<uint32_t> warm_normals_candidates;

  /* Which attributes a geometry's shaders ask for. Collecting them walks every shader of the
   * geometry, and `sync_object_attributes` needs them once per instance - the answer is a property
   * of the geometry, not of the instance. Cleared alongside the properties above. */
  unordered_map<const Geometry *, AttributeRequestSet> geometry_attribute_requests_cache;
  set<float> motion_times;
  void *world_map;
  bool world_recalc;
  BlenderViewportParameters viewport_parameters;

  Scene *scene;
  bool preview;
  bool use_adaptive_subdivision = false;
  bool use_developer_ui;

  CurveShapeType curve_shape = CURVE_RIBBON;

  float dicing_rate;
  int max_subdivisions;

  struct RenderLayerInfo {
    string name;
    blender::Material *material_override = nullptr;
    blender::World *world_override = nullptr;
    bool use_background_shader = true;
    bool use_surfaces = true;
    bool use_hair = true;
    bool use_volumes = true;
    bool use_motion_blur = true;
    int samples = 0;
    bool bound_samples = false;
  } view_layer;

  Progress &progress;

  /* Indicates that `sync_recalc()` detected changes in the scene.
   * If this flag is false then the data is considered to be up-to-date and will not be
   * synchronized at all. */
  bool has_updates_ = true;

  float frame_last_synced = 0;
};

CCL_NAMESPACE_END
