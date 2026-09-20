/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "graph/node.h"

/* included as Object::set_particle_system defined through NODE_SOCKET_API does
 * not select the right Node::set overload as it does not know that ParticleSystem
 * is a Node */
#include "scene/geometry.h"
#include "scene/particles.h"
#include "scene/scene.h"

#include "util/array.h"
#include "util/boundbox.h"
#include "util/param.h"
#include "util/transform.h"
#include "util/types.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceScene;
class Geometry;
class ParticleSystem;
class Progress;
class Scene;
struct Transform;
struct UpdateObjectTransformState;
class ObjectManager;

/* Object */

class Object : public Node {
 public:
  NODE_DECLARE

  NODE_SOCKET_API(Geometry *, geometry)
  /* Use base API because we need custom setter for tfm. */
  NODE_SOCKET_API_BASE(Transform, tfm, "tfm")
  BoundBox bounds;
  /* Whether `bounds` still describes this object. Without motion blur the bounds are just the
   * geometry's bounds through `tfm`, so only a changed transform or changed geometry can move
   * them - and on a scene of 33858 instances where 73 move, recomputing all of them costs about
   * 2 ms per scene update. It cannot be read off `is_modified()` at the point the bounds are
   * computed: the object manager runs before the geometry manager and clears those flags on its
   * way out, so the answer is carried here instead. */
  bool bounds_need_update = true;
  NODE_SOCKET_API(uint, random_id)
  NODE_SOCKET_API(int, pass_id)
  NODE_SOCKET_API(float3, color)
  NODE_SOCKET_API(float, alpha)
  NODE_SOCKET_API(ustring, asset_name)
  vector<ParamValue> attributes;
  NODE_SOCKET_API(uint, visibility)
  NODE_SOCKET_API_ARRAY(array<Transform>, motion)
  NODE_SOCKET_API(bool, hide_on_missing_motion)
  NODE_SOCKET_API(bool, use_holdout)
  NODE_SOCKET_API(bool, is_shadow_catcher)
  NODE_SOCKET_API(float, shadow_terminator_shading_offset)
  NODE_SOCKET_API(float, shadow_terminator_geometry_offset)

  NODE_SOCKET_API(bool, is_caustics_caster)
  NODE_SOCKET_API(bool, is_caustics_receiver)

  NODE_SOCKET_API(bool, is_bake_target)

  NODE_SOCKET_API(float3, dupli_generated)
  NODE_SOCKET_API(float2, dupli_uv)

  NODE_SOCKET_API(ParticleSystem *, particle_system);
  NODE_SOCKET_API(int, particle_index);

  NODE_SOCKET_API(float, ao_distance)

  NODE_SOCKET_API(ustring, lightgroup)
  NODE_SOCKET_API(uint, receiver_light_set)
  NODE_SOCKET_API(uint64_t, light_set_membership)
  NODE_SOCKET_API(uint, blocker_shadow_set)
  NODE_SOCKET_API(uint64_t, shadow_set_membership)

  /* Set during device update. */
  bool intersects_volume;

  /* Specifies the position of the object in scene->objects and
   * in the device vectors. Gets set in device_update. */
  int index;

  Object();
  ~Object() override;

  void tag_update(Scene *scene);

  void compute_bounds(bool motion_blur);
  void apply_transform(bool apply_to_motion);

  /* Convert between normalized -1..1 motion time and index
   * in the motion array. */
  bool use_motion() const;
  float motion_time(const int step) const;
  int motion_step(const float time) const;
  void update_motion();

  /* Maximum number of motion steps supported (due to Embree). */
  static const uint MAX_MOTION_STEPS = 129;

  /* Check whether object is traceable and it worth adding it to
   * kernel scene.
   */
  bool is_traceable() const;

  /* Combine object's visibility with all possible internal run-time
   * determined flags which denotes trace-time visibility.
   */
  uint visibility_for_tracing() const;

  /* Returns the index that is used in the kernel for this object. */
  int get_device_index() const;

  /* Compute step size from attributes, shaders, transforms. */
  float compute_volume_step_size(Progress &progress) const;

  /* Check whether this object can be used as light-emissive. */
  bool usable_as_light() const;

  /* Check whether the object participates in light or shadow linking, either as a receiver/blocker
   * or emitter. */
  bool has_light_linking() const;
  bool has_shadow_linking() const;

  /* Transform of some object types need to be modified to prevent render issues. */
  void adjust_volume_tfm(Transform &tfm);
  void set_tfm(Transform tfm);
  bool tfm_equals(Transform tfm);
  void set_motion_tfm(Transform tfm, const int step_index);

 protected:
  /* Reference to the attribute map with object attributes,
   * or 0 if none. Set in update_svm_attributes. */
  size_t attr_map_offset;

  friend class ObjectManager;
  friend class GeometryManager;
};

/* Object Manager */

class ObjectManager {
  uint32_t update_flags;

  /* Raised when the interactive motion pass advanced the transform history, so the device copy of
   * `object_motion_pass` still describes the movement of the frame that was just rendered and has
   * to be settled back to zero before the next one.
   *
   * Deliberately kept out of `update_flags`: raising a manager flag for this used to cost a second
   * full `Scene::device_update` per frame, because `tag_update` cascades into the geometry and
   * light managers. `device_update_flags` also clears `update_flags`, and this one has to survive
   * until the narrow flush actually reached the device. */
  bool motion_history_pending = false;

  /* The per-object offset arrays describe where each object's geometry sits, so they only go stale
   * when something tagged this manager. They are uploaded outside `device_update`, past the point
   * where `update_flags` has already been cleared, so the need is recorded separately. */
  bool offsets_need_update = true;

 public:
  /* Called by the geometry manager when a geometry's primitive offset actually moved - the only
   * thing that can invalidate `object_prim_offset` apart from the set of objects changing. */
  void tag_prim_offsets_modified()
  {
    offsets_need_update = true;
  }

  enum : uint32_t {
    PARTICLE_MODIFIED = (1 << 0),
    GEOMETRY_MANAGER = (1 << 1),
    MOTION_BLUR_MODIFIED = (1 << 2),
    OBJECT_ADDED = (1 << 3),
    OBJECT_REMOVED = (1 << 4),
    OBJECT_MODIFIED = (1 << 5),
    HOLDOUT_MODIFIED = (1 << 6),
    TRANSFORM_MODIFIED = (1 << 7),
    VISIBILITY_MODIFIED = (1 << 8),

    /* tag everything in the manager for an update */
    UPDATE_ALL = ~0u,

    UPDATE_NONE = 0u,
  };

  bool need_flags_update;

  ObjectManager();
  ~ObjectManager();

  void update_interactive_motion(Scene *scene);

  /* True when the transform history advanced and the device still holds the previous frame's
   * motion. */
  bool need_motion_history_flush() const;

  /* Settles the advanced history on the device without going through the manager pipeline: only
   * `object_motion_pass` is rewritten, leaving objects, flags, bounds, the scene BVH and the light
   * tree untouched. Returns false when the resident state is not in a shape this can patch, in
   * which case the caller has to fall back to a regular update. */
  bool device_update_motion_history(DeviceScene *dscene, Scene *scene);

  void device_update(Device *device, DeviceScene *dscene, Scene *scene, Progress &progress);
  void device_update_transforms(DeviceScene *dscene, Scene *scene, Progress &progress);
  void device_update_prim_offsets(Device *device, DeviceScene *dscene, Scene *scene);

  void device_update_flags(Device *device,
                           DeviceScene *dscene,
                           Scene *scene,
                           Progress &progress,
                           bool bounds_valid = true);
  void device_update_geom_offsets(Device *device, DeviceScene *dscene, Scene *scene);

  void device_free(Device *device, DeviceScene *dscene, bool force_free);

  void tag_update(Scene *scene, const uint32_t flag);

  bool need_update() const;

  void apply_static_transforms(DeviceScene *dscene, Scene *scene, Progress &progress);

  string get_cryptomatte_objects(Scene *scene);
  string get_cryptomatte_assets(Scene *scene);

 protected:
  void device_update_object_transform(UpdateObjectTransformState *state,
                                      Object *ob,
                                      bool update_all,
                                      const Scene *scene,
                                      bool buffer_is_fresh);
  void device_update_object_transform_task(UpdateObjectTransformState *state);
  bool device_update_object_transform_pop_work(UpdateObjectTransformState *state,
                                               int *start_index,
                                               int *num_objects);
};

CCL_NAMESPACE_END
