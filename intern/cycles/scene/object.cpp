/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/object.h"

#include <atomic>

#include "device/device.h"
#include "kernel/types.h"
#include "scene/camera.h"
#include "scene/curves.h"
#include "scene/hair.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/particles.h"
#include "scene/pointcloud.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/stats.h"
#include "scene/volume.h"

#include "util/hash.h"
#include "util/log.h"
#include "util/map.h"
#include "util/murmurhash.h"
#include "util/progress.h"
#include "util/set.h"
#include "util/tbb.h"
#include "util/vector.h"

#include "kernel/geom/attribute.h"

CCL_NAMESPACE_BEGIN

/* Global state of object transform update. */

struct UpdateObjectTransformState {
  /* Global state used by device_update_object_transform().
   * Common for both threaded and non-threaded update.
   */

  /* Type of the motion required by the scene settings. */
  Scene::MotionType need_motion;

  /* Mapping from particle system to a index in packed particle array.
   * Only used for read.
   */
  map<ParticleSystem *, int> particle_offset;

  /* Motion offsets for each object. */
  array<uint> motion_offset;

  /* Packed object arrays. Those will be filled in. */
  uint *object_flag;
  uint *object_visibility;
  KernelObject *objects;
  Transform *object_motion_pass;
  DecomposedTransform *object_motion;

  /* Flags which will be synchronized to Integrator. */
  bool have_motion;
  bool have_curves;
  bool have_points;
  bool have_volumes;

  /* ** Scheduling queue. ** */
  Scene *scene;

  /* First unused object index in the queue. */
  int queue_start_object;
};

/* Object */

NODE_DEFINE(Object)
{
  NodeType *type = NodeType::add("object", create);

  SOCKET_NODE(geometry, "Geometry", Geometry::get_node_base_type());
  SOCKET_TRANSFORM(tfm, "Transform", transform_identity());
  SOCKET_UINT(visibility, "Visibility", PATH_RAY_VISIBILITY_ALL);
  SOCKET_COLOR(color, "Color", zero_float3());
  SOCKET_FLOAT(alpha, "Alpha", 0.0f);
  SOCKET_UINT(random_id, "Random ID", 0);
  SOCKET_INT(pass_id, "Pass ID", 0);
  SOCKET_BOOLEAN(use_holdout, "Use Holdout", false);
  SOCKET_BOOLEAN(hide_on_missing_motion, "Hide on Missing Motion", false);
  SOCKET_POINT(dupli_generated, "Dupli Generated", zero_float3());
  SOCKET_POINT2(dupli_uv, "Dupli UV", zero_float2());
  SOCKET_TRANSFORM_ARRAY(motion, "Motion", array<Transform>());
  SOCKET_FLOAT(shadow_terminator_shading_offset, "Shadow Terminator Shading Offset", 0.0f);
  SOCKET_FLOAT(shadow_terminator_geometry_offset, "Shadow Terminator Geometry Offset", 0.1f);
  SOCKET_STRING(asset_name, "Asset Name", ustring());

  SOCKET_BOOLEAN(is_shadow_catcher, "Shadow Catcher", false);

  SOCKET_BOOLEAN(is_caustics_caster, "Cast Shadow Caustics", false);
  SOCKET_BOOLEAN(is_caustics_receiver, "Receive Shadow Caustics", false);

  SOCKET_BOOLEAN(is_bake_target, "Bake Target", false);

  SOCKET_NODE(particle_system, "Particle System", ParticleSystem::get_node_type());
  SOCKET_INT(particle_index, "Particle Index", 0);

  SOCKET_FLOAT(ao_distance, "AO Distance", 0.0f);

  SOCKET_STRING(lightgroup, "Light Group", ustring());
  SOCKET_UINT(receiver_light_set, "Light Set Index", 0);
  SOCKET_UINT64(light_set_membership, "Light Set Membership", LIGHT_LINK_MASK_ALL);
  SOCKET_UINT(blocker_shadow_set, "Shadow Set Index", 0);
  SOCKET_UINT64(shadow_set_membership, "Shadow Set Membership", LIGHT_LINK_MASK_ALL);

  return type;
}

Object::Object() : Node(get_node_type())
{
  particle_system = nullptr;
  particle_index = 0;
  attr_map_offset = 0;
  bounds = BoundBox::empty;
  intersects_volume = false;
}

Object::~Object() = default;

void Object::update_motion()
{
  if (!use_motion()) {
    return;
  }

  bool have_motion = false;

  for (size_t i = 0; i < motion.size(); i++) {
    if (motion[i] == transform_empty()) {
      if (hide_on_missing_motion) {
        /* Hide objects that have no valid previous or next
         * transform, for example particle that stop existing. It
         * would be better to handle this in the kernel and make
         * objects invisible outside certain motion steps. */
        tfm = transform_empty();
        motion.clear();
        return;
      }

      if (have_motion && i == (motion.size() - 1)) {
        /* Remove last motion when it is not actually set. */
        motion.resize(motion.size() - 1);
        return;
      }

      /* Otherwise just copy center motion. */
      motion[i] = tfm;
    }

    /* Test if any of the transforms are actually different. */
    have_motion = have_motion || motion[i] != tfm;
  }

  /* Clear motion array if there is no actual motion. */
  if (!have_motion) {
    motion.clear();
  }
}

void Object::compute_bounds(bool motion_blur)
{
  const BoundBox mbounds = geometry->bounds;

  if (motion_blur && use_motion()) {
    array<DecomposedTransform> decomp(motion.size());
    transform_motion_decompose(decomp.data(), motion.data(), motion.size());

    bounds = BoundBox::empty;

    /* TODO: this is really terrible. according to PBRT there is a better
     * way to find this iteratively, but did not find implementation yet
     * or try to implement myself */
    for (float t = 0.0f; t < 1.0f; t += (1.0f / 128.0f)) {
      Transform ttfm;

      transform_motion_array_interpolate(&ttfm, decomp.data(), motion.size(), t);
      bounds.grow(mbounds.transformed(&ttfm));
    }
  }
  else {
    /* No motion blur case. */
    if (geometry->transform_applied) {
      bounds = mbounds;
    }
    else {
      bounds = mbounds.transformed(&tfm);
    }
  }
}

void Object::apply_transform(bool apply_to_motion)
{
  if (!geometry || tfm == transform_identity()) {
    return;
  }

  geometry->apply_transform(tfm, apply_to_motion);

  /* we keep normals pointing in same direction on negative scale, notify
   * geometry about this in it (re)calculates normals */
  if (transform_negative_scale(tfm)) {
    geometry->transform_negative_scaled = true;
  }

  if (bounds.valid()) {
    geometry->compute_bounds();
    compute_bounds(false);
  }

  /* tfm is not reset to identity, all code that uses it needs to check the
   * transform_applied boolean */
}

void Object::tag_update(Scene *scene)
{
  uint32_t flag = ObjectManager::UPDATE_NONE;

  if (is_modified()) {
    flag |= ObjectManager::OBJECT_MODIFIED;

    /* The per-object primitive offsets come from the object's geometry, so swapping that geometry
     * moves them - while merely moving the object does not. OBJECT_MODIFIED covers both, so the
     * narrower question is asked here. */
    if (geometry_is_modified()) {
      scene->object_manager->tag_prim_offsets_modified();

      /* Swapping geometry can take an emitter out of the tree as easily as put one in, and the
       * emission check further down only ever sees the geometry the object carries now. */
      scene->light_manager->tag_update(scene, LightManager::EMISSIVE_MESH_MODIFIED);
    }

    /* Light linking. The tree sorts and groups its emitters by membership, and reads
     * `receiver_light_set` off every object in the scene, emitter or not. */
    if (light_set_membership_is_modified() || receiver_light_set_is_modified()) {
      scene->light_manager->tag_update(scene, LightManager::LIGHT_MODIFIED);
    }

    if (use_holdout_is_modified()) {
      flag |= ObjectManager::HOLDOUT_MODIFIED;
    }

    if (is_shadow_catcher_is_modified()) {
      scene->tag_shadow_catcher_modified();
      flag |= ObjectManager::VISIBILITY_MODIFIED;
    }
  }

  if (geometry) {
    if (tfm_is_modified() || motion_is_modified()) {
      flag |= ObjectManager::TRANSFORM_MODIFIED;
      if (geometry->has_volume) {
        scene->volume_manager->tag_update({this}, flag);
      }

      /* A real light carries its position through the object transform, and the emission check
       * below is about shaders on geometry - it is not the right question for a light. Asked
       * separately so that moving a lamp still rebuilds the tree once the manager stops being
       * woken by every object that moves. */
      if (geometry->is_light()) {
        scene->light_manager->tag_update(scene, LightManager::LIGHT_MODIFIED);
      }
    }

    if (visibility_is_modified()) {
      flag |= ObjectManager::VISIBILITY_MODIFIED;
    }

    if (geometry->has_emission_shader()) {
      scene->light_manager->tag_update(scene, LightManager::EMISSIVE_MESH_MODIFIED);
    }
  }

  scene->camera->need_flags_update = true;
  scene->object_manager->tag_update(scene, flag);
}

bool Object::use_motion() const
{
  return (motion.size() > 1);
}

float Object::motion_time(const int step) const
{
  return (use_motion()) ? 2.0f * step / (motion.size() - 1) - 1.0f : 0.0f;
}

int Object::motion_step(const float time) const
{
  if (use_motion()) {
    for (size_t step = 0; step < motion.size(); step++) {
      if (time == motion_time(step)) {
        return step;
      }
    }
  }

  return -1;
}

bool Object::is_traceable() const
{
  /* Not supported for lights yet. */
  if (geometry->is_light()) {
    return false;
  }
  /* Mesh itself can be empty,can skip all such objects. */
  if (!bounds.valid() || bounds.size() == zero_float3()) {
    return false;
  }
  /* TODO(sergey): Check for mesh vertices/curves. visibility flags. */
  return true;
}

uint Object::visibility_for_tracing() const
{
  assert((visibility & ~uint(PATH_RAY_VISIBILITY_ALL)) == 0);
  return SHADOW_CATCHER_OBJECT_VISIBILITY(is_shadow_catcher, visibility);
}

float Object::compute_volume_step_size(Progress &progress) const
{
  if (geometry->is_light()) {
    /* World volume. */
    assert(static_cast<const Light *>(geometry)->is_background_light());
    for (const Node *node : geometry->get_used_shaders()) {
      const Shader *shader = static_cast<const Shader *>(node);
      if (shader->has_volume) {
        return shader->get_volume_step_rate();
      }
    }
    assert(false);
    return FLT_MAX;
  }

  if (!geometry->is_mesh() && !geometry->is_volume()) {
    return FLT_MAX;
  }

  Mesh *mesh = static_cast<Mesh *>(geometry);

  if (!mesh->has_volume) {
    return FLT_MAX;
  }

  /* Compute step rate from shaders. */
  float step_rate = FLT_MAX;

  for (Node *node : mesh->get_used_shaders()) {
    Shader *shader = static_cast<Shader *>(node);
    if (shader->has_volume) {
      if (shader->has_volume_spatial_varying || shader->has_volume_attribute_dependency) {
        step_rate = fminf(shader->get_volume_step_rate(), step_rate);
      }
    }
  }

  if (step_rate == FLT_MAX) {
    return FLT_MAX;
  }

  /* Compute step size from voxel grids. */
  float step_size = FLT_MAX;

  if (geometry->is_volume()) {
    Volume *volume = static_cast<Volume *>(geometry);

    for (Attribute &attr : volume->attributes.attributes) {
      if (attr.element == ATTR_ELEMENT_VOXEL) {
        ImageHandle &handle = attr.data_voxel_for_write();
        const ImageMetaData &metadata = handle.metadata(progress);
        if (metadata.nanovdb_byte_size == 0) {
          continue;
        }

        /* User specified step size. */
        float voxel_step_size = volume->get_step_size();

        if (voxel_step_size == 0.0f) {
          /* Auto detect step size.
           * Step size is transformed from voxel to world space. */
          Transform voxel_tfm = tfm;
          if (metadata.use_transform_3d) {
            voxel_tfm = tfm * transform_inverse(metadata.transform_3d);
          }
          voxel_step_size = reduce_min(fabs(transform_direction(&voxel_tfm, one_float3())));
        }
        else if (volume->get_object_space()) {
          /* User specified step size in object space. */
          const float3 size = make_float3(voxel_step_size, voxel_step_size, voxel_step_size);
          voxel_step_size = reduce_min(fabs(transform_direction(&tfm, size)));
        }

        if (voxel_step_size > 0.0f) {
          step_size = fminf(voxel_step_size, step_size);
        }
      }
    }
  }

  if (step_size == FLT_MAX) {
    /* Fall back to 1/10th of bounds for procedural volumes. */
    assert(bounds.valid());
    step_size = 0.1f * average(bounds.size());
  }

  step_size *= step_rate;

  return step_size;
}

int Object::get_device_index() const
{
  return index;
}

bool Object::usable_as_light() const
{
  Geometry *geom = get_geometry();
  if (!geom->is_mesh() && !geom->is_volume()) {
    return false;
  }
  /* Skip non-traceable objects. */
  if (!is_traceable()) {
    return false;
  }
  /* Skip if we are not visible for BSDFs. */
  if (!(get_visibility() & (PATH_RAY_VISIBILITY_DIFFUSE | PATH_RAY_VISIBILITY_GLOSSY |
                            PATH_RAY_VISIBILITY_TRANSMIT | PATH_RAY_VISIBILITY_VOLUME_SCATTER)))
  {
    return false;
  }
  /* Skip if we have no emission shaders. */
  /* TODO(sergey): Ideally we want to avoid such duplicated loop, since it'll
   * iterate all geometry shaders twice (when counting and when calculating
   * triangle area.
   */
  for (Node *node : geom->get_used_shaders()) {
    Shader *shader = static_cast<Shader *>(node);
    if (shader->emission_sampling != EMISSION_SAMPLING_NONE) {
      return true;
    }
  }
  return false;
}

bool Object::has_light_linking() const
{
  if (get_receiver_light_set()) {
    return true;
  }

  if (get_light_set_membership() != LIGHT_LINK_MASK_ALL) {
    return true;
  }

  return false;
}

bool Object::has_shadow_linking() const
{
  if (get_blocker_shadow_set()) {
    return true;
  }

  if (get_shadow_set_membership() != LIGHT_LINK_MASK_ALL) {
    return true;
  }

  return false;
}

void Object::adjust_volume_tfm(Transform &tfm)
{
  if (geometry) {
    if (geometry->is_volume()) {
      /* Slightly offset vertex coordinates to avoid overlapping faces with other volumes or
       * meshes. The proper solution would be to improve intersection in the kernel to support
       * robust handling of multiple overlapping faces or use an all-hit intersection similar to
       * shadows. */
      const uint name_hash = hash_string(name.c_str());
      const float3 offset = transform_direction(&tfm,
                                                make_float3(hash_uint2_to_float(name_hash, 0),
                                                            hash_uint2_to_float(name_hash, 1),
                                                            hash_uint2_to_float(name_hash, 2)) *
                                                    0.001f);
      transform_translate(tfm, offset);
    }
  }
}

void Object::set_tfm(Transform tfm)
{
  adjust_volume_tfm(tfm);
  const SocketType *socket = get_tfm_socket();
  set(*socket, tfm);
}

bool Object::tfm_equals(Transform tfm)
{
  adjust_volume_tfm(tfm);
  return tfm == get_tfm();
}

void Object::set_motion_tfm(Transform tfm, const int step_index)
{
  adjust_volume_tfm(tfm);
  array<Transform> motion = get_motion();
  motion[step_index] = tfm;
  set_motion(motion);
}

/* Object Manager */

ObjectManager::ObjectManager()
{
  update_flags = UPDATE_ALL;
  need_flags_update = true;
}

ObjectManager::~ObjectManager() = default;

void ObjectManager::update_interactive_motion(Scene *scene)
{
  std::atomic<bool> update{false};

  parallel_for(blocked_range<size_t>(0, scene->objects.size(), 32),
               [&](const blocked_range<size_t> &r) {
                 for (size_t i = r.begin(); i != r.end(); i++) {
                   Object *ob = scene->objects[i];

                   /* Mutated in place rather than through `set_motion()`, which would copy the
                    * array per object per update. The modified tagging below reproduces what the
                    * socket setter did. */
                   array<Transform> &motion = ob->get_motion();
                   const bool use_motion = motion.size() > 1;

                   if (motion.empty()) {
                     /* Can always store current matrix in motion array with a single element,
                      * since that still causes 'use_motion()' to return false. */
                     motion.resize(1);
                     motion[0] = ob->tfm;
                     ob->tag_motion_modified();
                   }
                   else if (motion[0] != ob->tfm) {
                     motion[0] = ob->tfm;
                     ob->tag_motion_modified();

                     /* Trigger another update if there was motion compared to previous frame, so
                      * that last movement does not stick around. */
                     if (use_motion) {
                       update.store(true, std::memory_order_relaxed);
                     }
                   }
                 }
               });

  if (update.load(std::memory_order_relaxed)) {
    /* The device still holds the movement of the frame that was just rendered, and it has to be
     * settled back to zero before the next one - otherwise every temporal consumer, DLSS included,
     * keeps reprojecting along a movement that already happened.
     *
     * Raising `TRANSFORM_MODIFIED` here is what used to arrange that, but it cascades into the
     * geometry and light managers and so bought a second full scene update every frame. The
     * bookkeeping is instead recorded on its own and settled by `device_update_motion_history`,
     * which rewrites nothing but the motion pass. */
    motion_history_pending = true;
  }
}

bool ObjectManager::need_motion_history_flush() const
{
  return motion_history_pending;
}

bool ObjectManager::device_update_motion_history(DeviceScene *dscene, Scene *scene)
{
  if (!motion_history_pending) {
    return true;
  }

  const Scene::MotionType need_motion = scene->need_motion();
  if (need_motion != Scene::MOTION_PASS && need_motion != Scene::MOTION_PASS_INTERACTIVE) {
    /* The pass this settles is not in use, so there is nothing on the device to settle. */
    motion_history_pending = false;
    return true;
  }

  const size_t num_objects = scene->objects.size();

  /* Object indices are only meaningful while the resident arrays still describe this object list.
   * Adding or removing an object raises a manager flag, so the caller would have taken the regular
   * path - the checks below are what makes that an invariant rather than an assumption. */
  if (num_objects == 0 || dscene->object_motion_pass.size() != OBJECT_MOTION_PASS_SIZE * num_objects
      || dscene->object_flag.size() != num_objects ||
      dscene->object_motion_pass.device_pointer == 0 || dscene->object_motion_pass.need_realloc() ||
      dscene->object_flag.need_realloc())
  {
    return false;
  }

  const scoped_callback_timer timer([scene](double time) {
    if (scene->update_stats) {
      scene->update_stats->object.times.add_entry({"device_update (motion history)", time});
    }
  });

  Transform *object_motion_pass = dscene->object_motion_pass.data();
  const uint *object_flag = dscene->object_flag.data();

  parallel_for(blocked_range<size_t>(0, num_objects, 32), [&](const blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i != r.end(); i++) {
      Object *ob = scene->objects[i];

      /* Collapses the degenerate array the history advance left behind: once the previous slot
       * holds the current transform there is no motion left to describe. This mirrors what
       * `device_update_object_transform` does on the regular path. */
      ob->update_motion();

      Transform tfm_pre;
      Transform tfm_post;
      if (ob->use_motion()) {
        tfm_pre = ob->motion[0];
        tfm_post = ob->motion[ob->motion.size() - 1];
      }
      else {
        tfm_pre = ob->tfm;
        tfm_post = ob->tfm;
      }

      /* Objects carrying deformed positions already have the motion baked into object space. */
      if (!(object_flag[ob->index] & SD_OBJECT_HAS_VERTEX_MOTION)) {
        const Transform itfm = transform_inverse(ob->tfm);
        tfm_pre = tfm_pre * itfm;
        tfm_post = tfm_post * itfm;
      }

      const int motion_pass_offset = ob->index * OBJECT_MOTION_PASS_SIZE;
      object_motion_pass[motion_pass_offset + 0] = tfm_pre;
      object_motion_pass[motion_pass_offset + 1] = tfm_post;
    }
  });

  dscene->object_motion_pass.copy_to_device();
  dscene->object_motion_pass.clear_modified();

  /* Only cleared once the new state actually reached the device. */
  motion_history_pending = false;
  return true;
}

static float object_volume_density(const Transform &tfm, Geometry *geom)
{
  if (geom->is_volume()) {
    /* Volume density automatically adjust to object scale. */
    if (static_cast<Volume *>(geom)->get_object_space()) {
      const float3 unit = normalize(one_float3());
      return 1.0f / len(transform_direction(&tfm, unit));
    }
  }

  return 1.0f;
}

static int object_num_motion_verts(Geometry *geom)
{
  return (geom->is_mesh() || geom->is_volume()) ? static_cast<Mesh *>(geom)->num_verts() :
         geom->is_hair()                        ? static_cast<Hair *>(geom)->num_keys() :
         geom->is_pointcloud()                  ? static_cast<PointCloud *>(geom)->num_points() :
                                                  0;
}

void ObjectManager::device_update_object_transform(UpdateObjectTransformState *state,
                                                   Object *ob,
                                                   bool update_all,
                                                   const Scene *scene,
                                                   bool buffer_is_fresh)
{
  KernelObject &kobject = state->objects[ob->index];
  Transform *object_motion_pass = state->object_motion_pass;

  Geometry *geom = ob->geometry;
  uint flag = 0;

  /* Compute transformations. */
  const Transform tfm = ob->tfm;
  const Transform itfm = transform_inverse(tfm);

  const float3 color = ob->color;
  const float pass_id = ob->pass_id;
  const float random_number = (float)ob->random_id * (1.0f / (float)0xFFFFFFFF);
  const int particle_index = (ob->particle_system) ?
                                 ob->particle_index + state->particle_offset[ob->particle_system] :
                                 0;

  kobject.tfm = tfm;
  kobject.itfm = itfm;
  kobject.volume_density = object_volume_density(tfm, geom);
  kobject.color[0] = color.x;
  kobject.color[1] = color.y;
  kobject.color[2] = color.z;
  kobject.alpha = ob->alpha;
  kobject.pass_id = pass_id;
  kobject.random_number = random_number;
  kobject.particle_index = particle_index;
  kobject.motion_offset = 0;
  /* These three belong to the attribute phase, not to this one: device_update_geom_offsets fills
   * them in later from the attribute map. Writing sentinels here would make that phase see every
   * object as changed and upload the whole array a second time - about 7 MB, every frame, purely
   * to undo what this loop just did. Only a buffer that does not already hold last frame's values
   * needs them initialised. */
  if (buffer_is_fresh) {
    kobject.position_offset = ATTR_STD_NOT_FOUND;
    kobject.normal_offset = ATTR_STD_NOT_FOUND;
  }
  kobject.ao_distance = ob->ao_distance;
  kobject.receiver_light_set = ob->receiver_light_set >= LIGHT_LINK_SET_MAX ?
                                   0 :
                                   ob->receiver_light_set;
  kobject.light_set_membership = ob->light_set_membership;
  kobject.blocker_shadow_set = ob->blocker_shadow_set >= LIGHT_LINK_SET_MAX ?
                                   0 :
                                   ob->blocker_shadow_set;
  kobject.shadow_set_membership = ob->shadow_set_membership;

  if (geom->get_use_motion_blur()) {
    state->have_motion = true;
  }

  if (transform_negative_scale(tfm)) {
    flag |= SD_OBJECT_NEGATIVE_SCALE;
  }

  if (geom->is_hair()) {
    const Attribute *attr_P = geom->attributes.find(ATTR_STD_POSITION);
    if (attr_P->has_motion()) {
      flag |= SD_OBJECT_HAS_VERTEX_MOTION;
    }
  }
  else if (geom->is_pointcloud()) {
    const Attribute *attr_P = geom->attributes.find(ATTR_STD_POSITION);
    if (attr_P->has_motion()) {
      flag |= SD_OBJECT_HAS_VERTEX_MOTION;
    }
  }
  else if (geom->is_volume()) {
    Volume *volume = static_cast<Volume *>(geom);
    if (volume->attributes.find(ATTR_STD_VOLUME_VELOCITY) && volume->get_velocity_scale() != 0.0f)
    {
      flag |= SD_OBJECT_HAS_VOLUME_MOTION;
      kobject.velocity_scale = volume->get_velocity_scale();
    }
  }
  else if (geom->is_mesh()) {
    Mesh *mesh = static_cast<Mesh *>(geom);
    const Attribute *attr_P = mesh->attributes.find(ATTR_STD_POSITION);
    const Attribute *subd_attr_P = (mesh->get_subdivision_type() != Mesh::SUBDIVISION_NONE) ?
                                       mesh->subd_attributes.find(ATTR_STD_POSITION) :
                                       nullptr;
    if (attr_P->has_motion() || (subd_attr_P && subd_attr_P->has_motion())) {
      flag |= SD_OBJECT_HAS_VERTEX_MOTION;
    }
    else if (mesh->attributes.find(ATTR_STD_CORNER_NORMAL)) {
      flag |= SD_OBJECT_HAS_CORNER_NORMALS;
    }
  }

  if (state->need_motion == Scene::MOTION_PASS ||
      state->need_motion == Scene::MOTION_PASS_INTERACTIVE)
  {
    /* Clear motion array if there is no actual motion. */
    ob->update_motion();

    /* Compute motion transforms. */
    Transform tfm_pre;
    Transform tfm_post;
    if (ob->use_motion()) {
      tfm_pre = ob->motion[0];
      tfm_post = ob->motion[ob->motion.size() - 1];
    }
    else {
      tfm_pre = tfm;
      tfm_post = tfm;
    }

    /* Motion transformations, is world/object space depending if mesh
     * comes with deformed position in object space, or if we transform
     * the shading point in world space. */
    if (!(flag & SD_OBJECT_HAS_VERTEX_MOTION)) {
      tfm_pre = tfm_pre * itfm;
      tfm_post = tfm_post * itfm;
    }

    const int motion_pass_offset = ob->index * OBJECT_MOTION_PASS_SIZE;
    object_motion_pass[motion_pass_offset + 0] = tfm_pre;
    object_motion_pass[motion_pass_offset + 1] = tfm_post;
  }
  else if (state->need_motion == Scene::MOTION_BLUR) {
    if (ob->use_motion()) {
      kobject.motion_offset = state->motion_offset[ob->index];

      /* Decompose transforms for interpolation. */
      if (ob->tfm_is_modified() || ob->motion_is_modified() || update_all) {
        DecomposedTransform *decomp = state->object_motion + kobject.motion_offset;
        transform_motion_decompose(decomp, ob->motion.data(), ob->motion.size());
      }

      flag |= SD_OBJECT_MOTION;
      state->have_motion = true;
    }
  }

  /* Dupli object coords and motion info. */
  kobject.dupli_generated[0] = ob->dupli_generated[0];
  kobject.dupli_generated[1] = ob->dupli_generated[1];
  kobject.dupli_generated[2] = ob->dupli_generated[2];
  kobject.dupli_uv[0] = ob->dupli_uv[0];
  kobject.dupli_uv[1] = ob->dupli_uv[1];
  kobject.num_geom_steps = geom->get_motion_steps();
  kobject.num_tfm_steps = ob->motion.size();
  kobject.numverts = object_num_motion_verts(geom);
  kobject.numprims = (geom->is_mesh() || geom->is_volume()) ?
                         static_cast<Mesh *>(geom)->num_triangles() :
                         0;
  if (buffer_is_fresh) {
    kobject.attribute_map_offset = 0;
  }

  if (ob->asset_name_is_modified() || update_all) {
    const uint32_t hash_name = util_murmur_hash3(ob->name.c_str(), ob->name.length(), 0);
    const uint32_t hash_asset = util_murmur_hash3(
        ob->asset_name.c_str(), ob->asset_name.length(), 0);
    kobject.cryptomatte_object = util_hash_to_float(hash_name);
    kobject.cryptomatte_asset = util_hash_to_float(hash_asset);
  }

  kobject.shadow_terminator_shading_offset = 1.0f /
                                             (1.0f - 0.5f * ob->shadow_terminator_shading_offset);
  kobject.shadow_terminator_geometry_offset = ob->shadow_terminator_geometry_offset;

  kobject.visibility = ob->visibility_for_tracing();
  kobject.primitive_type = geom->primitive_type();

  /* Object shadow caustics flag */
  if (ob->is_caustics_caster) {
    flag |= SD_OBJECT_CAUSTICS_CASTER;
  }
  if (ob->is_caustics_receiver) {
    flag |= SD_OBJECT_CAUSTICS_RECEIVER;
  }

  /* Object flag. */
  if (ob->use_holdout) {
    flag |= SD_OBJECT_HOLDOUT_MASK;
  }
  state->object_flag[ob->index] = flag;

  /* Have curves. */
  if (geom->is_hair()) {
    state->have_curves = true;
  }
  if (geom->is_pointcloud()) {
    state->have_points = true;
  }
  if (geom->is_volume()) {
    state->have_volumes = true;
  }

  /* Light group. */
  auto it = scene->lightgroups.find(ob->lightgroup);
  if (it != scene->lightgroups.end()) {
    kobject.lightgroup = it->second;
  }
  else {
    kobject.lightgroup = LIGHTGROUP_NONE;
  }
}

void ObjectManager::device_update_prim_offsets(Device *device, DeviceScene *dscene, Scene *scene)
{
  /* Nothing tagged this manager since the last upload, so what is resident still describes the
   * scene. Cleared here rather than at the end because of the early return below. */
  const bool offsets_were_dirty = offsets_need_update;
  offsets_need_update = false;

  if (!offsets_were_dirty && dscene->object_prim_offset.size() == scene->objects.size() &&
      dscene->object_prim_offset.device_pointer != 0)
  {
    return;
  }

  if (!scene->integrator->get_use_light_tree()) {
    const BVHLayoutMask layout_mask = device->get_bvh_layout_mask(dscene->data.kernel_features);
    if (layout_mask != BVH_LAYOUT_METAL && layout_mask != BVH_LAYOUT_MULTI_METAL &&
        layout_mask != BVH_LAYOUT_MULTI_METAL_EMBREE && layout_mask != BVH_LAYOUT_HIPRT &&
        layout_mask != BVH_LAYOUT_MULTI_HIPRT && layout_mask != BVH_LAYOUT_MULTI_HIPRT_EMBREE)
    {
      return;
    }
  }

  /* Untimed until now, and it has no `need_update()` gate at all: with the light tree enabled this
   * walks every object and re-uploads the whole array on every scene update. */
  const scoped_callback_timer timer([scene](double time) {
    if (scene->update_stats) {
      scene->update_stats->object.times.add_entry({"device_update_prim_offsets", time});
    }
  });

  /* On MetalRT, primitive / curve segment offsets can't be baked at BVH build time. Intersection
   * handlers need to apply the offset manually. */
  uint *object_prim_offset = dscene->object_prim_offset.alloc(scene->objects.size());
  for (Object *ob : scene->objects) {
    uint32_t prim_offset = 0;
    if (Geometry *const geom = ob->geometry) {
      if (geom->is_hair()) {
        prim_offset = ((Hair *const)geom)->curve_segment_offset;
      }
      else {
        prim_offset = geom->prim_offset;
      }
    }
    const uint obj_index = ob->get_device_index();
    object_prim_offset[obj_index] = prim_offset;
  }

  dscene->object_prim_offset.copy_to_device();
  dscene->object_prim_offset.clear_modified();
}

void ObjectManager::device_update_transforms(DeviceScene *dscene, Scene *scene, Progress &progress)
{
  /* Where the object manager's 3.3 ms per scene update goes: the parallel pass that fills every
   * object's kernel record, or the uploads. Reported with its remainder. */
  static const bool report_parts = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  const double parts_started = time_dt();
  double ms_prepare = 0.0, ms_fill = 0.0, ms_upload_objects = 0.0, ms_upload_motion = 0.0;
  auto stamp = [](double &slot, const double started) { slot += (time_dt() - started) * 1000.0; };
  struct PartsReport {
    const bool &enabled;
    const double &started;
    const double &prepare, &fill, &upload_objects, &upload_motion;
    ~PartsReport()
    {
      if (!enabled) {
        return;
      }
      const double total = (time_dt() - started) * 1000.0;
      fprintf(stderr,
              "OBJ_PARTS total=%.2f prepare=%.2f fill=%.2f up_objects=%.2f up_motion=%.2f "
              "unaccounted=%.2f\n",
              total,
              prepare,
              fill,
              upload_objects,
              upload_motion,
              total - prepare - fill - upload_objects - upload_motion);
      fflush(stderr);
    }
  } parts_report{
      report_parts, parts_started, ms_prepare, ms_fill, ms_upload_objects, ms_upload_motion};

  const double prepare_started = time_dt();

  /* Whether the array can still hold what the previous update left in it. Taken before `alloc`,
   * which is what would make the size match. The size check is not redundant with the flag:
   * `device_vector::alloc` frees and reallocates on a size change without raising `need_realloc`,
   * so a buffer can be brand new while that flag is clear. */
  const bool buffer_is_fresh = dscene->objects.need_realloc() ||
                               dscene->objects.size() != scene->objects.size();

  UpdateObjectTransformState state;
  state.need_motion = scene->need_motion();
  state.have_motion = false;
  state.have_curves = false;
  state.have_points = false;
  state.have_volumes = false;
  state.scene = scene;
  state.queue_start_object = 0;

  state.objects = dscene->objects.alloc(scene->objects.size());
  state.object_flag = dscene->object_flag.alloc(scene->objects.size());
  state.object_motion = nullptr;
  state.object_motion_pass = nullptr;

  if (state.need_motion == Scene::MOTION_PASS ||
      state.need_motion == Scene::MOTION_PASS_INTERACTIVE)
  {
    state.object_motion_pass = dscene->object_motion_pass.alloc(OBJECT_MOTION_PASS_SIZE *
                                                                scene->objects.size());
  }
  else if (state.need_motion == Scene::MOTION_BLUR) {
    /* Set object offsets into global object motion array. */
    uint *motion_offsets = state.motion_offset.resize(scene->objects.size());
    uint motion_offset = 0;

    for (Object *ob : scene->objects) {
      *motion_offsets = motion_offset;
      motion_offsets++;

      /* Clear motion array if there is no actual motion. */
      ob->update_motion();
      motion_offset += ob->motion.size();
    }

    state.object_motion = dscene->object_motion.alloc(motion_offset);
  }

  /* Particle system device offsets
   * 0 is dummy particle, index starts at 1.
   */
  int numparticles = 1;
  for (ParticleSystem *psys : scene->particle_systems) {
    state.particle_offset[psys] = numparticles;
    numparticles += psys->particles.size();
  }

  /* Fields that are only written when their own socket changed need writing anyway when the buffer
   * cannot have kept the previous frame's value - and that is exactly `buffer_is_fresh`, which is
   * wider than the realloc flag: `device_vector::alloc` frees and reallocates on a size change
   * without raising it. Asking the flag alone left `cryptomatte_object`, `cryptomatte_asset` and
   * the decomposed motion transforms holding whatever the allocator returned, for every object
   * whose own socket happened not to change. */
  const bool update_all = buffer_is_fresh;

  stamp(ms_prepare, prepare_started);
  const double fill_started = time_dt();

  /* Parallel object update, with grain size to avoid too much threading overhead
   * for individual objects. */
  static const int OBJECTS_PER_TASK = 32;
  parallel_for(blocked_range<size_t>(0, scene->objects.size(), OBJECTS_PER_TASK),
               [&](const blocked_range<size_t> &r) {
                 for (size_t i = r.begin(); i != r.end(); i++) {
                   Object *ob = state.scene->objects[i];
                   device_update_object_transform(&state, ob, update_all, scene, buffer_is_fresh);
                 }
               });

  stamp(ms_fill, fill_started);

  if (progress.get_cancel()) {
    return;
  }

  const double upload_objects_started = time_dt();
  dscene->objects.copy_to_device_if_modified();
  stamp(ms_upload_objects, upload_objects_started);

  const double upload_motion_started = time_dt();
  if (state.need_motion == Scene::MOTION_PASS ||
      state.need_motion == Scene::MOTION_PASS_INTERACTIVE)
  {
    dscene->object_motion_pass.copy_to_device();

    /* The full path just wrote every motion transform from the current host state, which subsumes
     * whatever the narrow flush still had to settle. The history advance at the end of this scene
     * update raises it again if the objects actually moved. */
    motion_history_pending = false;
  }
  else if (state.need_motion == Scene::MOTION_BLUR) {
    dscene->object_motion.copy_to_device();
  }
  stamp(ms_upload_motion, upload_motion_started);

  dscene->data.bvh.have_motion = state.have_motion;
  dscene->data.bvh.have_curves = state.have_curves;
  dscene->data.bvh.have_points = state.have_points;
  dscene->data.bvh.have_volumes = state.have_volumes;

  dscene->objects.clear_modified();
  dscene->object_motion_pass.clear_modified();
  dscene->object_motion.clear_modified();
}

void ObjectManager::device_update(Device *device,
                                  DeviceScene *dscene,
                                  Scene *scene,
                                  Progress &progress)
{
  if (!need_update()) {
    return;
  }

  if (update_flags & (OBJECT_ADDED | OBJECT_REMOVED)) {
    dscene->objects.tag_realloc();
    dscene->object_motion_pass.tag_realloc();
    dscene->object_motion.tag_realloc();
    dscene->object_flag.tag_realloc();
    dscene->volume_step_size.tag_realloc();

    /* If objects are added to the scene or deleted, the object indices might change, so we need to
     * update the root indices of the volume octrees. */
    scene->volume_manager->tag_update_indices();
  }

  if (update_flags & HOLDOUT_MODIFIED) {
    dscene->object_flag.tag_modified();
  }

  if (update_flags & PARTICLE_MODIFIED) {
    dscene->objects.tag_modified();
  }

  LOG_INFO << "Total " << scene->objects.size() << " objects.";

  device_free(device, dscene, false);

  if (scene->objects.empty()) {
    return;
  }

  {
    /* Assign object IDs. */
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->object.times.add_entry({"device_update (assign index)", time});
      }
    });

    int index = 0;
    for (Object *object : scene->objects) {
      object->index = index++;

      /* this is a bit too broad, however a bigger refactor might be needed to properly separate
       * update each type of data (transform, flags, etc.) */
      if (object->is_modified()) {
        dscene->objects.tag_modified();
        dscene->object_motion_pass.tag_modified();
        dscene->object_motion.tag_modified();
        dscene->object_flag.tag_modified();
        dscene->volume_step_size.tag_modified();
      }

      /* Update world object index. */
      if (!object->get_geometry()->is_light()) {
        continue;
      }

      const Light *light = static_cast<const Light *>(object->get_geometry());
      if (light->is_background_light()) {
        dscene->data.background.object_index = object->index;
      }
    }
  }

  {
    /* set object transform matrices, before applying static transforms */
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->object.times.add_entry(
            {"device_update (copy objects to device)", time});
      }
    });

    progress.set_status("Updating Objects", "Copying Transformations to device");
    device_update_transforms(dscene, scene, progress);
  }

  for (Object *object : scene->objects) {
    /* Recorded before the flags go: the geometry manager computes bounds after this point and can
     * no longer tell which objects moved. */
    object->bounds_need_update = object->bounds_need_update || object->is_modified();
    object->clear_modified();
  }
}

void ObjectManager::device_update_flags(Device * /*unused*/,
                                        DeviceScene *dscene,
                                        Scene *scene,
                                        Progress & /*progress*/,
                                        bool bounds_valid)
{
  if (!need_update() && !need_flags_update) {
    return;
  }

  const scoped_callback_timer timer([scene](double time) {
    if (scene->update_stats) {
      scene->update_stats->object.times.add_entry({"device_update_flags", time});
    }
  });

  if (bounds_valid) {
    /* Object flags and calculations related to volume depend on proper bounds calculated, which
     * might not be available yet when object flags are updated for displacement or hair
     * transparency calculation. In this case do not clear the need_flags_update, so that these
     * values which depend on bounds are re-calculated when the device_update process comes back
     * here from the "Updating Objects Flags" stage. */
    update_flags = UPDATE_NONE;
    need_flags_update = false;
  }

  if (scene->objects.empty()) {
    return;
  }

  /* Object info flag. */
  uint *object_flag = dscene->object_flag.data();

  /* Object volume intersection. */
  vector<Object *> volume_objects;
  bool has_volume_objects = false;
  BoundBox volume_bounds = BoundBox::empty;
  for (Object *object : scene->objects) {
    if (object->geometry->has_volume) {
      /* If the bounds are not valid it is not always possible to calculate the volume step, and
       * the step size is not needed for the displacement. So, delay calculation of the volume
       * step size until the final bounds are known. */
      if (bounds_valid) {
        volume_objects.push_back(object);
        /* Where the media are, for the viewport's volume grid to reach exactly that far and no
         * further. Recorded here because this is the one place that walks the volume objects with
         * their final bounds; the grid is built per frame and has no scene to ask. */
        volume_bounds.grow(object->bounds);
      }
      has_volume_objects = true;
    }
  }

  if (bounds_valid) {
    const bool valid = volume_bounds.valid();
    dscene->data.froxel.bounds_min = valid ? make_float4(volume_bounds.min, 1.0f) : zero_float4();
    dscene->data.froxel.bounds_max = valid ? make_float4(volume_bounds.max, 1.0f) : zero_float4();
  }

  for (Object *object : scene->objects) {
    if (object->geometry->has_volume) {
      object_flag[object->index] |= SD_OBJECT_HAS_VOLUME;
      object_flag[object->index] &= ~SD_OBJECT_HAS_VOLUME_ATTRIBUTES;

      for (const Attribute &attr : object->geometry->attributes.attributes) {
        if (attr.element == ATTR_ELEMENT_VOXEL) {
          object_flag[object->index] |= SD_OBJECT_HAS_VOLUME_ATTRIBUTES;
        }
      }
    }
    else {
      object_flag[object->index] &= ~(SD_OBJECT_HAS_VOLUME | SD_OBJECT_HAS_VOLUME_ATTRIBUTES);
    }

    if (object->is_shadow_catcher) {
      object_flag[object->index] |= SD_OBJECT_SHADOW_CATCHER;
    }
    else {
      object_flag[object->index] &= ~SD_OBJECT_SHADOW_CATCHER;
    }

    /* Corner normals might get removed by subdivision and displacement. */
    if (object->geometry->attributes.find(ATTR_STD_CORNER_NORMAL)) {
      object_flag[object->index] |= SD_OBJECT_HAS_CORNER_NORMALS;
    }
    else {
      object_flag[object->index] &= ~SD_OBJECT_HAS_CORNER_NORMALS;
    }

    if (bounds_valid) {
      object->intersects_volume = false;
      for (Object *volume_object : volume_objects) {
        if (object == volume_object) {
          continue;
        }
        if (object->bounds.intersects(volume_object->bounds)) {
          object_flag[object->index] |= SD_OBJECT_INTERSECTS_VOLUME;
          object->intersects_volume = true;
          break;
        }
      }
    }
    else if (has_volume_objects) {
      /* Not really valid, but can't make more reliable in the case
       * of bounds not being up to date.
       */
      object_flag[object->index] |= SD_OBJECT_INTERSECTS_VOLUME;
    }
  }

  /* Copy object flag. */
  dscene->object_flag.copy_to_device();
  dscene->object_flag.clear_modified();
}

/* Where a geometry's positions and normals sit inside the attribute map segment starting at
 * `attr_map_offset`. Pulled out of the object loop so the per-geometry pass and the per-object one
 * ask the identical question - the object pass now only needs it for objects that carry their own
 * map. Reads the map, writes nothing, so it is safe to call from several threads. */
static int geometry_position_offset(const DeviceScene *dscene,
                                    const Geometry *geom,
                                    const size_t attr_map_offset)
{
  if (geom->is_mesh() || geom->is_volume()) {
    const int offset = find_attribute(dscene->attributes_map.data(),
                                      attr_map_offset,
                                      PRIMITIVE_TRIANGLE,
                                      ATTR_STD_POSITION)
                           .offset;
    assert(offset != ATTR_STD_NOT_FOUND ||
           static_cast<const Mesh *>(geom)->num_triangles() == 0);
    return offset;
  }
  if (geom->is_hair()) {
    const int offset = find_attribute(dscene->attributes_map.data(),
                                      attr_map_offset,
                                      PRIMITIVE_CURVE_THICK,
                                      ATTR_STD_POSITION)
                           .offset;
    assert(offset != ATTR_STD_NOT_FOUND || static_cast<const Hair *>(geom)->num_keys() == 0);
    return offset;
  }
  if (geom->is_pointcloud()) {
    const int offset = find_attribute(
                           dscene->attributes_map.data(), attr_map_offset, PRIMITIVE_POINT,
                           ATTR_STD_POSITION)
                           .offset;
    assert(offset != ATTR_STD_NOT_FOUND ||
           static_cast<const PointCloud *>(geom)->num_points() == 0);
    return offset;
  }
  return ATTR_STD_NOT_FOUND;
}

static int geometry_normal_offset(const DeviceScene *dscene,
                                  const Geometry *geom,
                                  const size_t attr_map_offset)
{
  /* Only meshes and volumes carry normals here; hair and point clouds leave this unset, as before.
   */
  if (!geom->is_mesh() && !geom->is_volume()) {
    return ATTR_STD_NOT_FOUND;
  }

  int offset = find_attribute(dscene->attributes_map.data(),
                              attr_map_offset,
                              PRIMITIVE_TRIANGLE,
                              ATTR_STD_CORNER_NORMAL)
                   .offset;
  if (offset == ATTR_STD_NOT_FOUND) {
    offset = find_attribute(dscene->attributes_map.data(),
                            attr_map_offset,
                            PRIMITIVE_TRIANGLE,
                            ATTR_STD_VERTEX_NORMAL)
                 .offset;
  }
  assert(offset != ATTR_STD_NOT_FOUND || static_cast<const Mesh *>(geom)->num_triangles() == 0);
  return offset;
}

void ObjectManager::device_update_geom_offsets(Device * /*unused*/,
                                               DeviceScene *dscene,
                                               Scene *scene)
{
  if (dscene->objects.size() == 0) {
    return;
  }

  /* Untimed until now, and ungated: every object is walked and its attribute maps are looked up by
   * linear scan, so this is O(objects x attributes) on every scene update. */
  const scoped_callback_timer timer([scene](double time) {
    if (scene->update_stats) {
      scene->update_stats->object.times.add_entry({"device_update_geom_offsets", time});
    }
  });

  KernelObject *kobjects = dscene->objects.data();

  std::atomic<bool> update{false};

  /* What the lookups below answer depends only on the geometry: every object that has no attribute
   * map of its own reads the same offsets, out of the same place in the map, as every other object
   * sharing that geometry. Each lookup is a linear scan of a map segment, so on a scene where
   * Geometry Nodes turn 837 objects into 33858 instances that is 33858 scans where 837 would do.
   * Objects that carry their own map offset are computed individually, as before. */
  struct SharedGeometryOffsets {
    int position;
    int normal;
    int numverts;
    bool valid = false;
  };
  /* Indexed by `Geometry::index`, not hashed. The indices are assigned at the top of
   * `device_update_attributes` - the function that calls this one - so they are current here. */
  vector<SharedGeometryOffsets> shared_offsets(scene->geometry.size());

  /* Resolving each geometry's offsets in its own pass, ahead of the objects.
   *
   * The cache used to be filled lazily from inside the object loop, by whichever object reached a
   * geometry first. That is the one thing preventing the object loop from running in parallel -
   * everything else in it writes to `kobjects[object->index]`, and those indices are unique and
   * equal to the object's position. Hoisting the fill removes the dependency without changing a
   * single value: the same lookups produce the same numbers, just once per geometry instead of
   * once per object that happens to be first.
   *
   * Gating this function on "the attribute map is byte-identical" was considered and rejected. The
   * map is addressed by layout, not by object-to-offset binding, so an instance rebinding to
   * another already-present geometry leaves the map identical while its offsets must change - and
   * on a scene where Geometry Nodes re-emit 33858 instances per frame that is routine, not exotic.
   * The gate would also have to cover `numverts`, which is not derived from the map at all. A stale
   * offset points the kernel at another geometry's attribute table, which shows as wrong shading
   * that never heals; parallelising costs the same milliseconds without inventing that failure. */
  parallel_for(blocked_range<size_t>(0, scene->geometry.size(), 32),
               [&](const blocked_range<size_t> &r) {
                 for (size_t i = r.begin(); i != r.end(); i++) {
                   Geometry *geom = scene->geometry[i];
                   if (geom->index < 0 || size_t(geom->index) >= shared_offsets.size()) {
                     continue;
                   }
                   shared_offsets[geom->index] = {
                       geometry_position_offset(dscene, geom, geom->attr_map_offset),
                       geometry_normal_offset(dscene, geom, geom->attr_map_offset),
                       object_num_motion_verts(geom),
                       true};
                 }
               });

  parallel_for(
      blocked_range<size_t>(0, scene->objects.size(), 32),
      [&](const blocked_range<size_t> &range) {
        bool range_update = false;

        for (size_t index = range.begin(); index != range.end(); index++) {
          Object *object = scene->objects[index];
          Geometry *geom = object->geometry;

          KernelObject &kobject = kobjects[object->index];

          /* An object attribute map cannot have a zero offset because mesh maps come first. */
          size_t attr_map_offset = object->attr_map_offset;
          const bool shares_geometry_map = (attr_map_offset == 0);
          if (shares_geometry_map) {
            attr_map_offset = geom->attr_map_offset;
          }

          if (kobject.attribute_map_offset != attr_map_offset) {
            kobject.attribute_map_offset = attr_map_offset;
            range_update = true;
          }

          /* Out-of-range would mean a geometry the attribute pass never saw, which cannot happen
           * while both walk `scene->geometry`; checked anyway so a future reordering degrades to
           * the slow path rather than reading past the end. */
          const bool index_usable = geom->index >= 0 &&
                                    size_t(geom->index) < shared_offsets.size();

          int position_offset;
          int normal_offset;
          int numverts;

          if (shares_geometry_map && index_usable && shared_offsets[geom->index].valid) {
            const SharedGeometryOffsets &cached = shared_offsets[geom->index];
            position_offset = cached.position;
            normal_offset = cached.normal;
            numverts = cached.numverts;
          }
          else {
            /* Object carries its own attribute map, so the offsets are its own. Reads the shared
             * map without writing anything, which is why this stays inside the parallel range. */
            position_offset = geometry_position_offset(dscene, geom, attr_map_offset);
            normal_offset = geometry_normal_offset(dscene, geom, attr_map_offset);
            numverts = object_num_motion_verts(geom);
          }

          if (kobject.position_offset != position_offset) {
            kobject.position_offset = position_offset;
            range_update = true;
          }
          if (kobject.normal_offset != normal_offset) {
            kobject.normal_offset = normal_offset;
            range_update = true;
          }
          if (kobject.numverts != numverts) {
            kobject.numverts = numverts;
            range_update = true;
          }
        }

        if (range_update) {
          update.store(true, std::memory_order_relaxed);
        }
      });

  if (update.load(std::memory_order_relaxed)) {
    /* The whole object array, about 7 MB on a scene with 33858 of them. Reported because it is not
     * obvious whether anything here actually differs from frame to frame - if it does not, this
     * never fires, and if it does, it is a bigger cost than the loop that decided it. */
    static const bool report_upload = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
    if (report_upload) {
      fprintf(stderr,
              "GEOM_OFFSETS uploaded objects=%zu bytes=%zu\n",
              scene->objects.size(),
              dscene->objects.memory_size());
      fflush(stderr);
    }
    dscene->objects.copy_to_device();
  }
}

void ObjectManager::device_free(Device * /*unused*/, DeviceScene *dscene, bool force_free)
{
  dscene->objects.free_if_need_realloc(force_free);
  dscene->object_motion_pass.free_if_need_realloc(force_free);
  dscene->object_motion.free_if_need_realloc(force_free);
  dscene->object_flag.free_if_need_realloc(force_free);
  dscene->object_prim_offset.free_if_need_realloc(force_free);
}

void ObjectManager::apply_static_transforms(DeviceScene *dscene, Scene *scene, Progress &progress)
{
  /* todo: normals and displacement should be done before applying transform! */
  /* todo: create objects/geometry in right order! */

  /* counter geometry users */
  map<Geometry *, int> geometry_users;
  const Scene::MotionType need_motion = scene->need_motion();
  const bool motion_blur = need_motion == Scene::MOTION_BLUR;
  const bool apply_to_motion = need_motion != Scene::MOTION_PASS &&
                               need_motion != Scene::MOTION_PASS_INTERACTIVE;
  int i = 0;

  for (Object *object : scene->objects) {
    const map<Geometry *, int>::iterator it = geometry_users.find(object->geometry);

    if (it == geometry_users.end()) {
      geometry_users[object->geometry] = 1;
    }
    else {
      it->second++;
    }
  }

  if (progress.get_cancel()) {
    return;
  }

  uint *object_flag = dscene->object_flag.data();

  /* apply transforms for objects with single user geometry */
  for (Object *object : scene->objects) {
    /* Annoying feedback loop here: we can't use is_instanced() because
     * it'll use uninitialized transform_applied flag.
     *
     * Could be solved by moving reference counter to Geometry.
     */
    Geometry *geom = object->geometry;
    bool apply = (geometry_users[geom] == 1) && !geom->has_surface_bssrdf &&
                 !geom->has_true_displacement();

    if (geom->is_mesh()) {
      Mesh *mesh = static_cast<Mesh *>(geom);
      apply = apply && mesh->get_subdivision_type() == Mesh::SUBDIVISION_NONE;
    }
    else if (geom->is_hair() || geom->is_pointcloud()) {
      /* Can't apply non-uniform scale to curves and points, this can't be
       * represented by control points and radius alone. */
      float scale;
      apply = apply && transform_uniform_scale(object->tfm, scale);
    }

    if (apply) {
      if (!(motion_blur && object->use_motion())) {
        if (!geom->transform_applied) {
          object->apply_transform(apply_to_motion);
          geom->transform_applied = true;

          if (progress.get_cancel()) {
            return;
          }
        }

        object_flag[i] |= SD_OBJECT_TRANSFORM_APPLIED;
      }
    }

    i++;
  }
}

void ObjectManager::tag_update(Scene *scene, const uint32_t flag)
{
  update_flags |= flag;

  /* `object_prim_offset` is one primitive offset per object, taken from the object's geometry. Only
   * a change of which objects there are, of which geometry an object carries, or of the geometry
   * offsets themselves can move it - and a moving object does none of those. Listing what cannot
   * affect it rather than what can, so an unrecognised flag still forces the update. */
  constexpr uint32_t offsets_unaffected = TRANSFORM_MODIFIED | VISIBILITY_MODIFIED |
                                          HOLDOUT_MODIFIED | PARTICLE_MODIFIED |
                                          GEOMETRY_MANAGER | OBJECT_MODIFIED;
  if ((flag & ~offsets_unaffected) != 0) {
    offsets_need_update = true;
  }

  /* avoid infinite loops if the geometry manager tagged us for an update */
  if ((flag & GEOMETRY_MANAGER) == 0) {
    uint32_t geometry_flag = GeometryManager::OBJECT_MANAGER;

    /* Also notify in case added or removed objects were instances, as no Geometry might have been
     * added or removed, but the BVH still needs to updated. */
    if ((flag & (OBJECT_ADDED | OBJECT_REMOVED)) != 0) {
      geometry_flag |= (GeometryManager::GEOMETRY_ADDED | GeometryManager::GEOMETRY_REMOVED);
    }

    if ((flag & TRANSFORM_MODIFIED) != 0) {
      geometry_flag |= GeometryManager::TRANSFORM_MODIFIED;
    }

    if ((flag & VISIBILITY_MODIFIED) != 0) {
      geometry_flag |= GeometryManager::VISIBILITY_MODIFIED;
    }

    scene->geometry_manager->tag_update(scene, geometry_flag);
  }

  /* The light manager has no partial path: any reason at all costs a full teardown and rebuild of
   * the light tree, measured at 5.7 ms on a scene of 33858 instances. Waking it from every object
   * tag meant paying that on every frame of playback, where 74 objects move and none of them
   * emits.
   *
   * Listed as what cannot reach the light tree rather than what can, so an unrecognised flag still
   * wakes the manager. Each exclusion is covered elsewhere:
   *
   * - TRANSFORM_MODIFIED: moving an emitter or a lamp is tagged directly in `Object::tag_update`,
   *   which knows whether this object is either. Moving anything else does not enter the tree.
   * - OBJECT_MODIFIED: the sockets the tree actually reads - light set membership, receiver light
   *   set, the geometry itself - are each tagged by name there too. The rest of what makes an
   *   object "modified" (pass id, colour, alpha, asset name, caustics, AO distance, shadow sets)
   *   is object data in the kernel, not tree structure.
   * - HOLDOUT_MODIFIED, PARTICLE_MODIFIED: neither is read while building the tree.
   * - GEOMETRY_MANAGER: the geometry manager tags the light manager on its own account, and this
   *   flag only exists to stop the two managers tagging each other in a loop. */
  constexpr uint32_t light_tree_unaffected = TRANSFORM_MODIFIED | OBJECT_MODIFIED |
                                             HOLDOUT_MODIFIED | PARTICLE_MODIFIED |
                                             GEOMETRY_MANAGER;
  if ((flag & ~light_tree_unaffected) != 0) {
    scene->light_manager->tag_update(scene, LightManager::OBJECT_MANAGER);
  }

  /* Integrator's shadow catcher settings depends on object visibility settings. */
  if (flag & (OBJECT_ADDED | OBJECT_REMOVED | OBJECT_MODIFIED)) {
    scene->integrator->tag_update(scene, Integrator::OBJECT_MANAGER);
  }
}

bool ObjectManager::need_update() const
{
  return update_flags != UPDATE_NONE;
}

string ObjectManager::get_cryptomatte_objects(Scene *scene)
{
  string manifest = "{";

  unordered_set<ustring> objects;
  for (Object *object : scene->objects) {
    if (objects.contains(object->name)) {
      continue;
    }
    objects.insert(object->name);
    const uint32_t hash_name = util_murmur_hash3(object->name.c_str(), object->name.length(), 0);
    manifest += string_printf("\"%s\":\"%08x\",", object->name.c_str(), hash_name);
  }
  manifest[manifest.size() - 1] = '}';
  return manifest;
}

string ObjectManager::get_cryptomatte_assets(Scene *scene)
{
  string manifest = "{";
  unordered_set<ustring> assets;
  for (Object *ob : scene->objects) {
    if (assets.contains(ob->asset_name)) {
      continue;
    }
    assets.insert(ob->asset_name);
    const uint32_t hash_asset = util_murmur_hash3(
        ob->asset_name.c_str(), ob->asset_name.length(), 0);
    manifest += string_printf("\"%s\":\"%08x\",", ob->asset_name.c_str(), hash_asset);
  }
  manifest[manifest.size() - 1] = '}';
  return manifest;
}

CCL_NAMESPACE_END
