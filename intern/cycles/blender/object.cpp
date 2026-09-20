/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "blender/light_linking.h"
#include "blender/object_cull.h"
#include "blender/sync.h"
#include "blender/util.h"

#include "scene/camera.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/particles.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"

#include "util/hash.h"
#include "util/log.h"
#include "util/task.h"

#include "BKE_duplilist.hh"
#include "BKE_layer.hh"
#include "BKE_lib_id.hh"
#include "BKE_material.hh"
#include "BKE_object.hh"

#include "DEG_depsgraph_query.hh"

#include "RE_engine.h"

using blender::Object;

CCL_NAMESPACE_BEGIN

/* Utilities */

bool BlenderSync::BKE_object_is_modified(blender::Object &b_ob)
{
  /* Walking the modifier stack and the material links is the same work for every instance of this
   * object, so it is answered once per pass - see `modifier_answers_cache`. */
  const auto cached = object_is_modified_cache.find(&b_ob);
  if (cached != object_is_modified_cache.end()) {
    return cached->second;
  }

  const bool result = compute_object_is_modified(b_ob);
  object_is_modified_cache.emplace(&b_ob, result);
  return result;
}

bool BlenderSync::compute_object_is_modified(blender::Object &b_ob)
{
  /* test if we can instance or if the object is modified */
  if (b_ob.type == blender::OB_MBALL) {
    /* Multi-user and dupli meta-balls are fused, can't instance. */
    return true;
  }
  const int settings = preview ? blender::eModifierMode_Realtime : blender::eModifierMode_Render;
  if ((blender::BKE_object_is_modified(b_scene, &b_ob) & settings) != 0) {
    /* modifiers */
    return true;
  }

  /* Object level material links. Note the geometry material slot array may not match
   * the object matbits array, so we need to guard against out of bounds. */
  for (const int i : blender::IndexRange(BKE_object_material_count_eval(&b_ob))) {
    if (i < b_ob.totcol && b_ob.matbits && b_ob.matbits[i] != 0) {
      return true;
    }
  }

  return false;
}

bool BlenderSync::object_is_geometry(BObjectInfo &b_ob_info)
{
  blender::ID *b_ob_data = b_ob_info.object_data;

  if (!b_ob_data) {
    return false;
  }

  const blender::ObjectType type = b_ob_info.iter_object->type;

  if (type == blender::OB_VOLUME || type == blender::OB_CURVES || type == blender::OB_POINTCLOUD ||
      type == blender::OB_LAMP)
  {
    /* Will be exported as geometry. */
    return true;
  }

  return GS(b_ob_data->name) == blender::ID_ME;
}

bool BlenderSync::object_can_have_geometry(blender::Object &b_ob)
{
  const blender::ObjectType type = b_ob.type;
  switch (type) {
    case blender::OB_MESH:
    case blender::OB_CURVES_LEGACY:
    case blender::OB_SURF:
    case blender::OB_MBALL:
    case blender::OB_FONT:
    case blender::OB_CURVES:
    case blender::OB_POINTCLOUD:
    case blender::OB_VOLUME:
      /* TODO(weizhen): OB_LAMP */
      return true;
    default:
      return false;
  }
}

bool BlenderSync::object_is_light(blender::Object &b_ob)
{
  blender::ID *b_ob_data = object_get_data(b_ob, true);

  return (b_ob_data && GS(b_ob_data->name) == blender::ID_LA);
}

bool BlenderSync::object_is_camera(blender::Object &b_ob)
{
  blender::ID *b_ob_data = object_get_data(b_ob, true);

  return (b_ob_data && GS(b_ob_data->name) == blender::ID_CA);
}

bool BlenderSync::instancer_looks_clean(blender::Object &b_parent)
{
  /* Cycles' own recalc sets, not `b_parent.id.recalc`: the depsgraph flags are snapshotted in
   * `sync_recalc` and Blender clears them once the editors have been updated, which happens even on
   * frames where this session failed to take its lock and never reached `sync_data`. The sets
   * survive across such frames, the flags do not. */
  if (object_map.check_recalc(&b_parent.id)) {
    return false;
  }

  /* Which key a change lands on depends on whether the instancer carries modifiers: a Geometry
   * Nodes host reports itself modified and is keyed by the object, a plain one by its data. A
   * change to the instances - their number, their placement, which object they come from - arrives
   * as ID_RECALC_GEOMETRY on that key and never touches `object_map`, so checking only the object
   * map would let every Geometry Nodes edit through unnoticed. */
  if (geometry_map.check_recalc(&b_parent.id)) {
    return false;
  }
  if (b_parent.data && geometry_map.check_recalc(b_parent.data)) {
    return false;
  }

  return true;
}

void BlenderSync::instancer_fill_fingerprint(const blender::DupliObject &dupli,
                                             DupliFingerprint &r_fingerprint)
{
  /* Zeroed first: the fingerprint is compared as raw bytes, and whatever the compiler leaves in the
   * padding would otherwise read as a change. */
  memset(&r_fingerprint, 0, sizeof(r_fingerprint));
  r_fingerprint.ob = dupli.ob;
  r_fingerprint.ob_data = dupli.ob_data;
  r_fingerprint.particle_system = dupli.particle_system;
  memcpy(r_fingerprint.mat, dupli.mat, sizeof(r_fingerprint.mat));
  memcpy(r_fingerprint.orco, dupli.orco, sizeof(r_fingerprint.orco));
  memcpy(r_fingerprint.uv, dupli.uv, sizeof(r_fingerprint.uv));
  r_fingerprint.random_id = dupli.random_id;
  memcpy(r_fingerprint.persistent_id, dupli.persistent_id, sizeof(r_fingerprint.persistent_id));
  r_fingerprint.type = dupli.type;
  r_fingerprint.no_draw = dupli.no_draw;
}

size_t BlenderSync::instancer_cache_signature(blender::View3D *b_v3d, bool show_lights)
{
  /* State outside the instancers that decides what an instance syncs to. None of it is reported
   * through any object's recalc flags, so a change here has to invalidate every group at once. */
  size_t hash = 0;
  auto mix = [&hash](const size_t value) { hash = (hash * 0x100000001b3ull) ^ value; };

  mix(size_t(scene->need_motion()));
  mix(size_t(view_layer.material_override));
  mix(size_t(view_layer.world_override));
  mix(size_t(view_layer.use_surfaces) | (size_t(view_layer.use_hair) << 1) |
      (size_t(view_layer.use_volumes) << 2) | (size_t(view_layer.use_motion_blur) << 3));
  mix(size_t(show_lights));
  mix(size_t(preview) | (size_t(use_adaptive_subdivision) << 1));
  mix(size_t(b_bake_target));
  /* Both feed `BKE_object_is_visible_in_viewport`, which decides whether an instance is walked. */
  mix(b_v3d ? size_t(b_v3d->local_view_uid) : 0);
  mix(b_v3d ? size_t(b_v3d->object_type_exclude_viewport) : 0);

  return hash;
}

size_t BlenderSync::instancer_visibility_signature(blender::ViewLayer &b_view_layer,
                                                   blender::Object &b_parent,
                                                   const vector<blender::Object *> &sources)
{
  /* `sync_object` reads holdout, indirect-only, ray visibility, shadow catcher and display type off
   * the instancer and its sources *before* the sync gate, so a skipped group keeps whatever those
   * were when it was recorded. None of them shows up in the recalc sets: their RNA callbacks raise
   * ID_RECALC_SYNC_TO_EVAL, and the depsgraph raises that on every evaluated-copy refresh, which
   * during playback is every frame - reacting to the flag disabled the cache outright and cost 133
   * frames of 141. So the values are compared instead of the flag.
   *
   * A digest rather than a field-by-field record: it is read once per group per frame against 18
   * groups, and it has to grow whenever something else joins the pre-gate reads. */
  auto add_object = [&b_view_layer](size_t hash, blender::Object &b_ob) {
    hash = hash_uint2(hash, b_ob.visibility_flag);
    hash = hash_uint2(hash, uint(b_ob.dt));
    /* Both shadow terminator offsets are read before the gate as well, and their RNA properties
     * carry no update callback - the depsgraph raises only SYNC_TO_EVAL and PARAMETERS for them,
     * neither of which any recalc set here looks at. Without these two the offsets edited on an
     * instanced source would simply not reach its instances until something else broke the group.
     * Hashed as raw bits because that is what has to be compared, not what the float means. */
    hash = hash_uint2(hash, __float_as_uint(b_ob.shadow_terminator_shading_offset));
    hash = hash_uint2(hash, __float_as_uint(b_ob.shadow_terminator_geometry_offset));
    if (const blender::Base *base = BKE_view_layer_base_find(&b_view_layer, &b_ob)) {
      hash = hash_uint2(hash, uint(base->flag));
    }
    return hash;
  };

  size_t hash = add_object(0, b_parent);
  for (blender::Object *b_source : sources) {
    hash = add_object(hash, *b_source);
  }
  return hash;
}

bool BlenderSync::instancer_group_is_clean(blender::ViewLayer &b_view_layer,
                                           blender::Object &b_parent,
                                           const InstancerCacheEntry &entry)
{
  if (!instancer_looks_clean(b_parent)) {
    return false;
  }

  if (instancer_visibility_signature(b_view_layer, b_parent, entry.sources) !=
      entry.visibility_signature)
  {
    return false;
  }

  /* The instancer's own keys say nothing about the objects it instances: a material assignment,
   * a visibility switch or a modifier edit on the source lands on the source's keys alone. */
  for (blender::Object *b_source : entry.sources) {
    if (!instancer_looks_clean(*b_source)) {
      return false;
    }
  }

  /* What an instance renders is often not the source object's own data - a Geometry Nodes instance
   * carries data that belongs to no object at all. A change to it lands on that data's key, which
   * neither the instancer nor the source objects cover. These are exactly the pairs recorded for
   * `instance_geometries_by_object`, which exists for the same reason. */
  for (const std::pair<void *, blender::ID *> &edge : entry.geometry_edges) {
    if (geometry_map.check_recalc(edge.second)) {
      return false;
    }
  }

  return true;
}

void BlenderSync::instancer_replay_shared(const InstancerCacheEntry &entry)
{
  for (Geometry *geometry : entry.geometries) {
    geometry_map.used(geometry);
  }
  for (ParticleSystem *particle_system : entry.particle_systems) {
    particle_system_map.used(particle_system);
  }
  /* Unioned rather than assigned: the map is keyed by source object, and the same source can be
   * instanced by more than one instancer - assigning a slice would drop another group's edges. */
  for (const std::pair<void *, blender::ID *> &edge : entry.geometry_edges) {
    instance_geometries_by_object[edge.first].insert(edge.second);
  }

}

void BlenderSync::sync_object_motion_init(blender::Object &b_parent,
                                          blender::Object &b_ob,
                                          Object *object)
{
  /* Initialize motion blur for object, detecting if it's enabled and creating motion
   * steps array if so. */
  array<Transform> motion = object->get_motion();

  Geometry *geom = object->get_geometry();
  if (!geom) {
    return;
  }

  int motion_steps = 0;
  bool use_motion_blur = false;

  const Scene::MotionType need_motion = scene->need_motion();
  if (need_motion == Scene::MOTION_BLUR) {
    motion_steps = object_motion_steps(b_parent, b_ob, Object::MAX_MOTION_STEPS);
    if (motion_steps && object_use_deform_motion(b_parent, b_ob)) {
      use_motion_blur = true;
    }
  }
  else if (need_motion != Scene::MOTION_NONE) {
    motion_steps = 3;
  }

  geom->set_use_motion_blur(use_motion_blur);

  motion.resize(motion_steps, transform_empty());

  if (motion_steps) {
    motion[motion_steps / 2] = object->get_tfm();

    /* update motion socket before trying to access object->motion_time */
    object->set_motion(motion);

    /* Only motion blur consumes these: the loop over motion times is skipped entirely in the
     * viewport. Filling the set there is three insertions per object per frame - 100k of them on a
     * scene where Geometry Nodes produce 33858 instances - for a value nothing reads. */
    if (need_motion == Scene::MOTION_BLUR) {
      for (size_t step = 0; step < motion_steps; step++) {
        motion_times.insert(object->motion_time(step));
      }
    }
  }
  else {
    object->set_motion(motion);
  }
}

Object *BlenderSync::sync_object(blender::ViewLayer &b_view_layer,
                                 blender::Object &b_ob,
                                 blender::DEGObjectIterData &b_deg_iter_data,
                                 const float motion_time,
                                 bool use_particle_hair,
                                 bool show_lights,
                                 BlenderObjectCulling &culling,
                                 TaskPool *geom_task_pool)
{
  const bool is_instance = b_deg_iter_data.dupli_object_current;
  blender::Object *b_parent = is_instance ? b_deg_iter_data.dupli_parent : &b_ob;
  blender::Object *b_real_object = is_instance ? b_deg_iter_data.dupli_object_current->ob : &b_ob;
  const bool use_adaptive_subdiv = object_subdivision_type(
                                       *b_real_object, preview, use_adaptive_subdivision) !=
                                   Mesh::SUBDIVISION_NONE;
  BObjectInfo b_ob_info{
      &b_ob, b_real_object, object_get_data(b_ob, use_adaptive_subdiv), use_adaptive_subdiv};
  const bool motion = motion_time != 0.0f;
  const Transform tfm = get_transform(b_ob.object_to_world());
  const int *persistent_id = nullptr;
  if (is_instance) {
    persistent_id = b_deg_iter_data.dupli_object_current->persistent_id;
    if (!motion && !b_ob_info.is_real_object_data()) {
      /* Remember which object data the geometry is coming from, so that we can sync it when the
       * object has changed. */
      instance_geometries_by_object[b_ob_info.real_object].insert(b_ob_info.object_data);

      /* Recorded here rather than from the objects the group returns: this runs before the geometry
       * and culling tests below, and an instance that fails them still contributed an edge. The
       * linear scan is over a handful of distinct sources, not over the instances. */
      if (instancer_recording) {
        const std::pair<void *, blender::ID *> edge(b_ob_info.real_object, b_ob_info.object_data);
        bool known = false;
        for (const std::pair<void *, blender::ID *> &existing : instancer_recording->geometry_edges)
        {
          if (existing == edge) {
            known = true;
            break;
          }
        }
        if (!known) {
          instancer_recording->geometry_edges.push_back(edge);
        }
      }
    }
  }

  /* only interested in object that we can create geometry from */
  if (!object_is_geometry(b_ob_info)) {
    return nullptr;
  }

  /* Perform object culling. */
  if (object_is_light(b_ob)) {
    if (!show_lights) {
      return nullptr;
    }
  }
  else if (culling.test(scene, b_ob, tfm)) {
    return nullptr;
  }

  /* Visibility flags for both parent and child. */
  /* Note base_parent is null for objects from the background scene. */
  const blender::Base *base_parent = BKE_view_layer_base_find(&b_view_layer, b_parent);
  const bool use_holdout = (base_parent && (base_parent->flag & blender::BASE_HOLDOUT) != 0) ||
                           ((b_parent->visibility_flag & blender::OB_HOLDOUT) != 0);
  PathRayVisibility visibility = object_ray_visibility(b_ob);

  if (b_parent != &b_ob) {
    visibility &= object_ray_visibility(*b_parent);
  }

  /* TODO: make holdout objects on excluded layer invisible for non-camera rays. */
#if 0
  if (use_holdout && (layer_flag & view_layer.exclude_layer)) {
    visibility &= ~(PATH_RAY_VISIBILITY_ALL & ~PATH_RAY_VISIBILITY_CAMERA);
  }
#endif

  /* Clear camera visibility for indirect only objects. */
  const bool use_indirect_only = !use_holdout && base_parent &&
                                 ((base_parent->flag & blender::BASE_INDIRECT_ONLY) != 0);
  if (use_indirect_only) {
    visibility &= ~PATH_RAY_VISIBILITY_CAMERA;
  }

  /* Don't export completely invisible objects. */
  if (visibility == PATH_RAY_VISIBILITY_NONE) {
    return nullptr;
  }

  /* Use task pool only for non-instances, since sync_dupli_particle accesses
   * geometry. This restriction should be removed for better performance. */
  TaskPool *object_geom_task_pool = (is_instance) ? nullptr : geom_task_pool;

  /* key to lookup object */
  const ObjectKey key(b_parent, persistent_id, b_ob_info.real_object, use_particle_hair);
  Object *object;

  /* motion vector case */
  if (motion) {
    object = object_map.find(key);

    if (object && object->use_motion()) {
      /* Set transform at matching motion time step. */
      const int time_index = object->motion_step(motion_time);
      if (time_index >= 0) {
        object->set_motion_tfm(tfm, time_index);
      }

      /* mesh deformation */
      if (object->get_geometry()) {
        sync_geometry_motion(
            b_ob_info, object, motion_time, use_particle_hair, object_geom_task_pool);
      }
    }

    return object;
  }

  /* test if we need to sync */
  const bool object_is_new = object_map.add_or_update(&object, &b_ob.id, &b_parent->id, key);
  const bool transform_changed = !object->tfm_equals(tfm);
  bool object_updated = object_is_new || transform_changed;

  /* Counts how much of the instance list actually moves between frames. The whole walk is built on
   * visiting all 33858 of them every frame; if only a fraction changes, the cheaper walk is the
   * wrong answer and not walking the rest is the right one. */
  sync_stats_objects_seen++;
  if (transform_changed) {
    sync_stats_transform_changed++;
  }

  /* mesh sync */
  Geometry *geometry = sync_geometry(
      b_ob_info, object_updated, use_particle_hair, object_geom_task_pool);
  object->set_geometry(geometry);

  /* special case not tracked by object update flags */

  if (sync_object_attributes(b_ob, b_deg_iter_data, object)) {
    object_updated = true;
  }

  /* holdout */
  object->set_use_holdout(use_holdout);

  object->set_visibility(visibility);

  object->set_is_shadow_catcher((b_ob.visibility_flag & blender::OB_SHADOW_CATCHER) != 0 ||
                                (b_parent->visibility_flag & blender::OB_SHADOW_CATCHER) != 0);

  object->set_shadow_terminator_shading_offset(b_ob.shadow_terminator_shading_offset);

  object->set_shadow_terminator_geometry_offset(b_ob.shadow_terminator_geometry_offset);

  /* Reading these means resolving property names through RNA and walking up the parent chain, and
   * an instanced object arrives here once per instance. Computed once per source object per pass -
   * see `object_properties_cache`. */
  /* Keyed by the real object, never by `&b_ob`: for an instance the iterator hands back the very
   * same temporary object every time, so its address identifies the iterator, not the geometry -
   * every instance would read back whatever the first one happened to store. */
  auto cached_properties = object_properties_cache.find(b_real_object);

  if (cached_properties == object_properties_cache.end()) {
    CachedObjectProperties properties;

    blender::PointerRNA b_ob_rna_ptr = RNA_id_pointer_create(&b_ob.id);
    blender::PointerRNA cobject = RNA_pointer_get(&b_ob_rna_ptr, "cycles");

    properties.ao_distance = get_float(cobject, "ao_distance");
    properties.is_caustics_caster = get_boolean(cobject, "is_caustics_caster");
    properties.is_caustics_receiver = get_boolean(cobject, "is_caustics_receiver");

    /* the asset name for Cryptomatte */
    blender::Object *parent = b_ob.parent;
    if (parent) {
      while (parent->parent) {
        parent = parent->parent;
      }
      properties.asset_name = BKE_id_name(parent->id);
    }
    else {
      properties.asset_name = BKE_id_name(b_ob.id);
    }

    cached_properties = object_properties_cache.emplace(b_real_object, properties).first;
  }

  const CachedObjectProperties &properties = cached_properties->second;

  /* Falls back to the parent's distance, so it is cached against the parent rather than baked into
   * the object's entry - one geometry can be instanced under parents that disagree. */
  float ao_distance = properties.ao_distance;
  if (ao_distance == 0.0f && b_parent != &b_ob) {
    auto cached_parent_ao = parent_ao_distance_cache.find(b_parent);
    if (cached_parent_ao == parent_ao_distance_cache.end()) {
      blender::PointerRNA b_parent_rna_ptr = RNA_id_pointer_create(&b_parent->id);
      blender::PointerRNA cparent = RNA_pointer_get(&b_parent_rna_ptr, "cycles");
      cached_parent_ao =
          parent_ao_distance_cache.emplace(b_parent, get_float(cparent, "ao_distance")).first;
    }
    ao_distance = cached_parent_ao->second;
  }

  object->set_ao_distance(ao_distance);
  object->set_is_caustics_caster(properties.is_caustics_caster);
  object->set_is_caustics_receiver(properties.is_caustics_receiver);
  object->set_is_bake_target(b_ob_info.real_object == b_bake_target);
  object->set_asset_name(properties.asset_name);

  /* object sync
   * transform comparison should not be needed, but duplis don't work perfect
   * in the depsgraph and may not signal changes, so this is a workaround */
  const bool do_sync = object->is_modified() || object_updated ||
                       (object->get_geometry() && object->get_geometry()->is_modified());
  if (do_sync) {
    object->name = BKE_id_name(b_ob.id);
    object->set_pass_id(b_ob.index);
    const float *object_color = b_ob.color;
    object->set_color(make_float3(object_color[0], object_color[1], object_color[2]));
    object->set_alpha(object_color[3]);
    object->set_tfm(tfm);

    /* dupli texture coordinates and random_id */
    if (is_instance) {
      const float *orco = b_deg_iter_data.dupli_object_current->orco;
      object->set_dupli_generated(0.5f * make_float3(orco[0], orco[1], orco[2]) -
                                  make_float3(0.5f, 0.5f, 0.5f));
      const float *uv = b_deg_iter_data.dupli_object_current->uv;
      object->set_dupli_uv(make_float2(uv[0], uv[1]));
      object->set_random_id(b_deg_iter_data.dupli_object_current->random_id);
    }
    else {
      object->set_dupli_generated(zero_float3());
      object->set_dupli_uv(zero_float2());
      object->set_random_id(hash_uint2(hash_string(object->name.c_str()), 0));
    }

    /* Light group and linking. */
    string lightgroup = b_ob.lightgroup ? b_ob.lightgroup->name : "";
    if (lightgroup.empty()) {
      lightgroup = b_parent->lightgroup ? b_parent->lightgroup->name : "";
    }
    object->set_lightgroup(ustring(lightgroup));

    object->set_light_set_membership(BlenderLightLink::get_light_set_membership(b_parent, b_ob));
    object->set_receiver_light_set(BlenderLightLink::get_receiver_light_set(b_parent, b_ob));
    object->set_shadow_set_membership(BlenderLightLink::get_shadow_set_membership(b_parent, b_ob));
    object->set_blocker_shadow_set(BlenderLightLink::get_blocker_shadow_set(b_parent, b_ob));
  }

  sync_object_motion_init(*b_parent, b_ob, object);

  if (do_sync || object->motion_is_modified()) {
    object->tag_update(scene);
  }

  if (is_instance) {
    /* Sync possible particle data. */
    sync_dupli_particle(*b_parent, b_deg_iter_data, b_ob, object);
  }

  return object;
}

static float4 lookup_instance_property(blender::Object &ob,
                                       blender::DEGObjectIterData &b_deg_iter_data,
                                       const string &name,
                                       bool use_instancer)
{
  blender::DupliObject *dupli = nullptr;
  blender::Object *dupli_parent = nullptr;

  /* If requesting instance data, check the parent particle system and object. */
  if (use_instancer && b_deg_iter_data.dupli_object_current) {
    dupli = b_deg_iter_data.dupli_object_current;
    dupli_parent = b_deg_iter_data.dupli_parent;
  }

  float4 value;
  BKE_object_dupli_find_rgba_attribute(&ob, dupli, dupli_parent, name.c_str(), &value.x);

  return value;
}

bool BlenderSync::sync_object_attributes(blender::Object &b_ob,
                                         blender::DEGObjectIterData &b_deg_iter_data,
                                         Object *object)
{
  /* Find which attributes are needed. Collected once per geometry - see the cache declaration. */
  const Geometry *geometry = object->get_geometry();
  auto cached_requests = geometry_attribute_requests_cache.find(geometry);
  if (cached_requests == geometry_attribute_requests_cache.end()) {
    cached_requests = geometry_attribute_requests_cache
                          .emplace(geometry, object->get_geometry()->needed_attributes())
                          .first;
  }
  const AttributeRequestSet &requests = cached_requests->second;

  /* Delete attributes that became unnecessary. */
  vector<ParamValue> &attributes = object->attributes;
  bool changed = false;

  for (int i = attributes.size() - 1; i >= 0; i--) {
    if (!requests.find(attributes[i].name())) {
      attributes.erase(attributes.begin() + i);
      changed = true;
    }
  }

  /* Update attribute values. */
  for (const AttributeRequest &req : requests.requests) {
    const ustring name = req.name;

    std::string real_name;
    const int type = blender_attribute_name_split_type(name, &real_name);

    if (type == blender::SHD_ATTRIBUTE_OBJECT || type == blender::SHD_ATTRIBUTE_INSTANCER) {
      const bool use_instancer = (type == blender::SHD_ATTRIBUTE_INSTANCER);
      float4 value = lookup_instance_property(b_ob, b_deg_iter_data, real_name, use_instancer);

      /* Try finding the existing attribute value. */
      ParamValue *param = nullptr;

      for (size_t i = 0; i < attributes.size(); i++) {
        if (attributes[i].name() == name) {
          param = &attributes[i];
          break;
        }
      }

      /* Replace or add the value. */
      const ParamValue new_param(name, TypeFloat4, 1, &value);
      assert(new_param.datasize() == sizeof(value));

      if (!param) {
        changed = true;
        attributes.push_back(new_param);
      }
      else {
        /* Cannot use param->get<float4>, ParamValue storage is not guaranteed to be aligned. */
        const float *param_data = static_cast<const float *>(param->data());
        if (make_float4(param_data[0], param_data[1], param_data[2], param_data[3]) != value) {
          changed = true;
          *param = new_param;
        }
      }
    }
  }

  return changed;
}

/* Start the expensive half of copying a dirty mesh before the walk instead of after it.
 *
 * Copying one geometry out of Blender measures 5.0 ms on this scene, and 4.2 of those are a single
 * `corner_normals()` - Blender's lazily computed corner normals over 284646 corners, recomputed
 * because the mesh deformed. Three other geometries change in the same frame and cost 0.01 ms each,
 * so that one call is the whole of what `geom_task_pool.wait_work()` waits for after the walk.
 *
 * The walk only discovers that geometry at its very end, so its task overlaps the walk by about a
 * millisecond and the remaining four become the wait. `sync_recalc` knew which geometries were
 * dirty ten milliseconds earlier; this reads that list and asks for the same value from a worker
 * while the main thread walks.
 *
 * Nothing about the result changes. `SharedCache::ensure` computes with the same closure over the
 * same inputs and publishes once, so whoever asks second - here or `create_mesh` - reads exactly
 * the bytes the unmodified code would have computed. What changes is only which thread pays and
 * when. Note that this does not reduce the work: `normals_calc_corners` is already parallel inside,
 * so this moves an already-parallel 4.2 ms onto cores that were idle while the walk ran.
 *
 * Two things make this safe rather than merely fast, and both are load-bearing:
 *
 * Nothing is carried across frames except a session identifier. An evaluated pointer would not
 * survive: a frame this viewport refuses returns to Blender's event loop, and the next depsgraph
 * evaluation frees exactly what such a pointer would name. Carrying something has to happen,
 * though, because Cycles' own recalc flags accumulate across refused frames - the geometry repacked
 * on an accepted frame was usually marked dirty on an earlier refused one, and a list holding only
 * the last pass fired on about one accepted frame in four. An identifier resolves, here and now, to
 * whatever object currently carries it, or to nothing at all if it has been deleted.
 *
 * The ID iterator that filled the list runs with `only_updated`, and in that mode it skips the
 * visibility check that the object iterator makes - `DEG_iterator_ids_step` says so, and the object
 * iterator explains why: an object invisible since the last evaluation can have an evaluated mesh
 * that points at a freed data-block. Hence the two depsgraph gates below, which are what the object
 * iterator itself applies before handing an object out. */
void BlenderSync::warm_dirty_geometry_normals(blender::Depsgraph &b_depsgraph,
                                              TaskPool &geom_task_pool,
                                              const bool motion)
{
  /* Taken from the member, so the list is spent whether or not anything below accepts it - the
   * motion pass reads normals from a different place and under different conditions, and baking
   * has no viewport to be ahead of. */
  const set<uint32_t> candidates = std::move(warm_normals_candidates);
  warm_normals_candidates.clear();

  if (motion || b_bake_target != nullptr) {
    return;
  }

  const auto eval_mode = blender::DEG_get_mode(&b_depsgraph);

  /* Two objects sharing one mesh reach the list separately; the cache they want is the same one. */
  set<blender::Mesh *> warmed;

  /* Every gate below can silently reduce this to nothing, and a warm-up that never fires looks
   * exactly like one that fires and does not help. Counted per reason so the two are distinct. */
  sync_stats_warm_candidates = int(candidates.size());

  for (const uint32_t session_uid : candidates) {
    blender::ID *b_original_id = BKE_libblock_find_session_uid(
        b_data, blender::ID_OB, session_uid);
    if (b_original_id == nullptr) {
      /* Deleted since it was recorded. Which is the whole reason this is an identifier. */
      sync_stats_warm_gone++;
      continue;
    }

    blender::Object *b_ob = blender::DEG_get_evaluated(
        &b_depsgraph, blender::id_cast<blender::Object *>(b_original_id));
    if (b_ob == nullptr) {
      /* Still in the file, but no longer part of this depsgraph. */
      sync_stats_warm_gone++;
      continue;
    }

    if (!blender::DEG_object_geometry_is_evaluated(*b_ob)) {
      sync_stats_warm_unevaluated++;
      continue;
    }
    if (!blender::DEG_iterator_object_is_visible(eval_mode, b_ob)) {
      sync_stats_warm_invisible++;
      continue;
    }

    /* Any subdivision at all sends `sync_mesh` through `create_subd_mesh`, and the corner normals
     * are read inside the non-subdivision branch of `create_mesh` - so for those the warm-up would
     * compute something nobody asks for. */
    if (object_subdivision_type(*b_ob, preview, use_adaptive_subdivision) !=
        Mesh::SUBDIVISION_NONE)
    {
      sync_stats_warm_subdivided++;
      continue;
    }

    /* The same expression `sync_object` uses to fill `BObjectInfo::object_data`, so the mesh warmed
     * here is the one `create_mesh` will be handed. It calls `BKE_mesh_wrapper_ensure_subdivision`,
     * which mutates the mesh, which is why this stays on the main thread. */
    blender::ID *b_object_data = object_get_data(*b_ob, false);
    if (GS(b_object_data->name) != blender::ID_ME) {
      sync_stats_warm_not_mesh++;
      continue;
    }

    blender::Mesh *b_mesh = blender::id_cast<blender::Mesh *>(b_object_data);
    /* An edit-mode mesh is flushed into a fresh mesh by `object_to_mesh`, so the cache warmed on
     * this one would be read by nobody. */
    if (b_mesh->runtime->edit_mesh || b_mesh->faces_num == 0) {
      sync_stats_warm_not_mesh++;
      continue;
    }

    if (!warmed.insert(b_mesh).second) {
      continue;
    }
    sync_stats_normals_warmed++;

    geom_task_pool.push([b_mesh]() {
      /* `normals_domain()` is not cached, so this is its second run in the frame - still cheaper
       * than filling a corner-sized array for a mesh whose normals Cycles will take per vertex. */
      if (b_mesh->normals_domain() == blender::bke::MeshNormalDomain::Corner) {
        b_mesh->corner_normals();
      }
      /* Same kind of lazy cache, invalidated by the same deformation and read a few lines above the
       * normals in `create_mesh`. It measures near zero here, but it costs nothing to ask. */
      b_mesh->corner_tris();
    });
  }
}

/* Object Loop */

void BlenderSync::sync_objects(blender::Depsgraph &b_depsgraph,
                               blender::bScreen *b_screen,
                               blender::View3D *b_v3d,
                               const float motion_time)
{
  /* The walk and the pool wait are timed, but what surrounds them inside this function is not, and
   * that gap holds the maps: `pre_sync` clears a hash set of 33858 live nodes, `post_sync` walks a
   * tree of 33858 and looks each one up in it again. Timed from the top so the two ends of the
   * function stop being a matter of inference. */
  const double sync_objects_start_time = time_dt();

  /* Task pool for multithreaded geometry sync. */
  TaskPool geom_task_pool;

  /* layer data */
  const bool motion = motion_time != 0.0f;

  if (!motion) {
    /* prepare for sync */
    geometry_map.pre_sync();
    object_map.pre_sync();
    procedural_map.pre_sync();
    particle_system_map.pre_sync();
    motion_times.clear();
  }
  else {
    geometry_motion_synced.clear();
  }

  if (!motion) {
    /* Object to geometry instance mapping is built for the reference time, as other
     * times just look up the corresponding geometry. */
    instance_geometries_by_object.clear();
  }

  /* initialize culling */
  BlenderObjectCulling culling(scene, *b_scene);

  /* object loop */
  bool cancel = false;
  const bool show_lights =
      BlenderViewportParameters(b_screen, b_v3d, use_developer_ui).use_scene_lights;

  blender::ViewLayer &b_view_layer = *DEG_get_evaluated_view_layer(&b_depsgraph);

  BKE_view_layer_synced_ensure(*b_data, b_scene, &b_view_layer);

  blender::DEGObjectIterSettings deg_iter_settings{};
  deg_iter_settings.depsgraph = &b_depsgraph;
  deg_iter_settings.flags = DEG_OBJECT_ITER_FOR_RENDER_ENGINE_FLAGS;
  blender::DEGObjectIterData deg_iter_data{};
  deg_iter_data.settings = &deg_iter_settings;
  deg_iter_data.graph = deg_iter_settings.depsgraph;
  deg_iter_data.flag = deg_iter_settings.flags;
  /* Cycles places geometry by its object-to-world matrix and never reads the inverse or the
   * negative-scale flag, so the iterator can skip inverting a 4x4 for each of the instances. */
  deg_iter_data.need_inverse_matrix = false;

  const double walk_start_time = time_dt();
  int num_objects_seen = 0;

  /* Values read once per source object and reused for its instances, valid for this pass only. */
  object_properties_cache.clear();
  parent_ao_distance_cache.clear();
  geometry_attribute_requests_cache.clear();
  object_is_modified_cache.clear();
  object_has_particle_hair_cache.clear();
  sync_stats_objects_seen = 0;
  sync_stats_transform_changed = 0;
  sync_stats_geometry_dirty = 0;
  sync_stats_geometry_corners = 0;
  sync_stats_geometry_synced_inline = 0;
  sync_stats_groups = 0;
  sync_stats_groups_moved = 0;
  sync_stats_instances_in_moved_groups = 0;
  sync_stats_largest_group = 0;
  sync_stats_instances_in_groups = 0;
  sync_stats_groups_clean = 0;
  sync_stats_normals_warmed = 0;
  sync_stats_warm_candidates = 0;
  sync_stats_warm_gone = 0;
  sync_stats_warm_unevaluated = 0;
  sync_stats_warm_invisible = 0;
  sync_stats_warm_subdivided = 0;
  sync_stats_warm_not_mesh = 0;
  sync_stats_full_path_ms = 0.0;
  sync_stats_full_path_count = 0;

  warm_dirty_geometry_normals(b_depsgraph, geom_task_pool, motion);
  sync_stats_instances_in_clean_groups = 0;
  object_map.clear_stats();
  geometry_map.clear_stats();

  /* Whether groups may be skipped at all this frame. Not a per-group question: a change to any of
   * this is invisible to every object's recalc flags. Culling is excluded rather than folded into
   * the signature - it reads the camera and per-object switches, and only runs under Simplify,
   * which a viewport rarely has on. The motion pass never touches the cache: it runs after the
   * reference pass has filled it, and re-entering would only tear down what that pass built. */
  if (!motion) {
    const size_t globals = instancer_cache_signature(b_v3d, show_lights);
    const bool simplify = (b_scene->r.mode & blender::R_SIMPLIFY) != 0;
    use_instancer_cache = (globals == instancer_cache_globals) && !simplify &&
                          !shader_map.has_recalc();
    instancer_cache_globals = globals;
    if (!use_instancer_cache) {
      instancer_cache.clear();
    }
  }

  /* Instances arrive grouped by their instancer, so the group ends when the parent changes. */
  enum { GROUP_NONE, GROUP_RECORD, GROUP_VERIFY, GROUP_TAIL };
  blender::Object *group_parent = nullptr;
  int group_size = 0;
  int group_moved_at_start = 0;
  int group_mode = GROUP_NONE;
  InstancerCacheEntry *group_entry = nullptr;
  size_t group_step = 0;
  /* Survives the switch to GROUP_TAIL, which clears `group_entry`. The steps confirmed before that
   * switch still have to have their objects marked used, and `group_step` counts exactly those. */
  const InstancerCacheEntry *verified_entry = nullptr;

  auto close_group = [&]() {
    /* Marking the confirmed instances used, in one pass over the cached array rather than one at a
     * time from inside the iterator loop.
     *
     * The set of objects and the order are identical either way - the cache holds them in walk
     * order - so this is purely about memory access, and it measures 13.88 -> 12.85 ms on a scene
     * with 33392 instances in clean groups. A tight walk of an array of pointers is something the
     * processor can run ahead of; the same walk spread between the iterator's own work is not.
     *
     * Only the prefix `group_step` covers. An instance whose fingerprint did not match is not
     * confirmed, and neither are the ones after it - marking the whole group at entry would keep an
     * object that should have been collected, which is the one way this could put geometry on
     * screen that no longer belongs there. */
    if (verified_entry != nullptr) {
      const size_t confirmed = std::min(group_step, verified_entry->duplis.size());
      for (size_t i = 0; i < confirmed; i++) {
        const CachedDupli &cached = verified_entry->duplis[i];
        for (int j = 0; j < cached.num_objects; j++) {
          object_map.used(cached.objects[j]);
        }
      }
      verified_entry = nullptr;
    }

    if (group_size > 0) {
      sync_stats_groups++;
      sync_stats_instances_in_groups += group_size;
      if (group_size > sync_stats_largest_group) {
        sync_stats_largest_group = group_size;
      }
      if (sync_stats_transform_changed > group_moved_at_start) {
        sync_stats_groups_moved++;
        sync_stats_instances_in_moved_groups += group_size;
      }
      if (group_mode == GROUP_VERIFY) {
        sync_stats_groups_clean++;
        sync_stats_instances_in_clean_groups += group_size;
      }
    }

    if (group_mode == GROUP_VERIFY && group_step != group_entry->duplis.size()) {
      /* The group ran out of instances early. The objects behind the missing steps were never
       * marked used and will be collected, which is right - but the entry no longer describes the
       * group, so it goes and the next frame rebuilds it. */
      instancer_cache.erase(group_parent);
    }
    else if (group_mode == GROUP_TAIL) {
      instancer_cache.erase(group_parent);
    }
    else if (group_mode == GROUP_RECORD) {
      /* Collect what a skip will need to mark used, deduplicated once here rather than per
       * instance. Only ever runs for a group that had to be walked in full anyway. */
      set<Geometry *> seen_geometries;
      set<ParticleSystem *> seen_particle_systems;
      set<const blender::Object *> seen_sources;
      for (const CachedDupli &cached : group_entry->duplis) {
        if (seen_sources.insert(cached.fingerprint.ob).second) {
          group_entry->sources.push_back(const_cast<blender::Object *>(cached.fingerprint.ob));
        }
        for (int i = 0; i < cached.num_objects; i++) {
          Geometry *geometry = cached.objects[i]->get_geometry();
          if (geometry && seen_geometries.insert(geometry).second) {
            group_entry->geometries.push_back(geometry);
          }
          ParticleSystem *particle_system = cached.objects[i]->get_particle_system();
          if (particle_system && seen_particle_systems.insert(particle_system).second) {
            group_entry->particle_systems.push_back(particle_system);
          }
        }
      }

      /* Taken after `sources` is filled, since it covers them too. */
      group_entry->visibility_signature = instancer_visibility_signature(
          b_view_layer, *group_parent, group_entry->sources);
    }

    group_mode = GROUP_NONE;
    group_entry = nullptr;
    group_step = 0;
    instancer_recording = nullptr;
  };

  auto start_group = [&]() {
    if (!group_parent || motion) {
      return;
    }

    if (use_instancer_cache) {
      const auto it = instancer_cache.find(group_parent);
      /* A group that owns particle systems is never skipped, and the reason is worth stating
       * because the obvious fix is the wrong one.
       *
       * `sync_dupli_particle` rebuilds a system's particle array from scratch, and it decides
       * whether to clear it first by asking whether the system has been marked used yet this pass.
       * Entering the fast path marks every system of the group used up front, so if the group then
       * breaks - one fingerprint fails to match and the rest takes the full path - those instances
       * find the array already "in use" and append to the previous frame's contents instead of
       * replacing them. The array grows, `ObjectManager::device_update` computes particle offsets
       * from the grown host sizes, and the tail objects end up indexing past the device buffer.
       *
       * Moving the mark to where the confirmed prefix is marked does not fix it: the first tail
       * instance would then clear the array, and the prefix that was already skipped would keep
       * indices pointing at what is now the tail's data - silent corruption instead of loud. So the
       * group is simply walked in full. On this scene every instancer is Geometry Nodes and carries
       * no particle systems at all, so this costs 18 comparisons per frame and nothing else. */
      const bool skippable = it != instancer_cache.end() && it->second.particle_systems.empty();
      if (skippable && instancer_group_is_clean(b_view_layer, *group_parent, it->second)) {
        group_entry = &it->second;
        group_entry->seen_this_frame = true;
        group_mode = GROUP_VERIFY;
        /* Kept separately because the entry pointer is dropped the moment an instance fails to
         * match, and the instances confirmed before that still need marking. */
        verified_entry = group_entry;
        instancer_replay_shared(*group_entry);
        return;
      }
    }

    InstancerCacheEntry &entry = instancer_cache[group_parent];
    entry.duplis.clear();
    entry.geometries.clear();
    entry.particle_systems.clear();
    entry.sources.clear();
    entry.geometry_edges.clear();
    entry.seen_this_frame = true;
    group_entry = &entry;
    group_mode = GROUP_RECORD;
    instancer_recording = &entry;
  };

  /* Diagnostics: run the iterator but do no work per instance. This is deliberately placed where a
   * per-instance skip would go, so the walk it reports is the floor such a filter could reach - and
   * at the same time what the depsgraph iterator alone costs. Renders an empty scene; the numbers
   * are read from the log, not from the picture. */
  static const bool iterate_only = getenv("CYCLES_DEBUG_ITERATE_ONLY") != nullptr;

  /* The same idea one step finer, for the 5.3 ms that sit between that floor and the real walk.
   *
   * Timing the phases from inside would cost more than it measures: two clock reads per instance
   * across 33858 of them is over a millisecond of pure overhead on a 5 ms subject. So the phases
   * are separated by leaving them out instead, and the difference between two runs is the phase.
   *
   *   CYCLES_DEBUG_WALK_NO_COMPARE   fill the fingerprint, take the match on trust
   *   CYCLES_DEBUG_WALK_NO_FINGERPRINT   also skip filling it (implies the above)
   *
   * Subtracting gives the comparison and the fill separately. Marking the cached objects used stays
   * in at every level, and that is the whole point of this shape: dropping it does not remove work,
   * it moves it - the objects are then collected as unused and rebuilt next frame, which measured
   * slower than the code being measured. Both levels trust the depsgraph where the real path
   * verifies it, so they are for reading numbers out of the log, exactly like `iterate_only`. */
  static const bool walk_no_fingerprint = getenv("CYCLES_DEBUG_WALK_NO_FINGERPRINT") != nullptr;
  static const bool walk_no_compare = walk_no_fingerprint ||
                                      getenv("CYCLES_DEBUG_WALK_NO_COMPARE") != nullptr;

  ITER_BEGIN (blender::DEG_iterator_objects_begin,
              blender::DEG_iterator_objects_next,
              blender::DEG_iterator_objects_end,
              &deg_iter_data,
              blender::Object *,
              b_ob)
  {
    num_objects_seen++;

    /* The iterator sets `dupli_parent` before it yields the parent object itself, so membership is
     * decided by `dupli_object_current` - the parent is not part of its own group. */
    blender::Object *const parent_now = deg_iter_data.dupli_object_current ?
                                            deg_iter_data.dupli_parent :
                                            nullptr;
    if (parent_now != group_parent) {
      close_group();
      group_parent = parent_now;
      group_size = 0;
      group_moved_at_start = sync_stats_transform_changed;
      start_group();
    }
    if (parent_now) {
      group_size++;
    }

    if (iterate_only) {
      continue;
    }

    /* An instance of a group the depsgraph reported unchanged. Still compared against what it was
     * before it is believed: the recalc sets are a summary, and this is the thing itself. */
    size_t record_index = size_t(-1);
    if (group_mode == GROUP_VERIFY) {
      DupliFingerprint fingerprint;
      if (!walk_no_fingerprint) {
        instancer_fill_fingerprint(*deg_iter_data.dupli_object_current, fingerprint);
      }

      if (group_step < group_entry->duplis.size()) {
        const CachedDupli &cached = group_entry->duplis[group_step];
        /* Taking the match on trust is what makes this a measurement of the comparison rather than
         * of its consequences. */
        if (walk_no_compare ||
            memcmp(&cached.fingerprint, &fingerprint, sizeof(fingerprint)) == 0)
        {
          /* Not marked here: `close_group` marks the confirmed prefix in one pass. */
          group_step++;
          continue;
        }
      }

      /* Something moved that nothing reported. Everything skipped so far did match, so it stays
       * skipped; the rest of the group takes the full path and the entry is dropped. */
      group_mode = GROUP_TAIL;
      group_entry = nullptr;
    }
    else if (group_mode == GROUP_RECORD) {
      /* Recorded before the visibility tests below, so that a step which produces no object still
       * occupies its place - otherwise the next frame would compare instances against the wrong
       * entries and reject the whole group. */
      CachedDupli cached;
      instancer_fill_fingerprint(*deg_iter_data.dupli_object_current, cached.fingerprint);
      cached.objects[0] = nullptr;
      cached.objects[1] = nullptr;
      cached.num_objects = 0;
      record_index = group_entry->duplis.size();
      group_entry->duplis.push_back(cached);
    }

    /* Everything below runs only for an instance the fast path did not claim - 466 of the 33858 on
     * this scene. Two clock reads on that many is under twenty microseconds, so unlike the loop
     * above this part can simply be timed. It answers what the staged flags could not: how much of
     * the walk is the minority that really syncs, and how much is the majority just being walked
     * past. Ends where the iteration does, whichever `continue` gets there first. */
    static const bool report_full_path = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
    struct FullPathTimer {
      const bool &enabled;
      double &total;
      int &count;
      const double started;
      ~FullPathTimer()
      {
        if (enabled) {
          total += (time_dt() - started) * 1000.0;
          count++;
        }
      }
    } full_path_timer{report_full_path,
                      sync_stats_full_path_ms,
                      sync_stats_full_path_count,
                      report_full_path ? time_dt() : 0.0};

    /* Viewport visibility. */
    const bool show_in_viewport = !b_v3d || BKE_object_is_visible_in_viewport(b_v3d, b_ob);
    if (show_in_viewport == false) {
      continue;
    }

    /* Load per-object culling data. */
    culling.init_object(scene, *b_ob);

    const int ob_visibility = BKE_object_visibility(b_ob, deg_iter_data.eval_mode);

    /* Ensure the object geom supporting the hair is processed before adding
     * the hair processing task to the task pool, calling .to_mesh() on the
     * same object in parallel does not work. */
    /* Asked about the source object, not the one the iterator handed out. For an instance the
     * latter is `DEGObjectIterData::temp_dupli_object` - a single struct the iterator refills for
     * every dupli, so all 33858 of them share one address. The answer is cached by pointer, so
     * asking about that address would freeze whatever the first instance answered and apply it to
     * every instance of every instancer for the rest of the pass: a source with hair inheriting
     * `false` would never be marked used and would be swept, and one without would get an empty
     * `Hair` built for it. `use_particle_hair` is part of the object key, so this decides identity,
     * not just work. Same idiom as `object_properties_cache` next door, which keys on the real
     * object for exactly this reason. */
    blender::Object *b_hair_source = deg_iter_data.dupli_object_current ?
                                         deg_iter_data.dupli_object_current->ob :
                                         b_ob;
    const bool sync_hair = (ob_visibility & blender::OB_VISIBLE_PARTICLES) != 0 &&
                           object_has_particle_hair(b_hair_source);

    /* Kept with the step that produced them, so a later frame that skips this step knows which
     * objects to mark used. */
    auto record_object = [&](Object *object) {
      if (record_index == size_t(-1) || object == nullptr) {
        return;
      }
      CachedDupli &cached = group_entry->duplis[record_index];
      cached.objects[cached.num_objects++] = object;
    };

    /* Object itself. */
    if ((ob_visibility & blender::OB_VISIBLE_SELF) != 0) {
      record_object(sync_object(b_view_layer,
                                *b_ob,
                                deg_iter_data,
                                motion_time,
                                false,
                                show_lights,
                                culling,
                                sync_hair ? nullptr : &geom_task_pool));
    }

    /* Particle hair as separate object. */
    if (sync_hair) {
      record_object(sync_object(b_view_layer,
                                *b_ob,
                                deg_iter_data,
                                motion_time,
                                true,
                                show_lights,
                                culling,
                                &geom_task_pool));
    }

    cancel = progress.get_cancel();
    if (cancel) {
      break;
    }
  }
  ITER_END;

  close_group();

  /* The walk queues one task per changed geometry, so the wait is what the mesh copies cost beyond
   * the walk itself - the two are what `sync_objects_and_motion` is made of. */
  static const bool report_walk = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  const double walk_ms = report_walk ? (time_dt() - walk_start_time) * 1000.0 : 0.0;
  const double wait_start_time = report_walk ? time_dt() : 0.0;

  geom_task_pool.wait_work();

  const double walk_wait_ms = (time_dt() - wait_start_time) * 1000.0;

  progress.set_sync_status("");

  if (!cancel && !motion) {
    /* After object for world_use_portal. */
    sync_background_light(b_screen, b_v3d);

    /* An instancer that did not appear this frame is gone, and the nodes its entry points at are
     * about to be deleted below. Dropping the entry first is what keeps those pointers from
     * outliving what they point to. A cancelled walk leaves the flags as they are and skips this,
     * together with the collection it guards. */
    for (auto it = instancer_cache.begin(); it != instancer_cache.end();) {
      if (it->second.seen_this_frame) {
        it->second.seen_this_frame = false;
        ++it;
      }
      else {
        it = instancer_cache.erase(it);
      }
    }

    /* Handle removed data and modified pointers, as this may free memory, delete Nodes in the
     * right order to ensure that dependent data is freed after their users. Objects should be
     * freed before particle systems and geometries. */
    object_map.post_sync();
    geometry_map.post_sync();
    particle_system_map.post_sync();
    procedural_map.post_sync();
  }

  const double sync_objects_tail_ms = report_walk ? (time_dt() - wait_start_time) * 1000.0 -
                                                        walk_wait_ms :
                                                    0.0;

  /* Reported after the collection, not before: deletions only happen there, and a count printed
   * ahead of it would be the previous frame's. */
  if (report_walk) {
    fprintf(stderr,
            "SYNC_OBJECTS head=%.2f walk=%.2f wait=%.2f tail=%.2f began=%.2f objects=%d synced=%d "
            "moved=%d "
            "geom_dirty=%d geom_corners=%d geom_inline=%d "
            "warmed=%d warm_cand=%d warm_gone=%d warm_uneval=%d warm_invis=%d warm_subd=%d "
            "warm_notmesh=%d full_path_ms=%.2f full_path_n=%d "
            "groups=%d groups_moved=%d in_moved=%d largest=%d in_groups=%d clean=%d in_clean=%d "
            "ob_new=%d ob_gone=%d geom_new=%d geom_gone=%d\n",
            (walk_start_time - sync_objects_start_time) * 1000.0,
            walk_ms,
            walk_wait_ms,
            sync_objects_tail_ms,
            /* Same clock as `GEOM_TASK began=`, so the two lines can be laid against each other:
             * whether a packing task overlapped the walk or only started once it ended is what
             * decides between splitting the copy and queueing it sooner. */
            walk_start_time * 1000.0,
            num_objects_seen,
            sync_stats_objects_seen,
            sync_stats_transform_changed,
            sync_stats_geometry_dirty,
            sync_stats_geometry_corners,
            sync_stats_geometry_synced_inline,
            sync_stats_normals_warmed,
            sync_stats_warm_candidates,
            sync_stats_warm_gone,
            sync_stats_warm_unevaluated,
            sync_stats_warm_invisible,
            sync_stats_warm_subdivided,
            sync_stats_warm_not_mesh,
            sync_stats_full_path_ms,
            sync_stats_full_path_count,
            sync_stats_groups,
            sync_stats_groups_moved,
            sync_stats_instances_in_moved_groups,
            sync_stats_largest_group,
            sync_stats_instances_in_groups,
            sync_stats_groups_clean,
            sync_stats_instances_in_clean_groups,
            object_map.stats_created,
            object_map.stats_deleted,
            geometry_map.stats_created,
            geometry_map.stats_deleted);
    fflush(stderr);
  }

  if (motion) {
    geometry_motion_synced.clear();
  }
}

void BlenderSync::sync_objects_and_motion(blender::RenderData &b_render,
                                          blender::Depsgraph &b_depsgraph,
                                          blender::bScreen *b_screen,
                                          blender::View3D *b_v3d,
                                          blender::RegionView3D *b_rv3d,
                                          const int width,
                                          const int height,
                                          void **python_thread_state)
{
  /* get camera object here to deal with camera switch */
  blender::Object *b_cam = get_camera_object(b_v3d, b_rv3d);

  const int frame_center = b_scene->r.cfra;
  const float subframe_center = b_scene->r.subframe;
  float frame_center_delta = 0.0f;

  if (scene->need_motion() == Scene::MOTION_BLUR &&
      scene->camera->get_motion_position() != MOTION_POSITION_CENTER)
  {
    const float shuttertime = scene->camera->get_shuttertime();
    if (scene->camera->get_motion_position() == MOTION_POSITION_END) {
      frame_center_delta = -shuttertime * 0.5f;
    }
    else {
      assert(scene->camera->get_motion_position() == MOTION_POSITION_START);
      frame_center_delta = shuttertime * 0.5f;
    }

    const float time = frame_center + subframe_center + frame_center_delta;
    const int frame = (int)floorf(time);
    const float subframe = time - frame;
    python_thread_state_restore(python_thread_state);
    RE_engine_frame_set(b_engine, frame, subframe);
    python_thread_state_save(python_thread_state);
    if (b_cam) {
      sync_camera_motion(b_render, b_cam, width, height, 0.0f);
    }
  }

  sync_objects(b_depsgraph, b_screen, b_v3d);

  /* In the viewport, only motion between previous frame and current frame is of interest, which is
   * kept updated separately. */
  if (b_v3d) {
    assert(scene->need_motion() == Scene::MOTION_NONE ||
           scene->need_motion() == Scene::MOTION_PASS_INTERACTIVE);
    return;
  }

  if (scene->need_motion() == Scene::MOTION_NONE) {
    return;
  }

  /* Insert motion times from camera. Motion times from other objects
   * have already been added in a sync_objects call. */
  if (b_cam) {
    const uint camera_motion_steps = object_motion_steps(*b_cam, *b_cam);
    for (size_t step = 0; step < camera_motion_steps; step++) {
      motion_times.insert(scene->camera->motion_time(step));
    }
  }

  /* Check which geometry already has motion blur so it can be skipped. */
  geometry_motion_attribute_synced.clear();
  for (Geometry *geom : scene->geometry) {
    const Attribute *attr_P = geom->attributes.find(ATTR_STD_POSITION);
    if (attr_P && attr_P->has_motion()) {
      geometry_motion_attribute_synced.insert(geom);
    }
  }

  /* note iteration over motion_times set happens in sorted order */
  for (const float relative_time : motion_times) {
    /* center time is already handled. */
    if (relative_time == 0.0f) {
      continue;
    }

    LOG_DEBUG << "Synchronizing motion for the relative time " << relative_time << ".";

    /* fixed shutter time to get previous and next frame for motion pass */
    const float shuttertime = scene->motion_shutter_time();

    /* compute frame and subframe time */
    const float time = frame_center + subframe_center + frame_center_delta +
                       relative_time * shuttertime * 0.5f;
    const int frame = (int)floorf(time);
    const float subframe = time - frame;

    /* change frame */
    python_thread_state_restore(python_thread_state);
    RE_engine_frame_set(b_engine, frame, subframe);
    python_thread_state_save(python_thread_state);

    /* Syncs camera motion if relative_time is one of the camera's motion times. */
    sync_camera_motion(b_render, b_cam, width, height, relative_time);

    /* sync object */
    sync_objects(b_depsgraph, b_screen, b_v3d, relative_time);
  }

  geometry_motion_attribute_synced.clear();

  /* we need to set the python thread state again because this
   * function assumes it is being executed from python and will
   * try to save the thread state */
  python_thread_state_restore(python_thread_state);
  RE_engine_frame_set(b_engine, frame_center, subframe_center);
  python_thread_state_save(python_thread_state);
}

CCL_NAMESPACE_END
