/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <atomic>

#include "bvh/bvh.h"

#include "device/device.h"

#include "scene/attribute.h"
#include "scene/camera.h"
#include "scene/geometry.h"
#include "scene/hair.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/osl.h"
#include "scene/pointcloud.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_nodes.h"
#include "scene/stats.h"
#include "scene/volume.h"

#include "subd/split.h"

#ifdef WITH_OSL
#  include "kernel/osl/globals.h"
#endif

#include "util/log.h"
#include "util/progress.h"
#include "util/task.h"

CCL_NAMESPACE_BEGIN

/* Geometry */

NODE_ABSTRACT_DEFINE(Geometry)
{
  NodeType *type = NodeType::add("geometry_base", nullptr);

  SOCKET_UINT(motion_steps, "Motion Steps", 0);
  SOCKET_BOOLEAN(use_motion_blur, "Use Motion Blur", false);
  SOCKET_NODE_ARRAY(used_shaders, "Shaders", Shader::get_node_type());

  return type;
}

Geometry::Geometry(const NodeType *node_type, const Type type)
    : Node(node_type), geometry_type(type), attributes(this, ATTR_PRIM_GEOMETRY)
{
  need_update_rebuild = false;
  need_update_bvh_for_offset = false;
  /* Newly created geometry has no previous-frame positions yet. */
  need_update_interactive_motion = true;

  transform_applied = false;
  transform_negative_scaled = false;
  transform_normal = transform_identity();
  bounds = BoundBox::empty;

  has_volume = false;
  has_surface_bssrdf = false;

  attr_map_offset = 0;
  prim_offset = 0;
}

Geometry::~Geometry()
{
  dereference_all_used_nodes();
}

void Geometry::clear(bool preserve_shaders)
{
  if (!preserve_shaders) {
    used_shaders.clear();
  }

  transform_applied = false;
  transform_negative_scaled = false;
  transform_normal = transform_identity();
  tag_modified();
}

const packed_float3 *Geometry::get_position() const
{
  const Attribute *attr = attributes.find(ATTR_STD_POSITION);
  return attr ? attr->data<packed_float3>() : nullptr;
}

packed_float3 *Geometry::get_position_for_write()
{
  Attribute *attr = attributes.add(ATTR_STD_POSITION);
  attr->modified = true;
  need_update_interactive_motion = true;
  tag_modified();
  return attr->data_for_write<packed_float3>();
}

void Geometry::tag_position_modified()
{
  Attribute *attr = attributes.add(ATTR_STD_POSITION);
  attr->modified = true;
  need_update_interactive_motion = true;
  tag_modified();
}

bool Geometry::position_is_modified() const
{
  Attribute *attr = attributes.find(ATTR_STD_POSITION);
  return (attr) ? attr->modified : false;
}

const float *Geometry::get_radius() const
{
  const Attribute *attr = attributes.find(ATTR_STD_RADIUS);
  return attr ? attr->data<float>() : nullptr;
}

float *Geometry::get_radius_for_write()
{
  Attribute *attr = attributes.add(ATTR_STD_RADIUS);
  attr->modified = true;
  tag_modified();
  return attr->data_for_write<float>();
}

void Geometry::tag_radius_modified()
{
  Attribute *attr = attributes.add(ATTR_STD_RADIUS);
  attr->modified = true;
  tag_modified();
}

bool Geometry::radius_is_modified() const
{
  Attribute *attr = attributes.find(ATTR_STD_RADIUS);
  return (attr) ? attr->modified : false;
}

float Geometry::motion_time(const int step) const
{
  return (motion_steps > 1) ? 2.0f * step / (motion_steps - 1) - 1.0f : 0.0f;
}

int Geometry::motion_step(const float time) const
{
  if (motion_steps > 1) {
    int attr_step = 0;

    for (int step = 0; step < motion_steps; step++) {
      const float step_time = motion_time(step);
      if (step_time == time) {
        return attr_step;
      }

      /* Center step is stored in a separate attribute. */
      if (step != motion_steps / 2) {
        attr_step++;
      }
    }
  }

  return -1;
}

bool Geometry::need_build_bvh(BVHLayout layout) const
{
  return is_instanced() || layout == BVH_LAYOUT_OPTIX || layout == BVH_LAYOUT_MULTI_OPTIX ||
         layout == BVH_LAYOUT_METAL || layout == BVH_LAYOUT_MULTI_OPTIX_EMBREE ||
         layout == BVH_LAYOUT_MULTI_METAL || layout == BVH_LAYOUT_MULTI_METAL_EMBREE ||
         layout == BVH_LAYOUT_HIPRT || layout == BVH_LAYOUT_MULTI_HIPRT ||
         layout == BVH_LAYOUT_MULTI_HIPRT_EMBREE || layout == BVH_LAYOUT_EMBREEGPU ||
         layout == BVH_LAYOUT_MULTI_EMBREEGPU || layout == BVH_LAYOUT_MULTI_EMBREEGPU_EMBREE;
}

bool Geometry::is_instanced() const
{
  /* Currently we treat subsurface objects as instanced.
   *
   * While it might be not very optimal for ray traversal, it avoids having
   * duplicated BVH in the memory, saving quite some space.
   */
  return !transform_applied || has_surface_bssrdf;
}

bool Geometry::has_true_displacement() const
{
  for (Node *node : used_shaders) {
    Shader *shader = static_cast<Shader *>(node);
    if (shader->has_displacement && shader->get_displacement_method() != DISPLACE_BUMP) {
      return true;
    }
  }

  return false;
}

bool Geometry::has_motion_blur() const
{
  if (!use_motion_blur) {
    return false;
  }
  const Attribute *attr_P = attributes.find(ATTR_STD_POSITION);
  return attr_P && attr_P->has_motion();
}

bool Geometry::has_emission_shader() const
{
  for (Node *node : used_shaders) {
    const Shader *shader = static_cast<const Shader *>(node);
    if (shader->emission_sampling != EMISSION_SAMPLING_NONE) {
      return true;
    }
  }

  return false;
}

void Geometry::tag_update(Scene *scene, bool rebuild)
{
  /* The sync paths replace whole attribute buffers before tagging, so the positions this geometry
   * carries may differ from the ones the interactive motion pass last recorded. */
  need_update_interactive_motion = true;

  if (rebuild) {
    need_update_rebuild = true;

    if (has_emission_shader()) {
      scene->light_manager->tag_update(scene, LightManager::MESH_NEED_REBUILD);
    }
    else {
      /* A geometry that emits nothing contributes no emitter to the light tree, so its own topology
       * is of no interest there. What is of interest is that changing a triangle count shifts the
       * primitive offsets of everything packed after it, and the tree bakes those offsets into its
       * emitters (`kemitter.triangle.id = prim_id + mesh->prim_offset`). Hence the unconditional
       * wake-up this replaces - it was not about emission at all, it was about the offsets.
       *
       * Whether the offsets actually moved is not knowable here: they are computed later, in
       * `geom_calc_offset`, which already tracks it per geometry. So the question is recorded and
       * answered there, before the light manager runs. On this scene four geometries are rebuilt
       * every frame while their sizes stay put, which is 5.7 ms of light tree per frame spent
       * rebuilding around offsets that never moved. */
      scene->geometry_manager->tag_light_needs_offset_check();
    }
  }
  else if (has_emission_shader()) {
    scene->light_manager->tag_update(scene, LightManager::EMISSIVE_MESH_MODIFIED);
  }

  scene->geometry_manager->tag_update(scene, GeometryManager::GEOMETRY_MODIFIED);
}

/* Geometry Manager */

GeometryManager::GeometryManager()
{
  update_flags = UPDATE_ALL;
  need_flags_update = true;
}

GeometryManager::~GeometryManager() = default;

void GeometryManager::update_interactive_motion(Scene *scene)
{
  std::atomic<bool> update{false};

  parallel_for(blocked_range<size_t>(0, scene->geometry.size(), 32),
               [&](const blocked_range<size_t> &r) {
                 for (size_t i = r.begin(); i != r.end(); i++) {
                   Geometry *geom = scene->geometry[i];

                   /* Nothing wrote positions since the last swap, so the comparison below can only
                    * come out equal. Skipping it avoids reading every vertex of every static
                    * geometry on every scene update - the pointer shortcut further down never
                    * fires here, because the motion steps live in their own buffer. */
                   if (!geom->need_update_interactive_motion) {
                     continue;
                   }
                   geom->need_update_interactive_motion = false;

                   Attribute *attr_P = geom->attributes.find(ATTR_STD_POSITION);
                   if (attr_P == nullptr || !attr_P->has_motion()) {
                     continue;
                   }

                   /* Compares pointers in case center and prev are implicitly shared. */
                   const void *center = attr_P->data();
                   const void *prev = attr_P->data(1);
                   if (center == prev ||
                       std::memcmp(center, prev, attr_P->data_sizeof() * attr_P->size) == 0) {
                     continue;
                   }

                   if (geom->is_mesh()) {
                     static_cast<Mesh *>(geom)->copy_center_to_motion_step(0);
                   }
                   else if (geom->is_hair()) {
                     static_cast<Hair *>(geom)->copy_center_to_motion_step(0);
                   }
                   else if (geom->is_pointcloud()) {
                     static_cast<PointCloud *>(geom)->copy_center_to_motion_step(0);
                   }
                   attr_P->modified = true;
                   update.store(true, std::memory_order_relaxed);
                 }
               });

  if (update.load(std::memory_order_relaxed)) {
    /* The advanced positions are already marked modified, so the next genuine scene update uploads
     * them along with everything else it packs - which is what happens on every frame of playback.
     * Tagging the manager here would instead cascade through the object manager into the light
     * manager and buy a second full scene update per frame, so the flush is deferred until the
     * scene actually goes idle. Deferring it is safe: the intermediate zero-motion state is never
     * what a frame needs, only the settled one is, and that is what the idle flush delivers. */
    motion_history_pending = true;
  }
}

void GeometryManager::update_osl_globals(Device *device, Scene *scene)
{
#ifdef WITH_OSL
  OSLGlobals *og = device->get_cpu_osl_memory();
  if (og == nullptr) {
    /* Can happen when rendering with multiple GPUs, but no CPU (in which case the name maps filled
     * below are not used anyway) */
    return;
  }

  og->object_name_map.clear();
  og->object_names.clear();

  for (size_t i = 0; i < scene->objects.size(); i++) {
    /* set object name to object index map */
    Object *object = scene->objects[i];
    og->object_name_map[object->name] = i;
    og->object_names.push_back(object->name);
  }
#else
  (void)device;
  (void)scene;
#endif
}

static void update_device_flags_attribute(uint32_t &device_update_flags,
                                          const AttributeSet &attributes)
{
  for (const Attribute &attr : attributes.attributes) {
    if (!attr.modified) {
      continue;
    }

    const AttrKernelDataType kernel_type = Attribute::kernel_type(attr);

    switch (kernel_type) {
      case AttrKernelDataType::FLOAT: {
        device_update_flags |= ATTR_FLOAT_MODIFIED;
        break;
      }
      case AttrKernelDataType::FLOAT2: {
        device_update_flags |= ATTR_FLOAT2_MODIFIED;
        break;
      }
      case AttrKernelDataType::FLOAT3: {
        device_update_flags |= ATTR_FLOAT3_MODIFIED;
        break;
      }
      case AttrKernelDataType::FLOAT4: {
        device_update_flags |= ATTR_FLOAT4_MODIFIED;
        break;
      }
      case AttrKernelDataType::UCHAR4: {
        device_update_flags |= ATTR_UCHAR4_MODIFIED;
        break;
      }
      case AttrKernelDataType::NORMAL: {
        device_update_flags |= ATTR_NORMAL_MODIFIED;
        break;
      }
      case AttrKernelDataType::NUM: {
        break;
      }
    }
  }
}

static void update_attribute_realloc_flags(uint32_t &device_update_flags,
                                           const AttributeSet &attributes)
{
  if (attributes.modified(AttrKernelDataType::FLOAT)) {
    device_update_flags |= ATTR_FLOAT_NEEDS_REALLOC;
  }
  if (attributes.modified(AttrKernelDataType::FLOAT2)) {
    device_update_flags |= ATTR_FLOAT2_NEEDS_REALLOC;
  }
  if (attributes.modified(AttrKernelDataType::FLOAT3)) {
    device_update_flags |= ATTR_FLOAT3_NEEDS_REALLOC;
  }
  if (attributes.modified(AttrKernelDataType::FLOAT4)) {
    device_update_flags |= ATTR_FLOAT4_NEEDS_REALLOC;
  }
  if (attributes.modified(AttrKernelDataType::UCHAR4)) {
    device_update_flags |= ATTR_UCHAR4_NEEDS_REALLOC;
  }
  if (attributes.modified(AttrKernelDataType::NORMAL)) {
    device_update_flags |= ATTR_NORMAL_NEEDS_REALLOC;
  }
}

void GeometryManager::geom_calc_offset(Scene *scene, BVHLayout bvh_layout)
{
  size_t tri_size = 0;

  size_t curve_size = 0;
  size_t curve_segment_size = 0;

  size_t point_size = 0;

  size_t face_size = 0;
  size_t corner_size = 0;

  for (Geometry *geom : scene->geometry) {
    bool prim_offset_changed = false;

    if (geom->is_mesh() || geom->is_volume()) {
      Mesh *mesh = static_cast<Mesh *>(geom);

      prim_offset_changed = (mesh->prim_offset != tri_size);

      mesh->prim_offset = tri_size;

      mesh->face_offset = face_size;
      mesh->corner_offset = corner_size;

      tri_size += mesh->num_triangles();

      face_size += mesh->get_num_subd_faces();
      corner_size += mesh->subd_face_corners.size();
    }
    else if (geom->is_hair()) {
      Hair *hair = static_cast<Hair *>(geom);

      prim_offset_changed = (hair->curve_segment_offset != curve_segment_size);
      hair->curve_segment_offset = curve_segment_size;
      hair->prim_offset = curve_size;

      curve_size += hair->num_curves();
      curve_segment_size += hair->num_segments();
    }
    else if (geom->is_pointcloud()) {
      PointCloud *pointcloud = static_cast<PointCloud *>(geom);

      prim_offset_changed = (pointcloud->prim_offset != point_size);

      pointcloud->prim_offset = point_size;
      point_size += pointcloud->num_points();
    }

    if (prim_offset_changed) {
      /* Need to rebuild BVH in OptiX, since refit only allows modified mesh data.
       * Metal has optimization for static BVH, that also require a rebuild. */
      const bool need_update_rebuild = (bvh_layout == BVH_LAYOUT_OPTIX ||
                                        bvh_layout == BVH_LAYOUT_MULTI_OPTIX ||
                                        bvh_layout == BVH_LAYOUT_MULTI_OPTIX_EMBREE) ||
                                       ((bvh_layout == BVH_LAYOUT_METAL ||
                                         bvh_layout == BVH_LAYOUT_MULTI_METAL ||
                                         bvh_layout == BVH_LAYOUT_MULTI_METAL_EMBREE) &&
                                        scene->params.bvh_type == BVH_TYPE_STATIC);
      geom->need_update_rebuild |= need_update_rebuild;
      geom->need_update_bvh_for_offset = true;

      /* This is the only place a geometry's primitive offset moves, so it is the only thing that
       * can invalidate the per-object primitive offset array. Raised here rather than from every
       * tag of the object manager, which is what made that array rebuild on every frame. The
       * order allows it: `device_update_prim_offsets` runs after the geometry manager. */
      scene->object_manager->tag_prim_offsets_modified();

      /* Same reasoning for the light tree, which bakes these offsets into its emitters. A rebuilt
       * emitting geometry woke the light manager directly; a rebuilt non-emitting one only had to
       * if it moved somebody's offsets, and here is where that turns out to be true. Answered for
       * whichever geometry moved first - one shifted offset is enough to invalidate the tree. */
      if (light_needs_offset_check_) {
        scene->light_manager->tag_update(scene, LightManager::MESH_NEED_REBUILD);
        light_needs_offset_check_ = false;
      }
    }
  }

  /* No offset moved, so the deferred question is answered in the negative and does not carry into
   * the next update - by then it would be about a rebuild that already settled. */
  light_needs_offset_check_ = false;
}

void GeometryManager::device_update_preprocess(Device *device, Scene *scene, Progress &progress)
{
  if (!need_update() && !need_flags_update) {
    return;
  }

  uint32_t device_update_flags = 0;

  const scoped_callback_timer timer([scene](double time) {
    if (scene->update_stats) {
      scene->update_stats->geometry.times.add_entry({"device_update_preprocess", time});
    }
  });

  progress.set_status("Updating Meshes Flags");

  /* Update flags. */
  bool volume_images_updated = false;

  for (Geometry *geom : scene->geometry) {
    const bool prev_has_volume = geom->has_volume;
    geom->has_volume = false;

    update_attribute_realloc_flags(device_update_flags, geom->attributes);

    if (geom->is_mesh()) {
      Mesh *mesh = static_cast<Mesh *>(geom);
      update_attribute_realloc_flags(device_update_flags, mesh->subd_attributes);
    }

    for (Node *node : geom->get_used_shaders()) {
      Shader *shader = static_cast<Shader *>(node);
      if (shader->has_volume) {
        geom->has_volume = true;
      }

      if (shader->has_surface_bssrdf) {
        geom->has_surface_bssrdf = true;
      }

      if (shader->need_update_uvs) {
        device_update_flags |= ATTR_FLOAT2_NEEDS_REALLOC;

        /* Attributes might need to be tessellated if added. */
        if (geom->is_mesh()) {
          Mesh *mesh = static_cast<Mesh *>(geom);
          if (mesh->need_tesselation()) {
            mesh->tag_modified();
          }
        }
      }

      if (shader->need_update_attribute) {
        device_update_flags |= ATTRS_NEED_REALLOC;

        /* Attributes might need to be tessellated if added. */
        if (geom->is_mesh()) {
          Mesh *mesh = static_cast<Mesh *>(geom);
          if (mesh->need_tesselation()) {
            mesh->tag_modified();
          }
        }
      }

      if (shader->need_update_displacement) {
        /* tag displacement related sockets as modified */
        if (geom->is_mesh()) {
          Mesh *mesh = static_cast<Mesh *>(geom);
          mesh->tag_position_modified();
          mesh->tag_subd_dicing_rate_modified();
          mesh->tag_subd_max_level_modified();
          mesh->tag_subd_objecttoworld_modified();

          device_update_flags |= ATTRS_NEED_REALLOC;
        }
      }

      if (geom->is_hair()) {
        if (shader->shadow_transparency_needs_realloc) {
          device_update_flags |= ATTR_FLOAT_NEEDS_REALLOC;
        }
        if (shader->need_update_shadow_transparency) {
          Attribute *attr = geom->attributes.find(ATTR_STD_SHADOW_TRANSPARENCY);
          if (attr) {
            attr->modified = true;
          }
        }
      }
    }

    /* only check for modified attributes if we do not need to reallocate them already */
    if ((device_update_flags & ATTRS_NEED_REALLOC) == 0) {
      update_device_flags_attribute(device_update_flags, geom->attributes);
      /* don't check for subd_attributes, as if they were modified, we would need to reallocate
       * anyway */
    }

    /* Re-create volume mesh if we will rebuild or refit the BVH. Note we
     * should only do it in that case, otherwise the BVH and mesh can go
     * out of sync. */
    if (geom->is_volume() && (geom->is_modified() || (update_flags & VOLUME_MODIFIED))) {
      /* Create volume meshes if there is voxel data. */
      if (!volume_images_updated) {
        progress.set_status("Updating Meshes Volume Bounds");
        device_update_volume_images(device, scene, progress);
        volume_images_updated = true;
      }

      Volume *volume = static_cast<Volume *>(geom);
      create_volume_mesh(scene, volume, progress);

      /* always reallocate when we have a volume, as we need to rebuild the BVH */
      device_update_flags |= DEVICE_MESH_DATA_NEEDS_REALLOC;
    }

    if (geom->has_volume) {
      if (geom->is_modified()) {
        scene->volume_manager->tag_update({geom});
      }
      if (!prev_has_volume) {
        scene->volume_manager->tag_update();
      }
    }
    else if (prev_has_volume) {
      scene->volume_manager->tag_update({geom});
    }

    if (geom->is_hair()) {
      Hair *hair = static_cast<Hair *>(geom);

      if (hair->need_update_rebuild) {
        device_update_flags |= DEVICE_CURVE_DATA_NEEDS_REALLOC;
      }
      else if (hair->is_modified()) {
        device_update_flags |= DEVICE_CURVE_DATA_MODIFIED;
      }
    }

    if (geom->is_mesh()) {
      Mesh *mesh = static_cast<Mesh *>(geom);

      if (mesh->need_update_rebuild) {
        device_update_flags |= DEVICE_MESH_DATA_NEEDS_REALLOC;
      }
      else if (mesh->is_modified()) {
        device_update_flags |= DEVICE_MESH_DATA_MODIFIED;
      }
    }

    if (geom->is_pointcloud()) {
      PointCloud *pointcloud = static_cast<PointCloud *>(geom);

      if (pointcloud->need_update_rebuild) {
        device_update_flags |= DEVICE_POINT_DATA_NEEDS_REALLOC;
      }
      else if (pointcloud->is_modified()) {
        device_update_flags |= DEVICE_POINT_DATA_MODIFIED;
      }
    }
  }

  if (update_flags & (MESH_ADDED | MESH_REMOVED | MOTION_PASS_NEEDED)) {
    device_update_flags |= DEVICE_MESH_DATA_NEEDS_REALLOC;
  }

  if (update_flags & (HAIR_ADDED | HAIR_REMOVED | MOTION_PASS_NEEDED)) {
    device_update_flags |= DEVICE_CURVE_DATA_NEEDS_REALLOC;
  }

  if (update_flags & (POINT_ADDED | POINT_REMOVED | MOTION_PASS_NEEDED)) {
    device_update_flags |= DEVICE_POINT_DATA_NEEDS_REALLOC;
  }

  /* Decide whether the scene-wide arrays can keep their allocation.
   *
   * `need_update_rebuild` on a single geometry currently marks every array in the scene for
   * reallocation, so a scene whose geometry is driven by modifiers re-uploads all of it every
   * frame. When the geometries sit in the same order at the same offsets with the same sizes,
   * the allocation is still valid and only the rewritten slices need to go up.
   *
   * The BVH keeps its previous, conservative behaviour: `scene->bvh.reset()` and the `bvh_*` /
   * `prim_*` arrays are still invalidated whenever they were before. Keeping a stale BVH across an
   * object removal would leave dangling `Object *` in it and crash the acceleration structure
   * build, and it only costs about three milliseconds anyway. */
  const uint32_t geometry_realloc_causes = device_update_flags &
                                           (DEVICE_MESH_DATA_NEEDS_REALLOC |
                                            DEVICE_CURVE_DATA_NEEDS_REALLOC |
                                            DEVICE_POINT_DATA_NEEDS_REALLOC);
  bool keep_geometry_allocation = false;
  if (geometry_realloc_causes != 0) {
    /* Adding or removing geometry is excluded outright: a newly created geometry starts with
     * `prim_offset == 0`, so a removal plus an addition of the same size would look like an
     * unchanged layout. `erase_by_swap` can also reorder the list. */
    const uint32_t membership_changed = update_flags &
                                        (MESH_ADDED | MESH_REMOVED | HAIR_ADDED | HAIR_REMOVED |
                                         POINT_ADDED | POINT_REMOVED);
    if (membership_changed != 0) {
      layout_reuse_rejected_reason_ = "geometry added or removed";
    }
    else {
      const DeviceGeometryLayout layout = compute_layout(scene);
      keep_geometry_allocation = layout_can_reuse_allocation(scene, layout);
    }
  }
  else {
    layout_reuse_rejected_reason_ = "no reallocation requested";
  }

  /* tag the device arrays for reallocation or modification */
  DeviceScene *dscene = &scene->dscene;

  if (device_update_flags & (DEVICE_MESH_DATA_NEEDS_REALLOC | DEVICE_CURVE_DATA_NEEDS_REALLOC |
                             DEVICE_POINT_DATA_NEEDS_REALLOC))
  {
    scene->bvh.reset();

    dscene->bvh_nodes.tag_realloc();
    dscene->bvh_leaf_nodes.tag_realloc();
    dscene->object_node.tag_realloc();
    dscene->prim_type.tag_realloc();
    dscene->prim_visibility.tag_realloc();
    dscene->prim_index.tag_realloc();
    dscene->prim_object.tag_realloc();
    dscene->prim_time.tag_realloc();

    /* Only the geometry arrays can keep their allocation - the arrays above are tied to the BVH,
     * which is still rebuilt from scratch. When the allocation is kept the arrays are marked
     * modified instead, so `device_update_mesh` uploads the rewritten slices. */
    if (device_update_flags & DEVICE_MESH_DATA_NEEDS_REALLOC) {
      if (keep_geometry_allocation) {
        dscene->tri_vindex.tag_modified();
        dscene->tri_shader.tag_modified();
      }
      else {
        dscene->tri_vindex.tag_realloc();
        dscene->tri_shader.tag_realloc();
      }
    }

    /* Curves and point clouds keep the previous behaviour: the reuse predicate only verifies that
     * the triangle arrays are resident at the expected size, so downgrading theirs would be
     * unchecked. */
    if (device_update_flags & DEVICE_CURVE_DATA_NEEDS_REALLOC) {
      dscene->curves.tag_realloc();
      dscene->curve_segments.tag_realloc();
    }

    if (device_update_flags & DEVICE_POINT_DATA_NEEDS_REALLOC) {
      dscene->points_shader.tag_realloc();
    }
  }

  if ((update_flags & VISIBILITY_MODIFIED) != 0) {
    dscene->prim_visibility.tag_modified();
  }

  /* Attribute tables keep their allocation under the same conditions as the geometry arrays, plus
   * two of their own.
   *
   * The three defects that made an earlier attempt render entire scenes black are addressed
   * individually rather than by giving up on the reuse:
   *
   * - the attribute map is left alone. Its early return keys on `need_realloc()`, and the map is
   *   rebuilt whenever the tables are reallocated. Reuse is therefore only allowed when the request
   *   sets cannot have changed - no shader attribute or displacement change, no geometry added or
   *   removed - which is exactly when the map would come out identical. Rebuilding it
   *   unconditionally instead was measured at +460 ms on a 742-mesh scene, so it is not an option.
   * - a table whose size did change is flagged for reallocation inside `device_update_attributes`
   *   by `tag_resized_tables()`, so the full repack still happens for it.
   * - attributes written through a mutable pointer now mark themselves modified, so a table that
   *   keeps its allocation still receives every change.
   *
   * Table layout stability follows from geometry stability plus an unchanged request set: attribute
   * data is sized per vertex, per triangle or per corner, and the order within a table follows the
   * order of geometries and requests. */
  const bool attribute_inputs_stable = (update_flags & (SHADER_ATTRIBUTE_MODIFIED |
                                                        SHADER_DISPLACEMENT_MODIFIED |
                                                        GEOMETRY_ADDED | GEOMETRY_REMOVED |
                                                        MOTION_PASS_NEEDED | UV_PASS_NEEDED)) == 0;
  const bool keep_attribute_allocation = keep_geometry_allocation && attribute_inputs_stable;

  /* Historical note on what this used to say.
   *
   * Three things break if they are, and the first one was observed as a scene going entirely black:
   *
   * - `update_svm_attributes` returns early when the map size is unchanged and the map does not
   *   need reallocating - it checks `need_realloc()`, not `is_modified()`. Marking the map modified
   *   therefore leaves the old map in place while the new `attr_map_offset` values have already been
   *   written, so the kernel reads normals and positions at the wrong offsets.
   * - `attributes_need_realloc[]` inside `device_update_attributes` is derived from the actual
   *   `device_vector::need_realloc()` after `builder.alloc()`, not from these flags. Downgrading to
   *   modified makes it false, so the full repack it is supposed to force does not happen and an
   *   attribute that shifted inside the table but is not itself modified stays at its old place.
   * - a table whose size did change is freed and reallocated by `alloc()` without raising
   *   `need_realloc_`, so the fresh buffer would keep whatever the unmodified attributes did not
   *   write.
   *
   * Table stability also does not follow from geometry stability: attribute sizes depend on the
   * element type, the motion step count and the shader and object request sets, any of which can
   * change while the triangle count does not. Reusing them needs its own snapshot of every entry,
   * which is separate work. */
  if (device_update_flags & ATTR_FLOAT_NEEDS_REALLOC) {
    if (keep_attribute_allocation) {
      dscene->attributes_float.tag_modified();
    }
    else {
      dscene->attributes_map.tag_realloc();
      dscene->attributes_float.tag_realloc();
    }
  }
  else if (device_update_flags & ATTR_FLOAT_MODIFIED) {
    dscene->attributes_float.tag_modified();
  }

  if (device_update_flags & ATTR_FLOAT2_NEEDS_REALLOC) {
    if (keep_attribute_allocation) {
      dscene->attributes_float2.tag_modified();
    }
    else {
      dscene->attributes_map.tag_realloc();
      dscene->attributes_float2.tag_realloc();
    }
  }
  else if (device_update_flags & ATTR_FLOAT2_MODIFIED) {
    dscene->attributes_float2.tag_modified();
  }

  if (device_update_flags & ATTR_FLOAT3_NEEDS_REALLOC) {
    if (keep_attribute_allocation) {
      /* Positions live in their own physical tables, separate from the generic float3 one. */
      dscene->attributes_float3.tag_modified();
      dscene->tri_verts.tag_modified();
      dscene->curve_keys.tag_modified();
      dscene->points.tag_modified();
    }
    else {
      dscene->attributes_map.tag_realloc();
      dscene->attributes_float3.tag_realloc();
      dscene->tri_verts.tag_realloc();
      dscene->curve_keys.tag_realloc();
      dscene->points.tag_realloc();
    }
  }
  else if (device_update_flags & ATTR_FLOAT3_MODIFIED) {
    dscene->attributes_float3.tag_modified();
  }

  if (device_update_flags & ATTR_FLOAT4_NEEDS_REALLOC) {
    if (keep_attribute_allocation) {
      dscene->attributes_float4.tag_modified();
    }
    else {
      dscene->attributes_map.tag_realloc();
      dscene->attributes_float4.tag_realloc();
    }
  }
  else if (device_update_flags & ATTR_FLOAT4_MODIFIED) {
    dscene->attributes_float4.tag_modified();
  }

  if (device_update_flags & ATTR_UCHAR4_NEEDS_REALLOC) {
    if (keep_attribute_allocation) {
      dscene->attributes_uchar4.tag_modified();
    }
    else {
      dscene->attributes_map.tag_realloc();
      dscene->attributes_uchar4.tag_realloc();
    }
  }
  else if (device_update_flags & ATTR_UCHAR4_MODIFIED) {
    dscene->attributes_uchar4.tag_modified();
  }

  if (device_update_flags & ATTR_NORMAL_NEEDS_REALLOC) {
    if (keep_attribute_allocation) {
      dscene->attributes_normal.tag_modified();
    }
    else {
      dscene->attributes_map.tag_realloc();
      dscene->attributes_normal.tag_realloc();
    }
  }
  else if (device_update_flags & ATTR_NORMAL_MODIFIED) {
    dscene->attributes_normal.tag_modified();
  }

  if (device_update_flags & DEVICE_MESH_DATA_MODIFIED) {
    /* if anything else than vertices or shaders are modified, we would need to reallocate, so
     * these are the only arrays that can be updated */
    dscene->tri_shader.tag_modified();
  }

  if (device_update_flags & DEVICE_CURVE_DATA_MODIFIED) {
    dscene->curves.tag_modified();
    dscene->curve_segments.tag_modified();
  }

  if (device_update_flags & DEVICE_POINT_DATA_MODIFIED) {
    dscene->points_shader.tag_modified();
  }

  need_flags_update = false;
}

void GeometryManager::device_update_displacement_images(Device *device,
                                                        Scene *scene,
                                                        Progress &progress)
{
  progress.set_status("Updating Displacement Images");
  ImageManager *image_manager = scene->image_manager.get();
  set<const ImageSingle *> bump_images;
#ifdef WITH_OSL
  bool has_osl_node = false;
#endif
  for (Geometry *geom : scene->geometry) {
    if (geom->is_modified()) {
      /* Geometry-level check for hair shadow transparency.
       * This matches the logic in the `Hair::update_shadow_transparency()`, avoiding access to
       * possible non-loaded images. */
      bool need_shadow_transparency = false;
      if (geom->is_hair()) {
        Hair *hair = static_cast<Hair *>(geom);
        need_shadow_transparency = hair->need_shadow_transparency();
      }

      for (Node *node : geom->get_used_shaders()) {
        Shader *shader = static_cast<Shader *>(node);
        const bool is_true_displacement = (shader->has_displacement &&
                                           shader->get_displacement_method() != DISPLACE_BUMP);
        if (!is_true_displacement && !need_shadow_transparency) {
          continue;
        }
        for (ShaderNode *node : shader->graph->nodes) {
#ifdef WITH_OSL
          if (node->special_type == SHADER_SPECIAL_TYPE_OSL) {
            has_osl_node = true;
          }
#endif
          if (node->special_type != SHADER_SPECIAL_TYPE_IMAGE_SLOT) {
            continue;
          }

          ImageSlotTextureNode *image_node = static_cast<ImageSlotTextureNode *>(node);
          if (!image_node->handle.empty()) {
            image_node->handle.add_to_set(bump_images);
          }
        }
      }
    }
  }

#ifdef WITH_OSL
  /* If any OSL node is used for displacement, it may reference a texture. But it's
   * unknown which ones, so have to load them all. */
  if (has_osl_node) {
    OSLShaderManager::osl_image_handles(device, image_manager, bump_images);
  }
#endif

  image_manager->device_load_images(device, scene, progress, bump_images);
}

void GeometryManager::device_update_volume_images(Device *device, Scene *scene, Progress &progress)
{
  progress.set_status("Updating Volume Images");
  TaskPool pool;
  ImageManager *image_manager = scene->image_manager.get();
  set<const ImageSingle *> volume_images;

  for (Geometry *geom : scene->geometry) {
    if (!geom->is_modified()) {
      continue;
    }

    for (Attribute &attr : geom->attributes.attributes) {
      if (!(attr.element & ATTR_ELEMENT_VOXEL)) {
        continue;
      }

      const ImageHandle &handle = attr.data_voxel();
      if (!handle.empty()) {
        handle.add_to_set(volume_images);
      }
    }
  }

  image_manager->device_load_images(device, scene, progress, volume_images);
}

void GeometryManager::device_update(Device *device,
                                    DeviceScene *dscene,
                                    Scene *scene,
                                    Progress &progress)
{
  if (!need_update()) {
    return;
  }

  LOG_INFO << "Total " << scene->geometry.size() << " meshes.";

  /* Which part of the geometry update the time goes to. This manager measured 16.3 ms of a 26 ms
   * scene update, and another 8.3 ms on the motion-history pass that should not be touching it at
   * all, so it needs breaking down before anything is changed. Printed with its remainder. */
  static const bool report_parts = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  const double parts_started = time_dt();
  double ms_attributes = 0.0, ms_mesh = 0.0, ms_blas = 0.0, ms_tlas = 0.0, ms_displace = 0.0;
  double ms_free = 0.0, ms_offsets = 0.0, ms_tessellate = 0.0, ms_statics = 0.0;
  auto stamp = [](double &slot, const double started) { slot += (time_dt() - started) * 1000.0; };
  struct PartsReport {
    const bool &enabled;
    const double &started;
    const double &attributes, &mesh, &blas, &tlas, &displace;
    const double &free_, &offsets, &tessellate, &statics;
    ~PartsReport()
    {
      if (!enabled) {
        return;
      }
      const double total = (time_dt() - started) * 1000.0;
      fprintf(stderr,
              "GEOM_PARTS total=%.2f attributes=%.2f mesh=%.2f blas=%.2f tlas=%.2f "
              "displace=%.2f free=%.2f offsets=%.2f tess=%.2f statics=%.2f unaccounted=%.2f\n",
              total,
              attributes,
              mesh,
              blas,
              tlas,
              displace,
              free_,
              offsets,
              tessellate,
              statics,
              total - attributes - mesh - blas - tlas - displace - free_ - offsets - tessellate -
                  statics);
      fflush(stderr);
    }
  } parts_report{report_parts,
                 parts_started,
                 ms_attributes,
                 ms_mesh,
                 ms_blas,
                 ms_tlas,
                 ms_displace,
                 ms_free,
                 ms_offsets,
                 ms_tessellate,
                 ms_statics};

  /* The resident layout is only valid if this update runs to completion. There are many early
   * returns below - cancel, device error, displacement failure - and each one leaves the device
   * holding a partially updated scene, so an RAII guard invalidates the snapshot by default and the
   * successful path re-commits it at the end. Hand-written `if (cancel)` checks would have to be
   * repeated at every one of those returns. */
  struct LayoutSnapshotGuard {
    DeviceGeometryLayout &layout;
    bool committed = false;

    ~LayoutSnapshotGuard()
    {
      if (!committed) {
        layout.clear();
      }
    }
  } layout_guard{device_layout_};

  bool true_displacement_used = false;
  bool curve_need_update_shadow_transparency = false;
  size_t num_tessellation = 0;

  {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry({"device_update (normals)", time});
      }
    });

    for (Geometry *geom : scene->geometry) {
      if (geom->is_modified()) {
        if (geom->is_mesh() || geom->is_volume()) {
          Mesh *mesh = static_cast<Mesh *>(geom);

          /* Test if we need tessellation and setup normals if required. */
          if (mesh->need_tesselation()) {
            num_tessellation++;
            /* OPENSUBDIV Catmull-Clark does not make use of input normals and will overwrite them.
             */
#ifdef WITH_OPENSUBDIV
            if (mesh->get_subdivision_type() != Mesh::SUBDIVISION_CATMULL_CLARK)
#endif
            {
              mesh->add_vertex_normals();
            }
          }
          else {
            mesh->add_vertex_normals();
          }

          /* Test if we need displacement. */
          if (mesh->has_true_displacement()) {
            true_displacement_used = true;
          }
        }
      }

      if (progress.get_cancel()) {
        return;
      }
    }

    for (Geometry *geom : scene->geometry) {
      if (geom->is_hair()) {
        Hair *hair = static_cast<Hair *>(geom);
        if (hair->need_shadow_transparency() &&
            (geom->is_modified() || hair->need_update_shadow_transparency()))
        {
          curve_need_update_shadow_transparency = true;
          break;
        }
      }

      if (progress.get_cancel()) {
        return;
      }
    }
  }

  if (progress.get_cancel()) {
    return;
  }

  /* Tessellate meshes that are using subdivision.
   *
   * The timer is scoped to this block only. It used to be declared at function scope, so its
   * destructor fired at the end of the whole function and the `device_update (tangents)` entry
   * silently included the attribute update, the mesh upload and the BVH build - on a 742-mesh scene
   * it reported 621 ms for a block whose actual body costs under a millisecond. */
  {
    const scoped_callback_timer timer([scene, num_tessellation](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry(
            {(num_tessellation) ? "device_update (tessellation and tangents)" :
                                  "device_update (tangents)",
             time});
      }
    });

    Camera *dicing_camera = scene->dicing_camera;
    if (num_tessellation) {
      dicing_camera->set_screen_size(dicing_camera->get_full_width(),
                                     dicing_camera->get_full_height());
      dicing_camera->update(scene);
    }

    size_t i = 0;
    thread_mutex status_mutex;

    parallel_for_each(scene->geometry.begin(), scene->geometry.end(), [&](Geometry *geom) {
    if (progress.get_cancel()) {
      return;
    }

    if (!(geom->is_modified() && geom->is_mesh())) {
      return;
    }

    Mesh *mesh = static_cast<Mesh *>(geom);
    /* Apply generated attribute if needed or remove if not needed */
    mesh->update_generated(scene);

    if (num_tessellation && mesh->need_tesselation()) {
      {
        const thread_scoped_lock status_lock(status_mutex);
        string msg = "Tessellating ";
        if (mesh->name.empty()) {
          msg += string_printf("%u/%u", (uint)(i + 1), (uint)num_tessellation);
        }
        else {
          msg += string_printf(
              "%s %u/%u", mesh->name.c_str(), (uint)(i + 1), (uint)num_tessellation);
        }

        progress.set_status("Updating Mesh", msg);
        i++;
      }

      SubdParams subd_params(mesh);
      subd_params.dicing_rate = mesh->get_subd_dicing_rate();
      subd_params.max_level = mesh->get_subd_max_level();
      if (mesh->get_subd_adaptive_space() == Mesh::SUBDIVISION_ADAPTIVE_SPACE_PIXEL) {
        subd_params.objecttoworld = mesh->get_subd_objecttoworld();
        subd_params.camera = dicing_camera;
      }

      mesh->tessellate(subd_params);
    }

      /* Apply tangents for generated and UVs (if any need them) or remove if not needed */
      mesh->update_tangents(scene, true);
      if (!mesh->has_true_displacement()) {
        mesh->update_tangents(scene, false);
      }
    });
  }

  if (progress.get_cancel()) {
    return;
  }

  /* Update images needed for true displacement. */
  if (true_displacement_used || curve_need_update_shadow_transparency) {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry(
            {"device_update (displacement: load images)", time});
      }
    });
    {
      const double started = time_dt();
      device_update_displacement_images(device, scene, progress);
      stamp(ms_displace, started);
    }
    scene->object_manager->device_update_flags(device, dscene, scene, progress, false);
  }

  /* Device update. */
  {
    const double started = time_dt();
    device_free(device, dscene, false);
    stamp(ms_free, started);
  }

  const BVHLayout bvh_layout = BVHParams::best_bvh_layout(
      scene->params.bvh_layout, device->get_bvh_layout_mask(dscene->data.kernel_features));
  {
    const double started = time_dt();
    geom_calc_offset(scene, bvh_layout);
    stamp(ms_offsets, started);
  }
  if (true_displacement_used || curve_need_update_shadow_transparency) {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry(
            {"device_update (displacement: copy meshes to device)", time});
      }
    });
    {
      const double started = time_dt();
      device_update_mesh(device, dscene, scene, progress);
      stamp(ms_mesh, started);
    }
  }

  if (progress.get_cancel()) {
    return;
  }

  /* Apply transforms, to prepare for static BVH building. */
  if (scene->params.bvh_type == BVH_TYPE_STATIC) {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->object.times.add_entry(
            {"device_update (apply static transforms)", time});
      }
    });

    progress.set_status("Updating Objects", "Applying Static Transformations");
    const double statics_started = time_dt();
    scene->object_manager->apply_static_transforms(dscene, scene, progress);
    stamp(ms_statics, statics_started);
  }

  if (progress.get_cancel()) {
    return;
  }

  /* Attributes must be uploaded to the device before displacement and hair shadow
   * transparency, which run shader evaluation. Otherwise the upload is deferred
   * until after BVH building. */
  const bool need_attributes_before_bvh = true_displacement_used ||
                                          curve_need_update_shadow_transparency;

  if (need_attributes_before_bvh) {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry({"device_update (attributes)", time});
      }
    });
    {
      const double started = time_dt();
      device_update_attributes(device, dscene, scene, progress);
      stamp(ms_attributes, started);
    }
    if (progress.get_cancel()) {
      return;
    }
  }

  /* Update displacement. */
  bool displacement_done = false;
  {
    /* Copy constant data needed by shader evaluation. */
    device->const_copy_to("data", &dscene->data, sizeof(dscene->data));

    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry({"device_update (displacement)", time});
      }
    });

    for (Geometry *geom : scene->geometry) {
      if (geom->is_modified()) {
        if (geom->is_mesh()) {
          Mesh *mesh = static_cast<Mesh *>(geom);
          if (displace(device, scene, mesh, progress)) {
            displacement_done = true;
            need_flags_update = true;
          }
        }
      }

      if (progress.get_cancel()) {
        return;
      }
    }
  }

  if (progress.get_cancel()) {
    return;
  }

  /* Update hair shadow transparency. */
  bool curve_shadow_transparency_done = false;
  {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry(
            {"device_update (curve shadow transparency)", time});
      }
    });

    for (Geometry *geom : scene->geometry) {
      if (geom->is_hair()) {
        Hair *hair = static_cast<Hair *>(geom);
        if (hair->update_shadow_transparency(device, scene, progress)) {
          curve_shadow_transparency_done = true;
        }
      }

      if (progress.get_cancel()) {
        return;
      }
    }
  }

  if (progress.get_cancel()) {
    return;
  }

  /* Displacement and hair shadow transparency modified the geometry. Free device buffers
   * that need to be reallocated, so the BVH and attributes are rebuilt from the updated
   * data below. Freeing the attribute buffers uploaded earlier also keeps them from
   * overlapping with the temporary BVH building buffers, lowering peak memory usage. */
  if (displacement_done || curve_shadow_transparency_done) {
    device_free(device, dscene, false);
  }

  /* Update the BVH even when there is no geometry so the kernel's BVH data is still valid,
   * especially when removing all of the objects during interactive renders.
   * Also update the BVH if the transformations change, we cannot rely on tagging the Geometry
   * as modified in this case, as we may accumulate displacement if the vertices do not also
   * change. */
  bool need_update_scene_bvh = (scene->bvh == nullptr ||
                                (update_flags & (TRANSFORM_MODIFIED | VISIBILITY_MODIFIED)) != 0);
  {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry({"device_update (build object BVHs)", time});
      }
    });

    size_t i = 0;
    size_t num_bvh = 0;
    for (Geometry *geom : scene->geometry) {
      if (geom->is_modified() || geom->need_update_bvh_for_offset) {
        need_update_scene_bvh = true;

        if (geom->need_build_bvh(bvh_layout)) {
          i++;
          num_bvh++;
        }

        /* Note the use of #bvh_task_pool_, see its definition for details. */
        bvh_task_pool_.push([geom, device, dscene, scene, &progress, i, &num_bvh] {
          geom->compute_bvh(device, dscene, &scene->params, &progress, i, num_bvh);
        });
      }
    }

    /* The pool, not the individual builds: they run in parallel, so what the frame waits for is
     * the pool draining, and summing per-build times would report several times the wall clock. */
    const double blas_started = time_dt();
    TaskPool::Summary summary;
    bvh_task_pool_.wait_work(&summary);
    stamp(ms_blas, blas_started);
    LOG_DEBUG << "Objects BVH build pool statistics:\n" << summary.full_report();
  }

  for (Shader *shader : scene->shaders) {
    shader->need_update_uvs = false;
    shader->need_update_attribute = false;
    shader->need_update_displacement = false;
    shader->need_update_shadow_transparency = false;
    shader->shadow_transparency_needs_realloc = false;
  }

  const Scene::MotionType need_motion = scene->need_motion();
  const bool motion_blur = need_motion == Scene::MOTION_BLUR;

  /* Update objects. */
  {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry({"device_update (compute bounds)", time});
      }
    });
    /* Only what actually moved. Bounds are the geometry's bounds through the object transform, so
     * an object whose transform did not change and whose geometry did not change still has the
     * bounds it had. A change of motion blur mode changes how they are computed for every object,
     * so that forces the whole pass. */
    const bool recompute_all = (motion_blur != bounds_used_motion_blur_);
    bounds_used_motion_blur_ = motion_blur;

    for (Object *object : scene->objects) {
      const Geometry *geom = object->get_geometry();
      if (!recompute_all && !object->bounds_need_update && !(geom && geom->is_modified())) {
        continue;
      }
      object->compute_bounds(motion_blur);
      object->bounds_need_update = false;
    }
  }

  if (progress.get_cancel()) {
    return;
  }

  if (need_update_scene_bvh) {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry({"device_update (build scene BVH)", time});
      }
    });
    {
      const double started = time_dt();
      device_update_bvh(device, dscene, scene, progress);
      stamp(ms_tlas, started);
    }
    if (progress.get_cancel()) {
      return;
    }
  }

  /* Always set BVH layout again after displacement where it was set to none,
   * to avoid ray-tracing at that stage. */
  dscene->data.bvh.bvh_layout = BVHParams::best_bvh_layout(
      scene->params.bvh_layout, device->get_bvh_layout_mask(dscene->data.kernel_features));

  /* Upload attributes to the device.
   *
   * This is deferred until after BVH building so the attribute buffers do not overlap with
   * the temporary buffers allocated during BVH building. */
  {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry({"device_update (attributes)", time});
      }
    });
    {
      const double started = time_dt();
      device_update_attributes(device, dscene, scene, progress);
      stamp(ms_attributes, started);
    }
    if (progress.get_cancel()) {
      return;
    }
  }

  {
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->geometry.times.add_entry(
            {"device_update (copy meshes to device)", time});
      }
    });
    {
      const double started = time_dt();
      device_update_mesh(device, dscene, scene, progress);
      stamp(ms_mesh, started);
    }
    if (progress.get_cancel()) {
      return;
    }
  }

  /* The device now holds this layout. Recorded here, at the one point where the update is known to
   * have completed, so the next update can compare against what is actually resident. */
  device_layout_ = compute_layout(scene);
  layout_guard.committed = true;

  /* This upload carried whatever the deferred history advance had left marked, so there is nothing
   * left owing. The advance at the end of this scene update raises it again if positions moved. */
  motion_history_pending = false;

  /* unset flags */

  for (Geometry *geom : scene->geometry) {
    geom->clear_modified();
    geom->attributes.clear_modified();

    if (geom->is_mesh()) {
      Mesh *mesh = static_cast<Mesh *>(geom);
      mesh->subd_attributes.clear_modified();
    }
  }

  update_flags = UPDATE_NONE;

  dscene->bvh_nodes.clear_modified();
  dscene->bvh_leaf_nodes.clear_modified();
  dscene->object_node.clear_modified();
  dscene->prim_type.clear_modified();
  dscene->prim_visibility.clear_modified();
  dscene->prim_index.clear_modified();
  dscene->prim_object.clear_modified();
  dscene->prim_time.clear_modified();
  dscene->tri_shader.clear_modified();
  dscene->tri_vindex.clear_modified();
  dscene->tri_verts.clear_modified();
  dscene->curves.clear_modified();
  dscene->curve_keys.clear_modified();
  dscene->curve_segments.clear_modified();
  dscene->points.clear_modified();
  dscene->points_shader.clear_modified();
  dscene->attributes_map.clear_modified();
  dscene->attributes_float.clear_modified();
  dscene->attributes_float2.clear_modified();
  dscene->attributes_float3.clear_modified();
  dscene->attributes_float4.clear_modified();
  dscene->attributes_uchar4.clear_modified();
  dscene->attributes_normal.clear_modified();
}

DeviceGeometryLayout GeometryManager::compute_layout(Scene *scene)
{
  /* Mirrors the accumulation in `geom_calc_offset`, but writes nothing: the live `prim_offset`
   * fields must stay untouched until the layout has been compared against the resident one. */
  DeviceGeometryLayout layout;
  layout.spans.reserve(scene->geometry.size());

  for (Geometry *geom : scene->geometry) {
    DeviceGeometrySpan span = {geom, 0, 0, 0, 0};

    if (geom->is_mesh() || geom->is_volume()) {
      const Mesh *mesh = static_cast<const Mesh *>(geom);
      span.offset = layout.triangles;
      span.count = mesh->num_triangles();
      layout.triangles += span.count;
    }
    else if (geom->is_hair()) {
      const Hair *hair = static_cast<const Hair *>(geom);
      /* Hair occupies two independent arrays, and their sizes move independently: the number of
       * curves can change without the number of segments changing. Both have to match. */
      span.offset = layout.curves;
      span.count = hair->num_curves();
      span.aux_offset = layout.curve_segments;
      span.aux_count = hair->num_segments();
      layout.curves += span.count;
      layout.curve_segments += span.aux_count;
    }
    else if (geom->is_pointcloud()) {
      const PointCloud *pointcloud = static_cast<const PointCloud *>(geom);
      span.offset = layout.points;
      span.count = pointcloud->num_points();
      layout.points += span.count;
    }

    layout.spans.push_back(span);
  }

  layout.valid = true;
  return layout;
}

bool GeometryManager::layout_can_reuse_allocation(Scene *scene, const DeviceGeometryLayout &layout)
{
  if (!geometry_layout_matches(device_layout_, layout, &layout_reuse_rejected_reason_)) {
    return false;
  }

  /* A matching plan is not enough: the arrays have to be resident at that size. A device reset, a
   * first upload or a migration to host memory leaves them different. */
  DeviceScene *dscene = &scene->dscene;
  if (dscene->tri_vindex.size() != layout.triangles ||
      dscene->tri_shader.size() != layout.triangles)
  {
    layout_reuse_rejected_reason_ = "triangle arrays not resident at this size";
    return false;
  }

  /* An allocation already flagged for reallocation by something else must not be downgraded. The
   * flag is never cleared here, only not added. */
  if (dscene->tri_vindex.need_realloc() || dscene->tri_shader.need_realloc()) {
    layout_reuse_rejected_reason_ = "reallocation already required for another reason";
    return false;
  }

  return true;
}

void GeometryManager::device_free(Device *device, DeviceScene *dscene, bool force_free)
{
  dscene->bvh_nodes.free_if_need_realloc(force_free);
  dscene->bvh_leaf_nodes.free_if_need_realloc(force_free);
  dscene->object_node.free_if_need_realloc(force_free);
  dscene->prim_type.free_if_need_realloc(force_free);
  dscene->prim_visibility.free_if_need_realloc(force_free);
  dscene->prim_index.free_if_need_realloc(force_free);
  dscene->prim_object.free_if_need_realloc(force_free);
  dscene->prim_time.free_if_need_realloc(force_free);
  dscene->tri_shader.free_if_need_realloc(force_free);
  dscene->tri_vindex.free_if_need_realloc(force_free);
  dscene->tri_verts.free_if_need_realloc(force_free);
  dscene->curves.free_if_need_realloc(force_free);
  dscene->curve_keys.free_if_need_realloc(force_free);
  dscene->curve_segments.free_if_need_realloc(force_free);
  dscene->points.free_if_need_realloc(force_free);
  dscene->points_shader.free_if_need_realloc(force_free);
  dscene->attributes_map.free_if_need_realloc(force_free);
  if (dscene->attributes_map.device_pointer == 0) {
    /* The device no longer holds what the shadow claims, so the next update has to upload rather
     * than compare. `free_if_need_realloc` only releases conditionally, hence the check. */
    shadow_attributes_map_.clear();
  }
  dscene->attributes_float.free_if_need_realloc(force_free);
  dscene->attributes_float2.free_if_need_realloc(force_free);
  dscene->attributes_float3.free_if_need_realloc(force_free);
  dscene->attributes_float4.free_if_need_realloc(force_free);
  dscene->attributes_uchar4.free_if_need_realloc(force_free);
  dscene->attributes_normal.free_if_need_realloc(force_free);

  /* Signal for shaders like displacement not to do ray tracing. */
  dscene->data.bvh.bvh_layout = BVH_LAYOUT_NONE;

#ifdef WITH_OSL
  OSLGlobals *og = device->get_cpu_osl_memory();

  if (og) {
    og->object_name_map.clear();
    og->object_names.clear();
  }
#else
  (void)device;
#endif
}

void GeometryManager::tag_update(Scene *scene, const uint32_t flag)
{
  update_flags |= flag;

  /* do not tag the object manager for an update if it is the one who tagged us, and do not tag it
   * for a motion history flush either - that only moves attribute data, and waking the object
   * manager would cascade into the light manager and cost far more than the flush itself */
  if ((flag & (OBJECT_MANAGER | MOTION_HISTORY_MODIFIED)) == 0) {
    scene->object_manager->tag_update(scene, ObjectManager::GEOMETRY_MANAGER);
  }
}

bool GeometryManager::need_update() const
{
  return update_flags != UPDATE_NONE;
}

void GeometryManager::collect_statistics(const Scene *scene, RenderStats *stats)
{
  for (const Geometry *geometry : scene->geometry) {
    stats->mesh.geometry.add_entry(
        NamedSizeEntry(string(geometry->name.c_str()), geometry->get_total_size_in_bytes()));
  }
}

CCL_NAMESPACE_END
