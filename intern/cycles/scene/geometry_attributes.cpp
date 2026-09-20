/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "bvh/bvh.h"

#include "device/device.h"

#include "scene/attribute.h"
#include "scene/camera.h"
#include "scene/geometry.h"
#include "scene/hair.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_nodes.h"

#include "util/progress.h"

CCL_NAMESPACE_BEGIN

bool Geometry::need_attribute(const Scene *scene, AttributeStandard std)
{
  if (std == ATTR_STD_NONE) {
    return false;
  }

  if (scene->need_global_attribute(std)) {
    return true;
  }

  for (Node *node : used_shaders) {
    Shader *shader = static_cast<Shader *>(node);
    if (shader->attributes.find(std)) {
      return true;
    }
  }

  return false;
}

bool Geometry::need_attribute(Scene *scene, ustring name)
{
  if (name.empty()) {
    return false;
  }

  for (const Shader *shader : scene->shaders) {
    if (shader->global_attributes.find(name)) {
      return true;
    }
  }

  for (Node *node : used_shaders) {
    Shader *shader = static_cast<Shader *>(node);
    if (shader->attributes.find(name)) {
      return true;
    }
  }

  return false;
}

AttributeRequestSet Geometry::needed_attributes()
{
  AttributeRequestSet result;

  for (Node *node : used_shaders) {
    Shader *shader = static_cast<Shader *>(node);
    result.add(shader->attributes);
  }

  return result;
}

bool Geometry::has_voxel_attributes() const
{
  for (const Attribute &attr : attributes.attributes) {
    if (attr.element & ATTR_ELEMENT_VOXEL) {
      return true;
    }
  }

  return false;
}

/* Generate a normal attribute map entry from an attribute descriptor. */
static void emit_attribute_map_entry(AttributeMap *attr_map,
                                     const size_t index,
                                     const uint64_t id,
                                     const TypeDesc type,
                                     const AttributeDescriptor &desc)
{
  attr_map[index].id = id;
  attr_map[index].element = desc.element;
  attr_map[index].offset = as_uint(desc.offset);

  if (type == TypeFloat) {
    attr_map[index].type = NODE_ATTR_FLOAT;
  }
  else if (type == TypeMatrix) {
    attr_map[index].type = NODE_ATTR_MATRIX;
  }
  else if (type == TypeFloat2) {
    attr_map[index].type = NODE_ATTR_FLOAT2;
  }
  else if (type == TypeFloat4) {
    attr_map[index].type = NODE_ATTR_FLOAT4;
  }
  else if (type == TypeRGBA) {
    attr_map[index].type = NODE_ATTR_RGBA;
  }
  else {
    attr_map[index].type = NODE_ATTR_FLOAT3;
  }
}

/* Generate an attribute map end marker, optionally including a link to another map.
 * Links are used to connect object attribute maps to mesh attribute maps. */
static void emit_attribute_map_terminator(AttributeMap *attr_map,
                                          const size_t index,
                                          const bool chain,
                                          const uint chain_link)
{
  for (int j = 0; j < ATTR_PRIM_TYPES; j++) {
    attr_map[index + j].id = ATTR_STD_NONE;
    attr_map[index + j].element = chain;                     /* link is valid flag */
    attr_map[index + j].offset = chain ? chain_link + j : 0; /* link to the correct sub-entry */
    attr_map[index + j].type = 0;
  }
}

/* Generate all necessary attribute map entries from the attribute request. */
static void emit_attribute_mapping(AttributeMap *attr_map,
                                   const size_t index,
                                   const uint64_t id,
                                   AttributeRequest &req)
{
  emit_attribute_map_entry(attr_map, index, id, req.type, req.desc);
}

void GeometryManager::update_svm_attributes(Device * /*unused*/,
                                            DeviceScene *dscene,
                                            Scene *scene,
                                            vector<AttributeRequestSet> &geom_attributes,
                                            vector<AttributeRequestSet> &object_attributes)
{
  /* for SVM, the attributes_map table is used to lookup the offset of an
   * attribute, based on a unique shader attribute id. */
  const bool use_osl = scene->shader_manager->use_osl();

  /* compute array stride */
  size_t attr_map_size = 0;

  for (size_t i = 0; i < scene->geometry.size(); i++) {
    Geometry *geom = scene->geometry[i];
    geom->attr_map_offset = attr_map_size;

    size_t attr_count = 0;
    if (use_osl) {
      for (const AttributeRequest &req : geom_attributes[i].requests) {
        if (req.std != ATTR_STD_NONE &&
            scene->shader_manager->get_attribute_id(req.std) != (uint64_t)req.std)
        {
          attr_count += 2;
        }
        else {
          attr_count += 1;
        }
      }
    }
    else {
      attr_count = geom_attributes[i].size();
    }

    attr_map_size += (attr_count + 1) * ATTR_PRIM_TYPES;
  }

  for (size_t i = 0; i < scene->objects.size(); i++) {
    Object *object = scene->objects[i];

    /* only allocate a table for the object if it actually has attributes */
    if (object_attributes[i].size() == 0) {
      object->attr_map_offset = 0;
    }
    else {
      object->attr_map_offset = attr_map_size;
      attr_map_size += (object_attributes[i].size() + 1) * ATTR_PRIM_TYPES;
    }
  }

  if (attr_map_size == 0) {
    return;
  }

  /* The map used to be left alone whenever its total size matched, on the assumption that the same
   * size means the same contents. It does not: the entries carry each attribute's offset inside its
   * table, and those move whenever the packing shifts - one mesh growing while another shrinks
   * keeps every total identical. A stale map then points the kernel at another geometry's data,
   * which shows up as dirty patches and wrong shading that never heal.
   *
   * The map is small next to the tables it describes - a couple of entries per attribute request -
   * so it is rebuilt unconditionally rather than fingerprinted. */

  /* create attribute map */
  AttributeMap *attr_map = dscene->attributes_map.alloc(attr_map_size);
  memset(attr_map, 0, dscene->attributes_map.size() * sizeof(*attr_map));

  for (size_t i = 0; i < scene->geometry.size(); i++) {
    Geometry *geom = scene->geometry[i];
    AttributeRequestSet &attributes = geom_attributes[i];

    /* set geometry attributes */
    size_t index = geom->attr_map_offset;

    for (AttributeRequest &req : attributes.requests) {
      uint64_t id;
      if (req.std == ATTR_STD_NONE) {
        id = scene->shader_manager->get_attribute_id(req.name);
      }
      else {
        id = scene->shader_manager->get_attribute_id(req.std);
      }

      emit_attribute_mapping(attr_map, index, id, req);
      index += ATTR_PRIM_TYPES;

      if (use_osl) {
        /* Some standard attributes are explicitly referenced via their standard ID, so add those
         * again in case they were added under a different attribute ID. */
        if (req.std != ATTR_STD_NONE && id != (uint64_t)req.std) {
          emit_attribute_mapping(attr_map, index, (uint64_t)req.std, req);
          index += ATTR_PRIM_TYPES;
        }
      }
    }

    emit_attribute_map_terminator(attr_map, index, false, 0);
  }

  for (size_t i = 0; i < scene->objects.size(); i++) {
    Object *object = scene->objects[i];
    AttributeRequestSet &attributes = object_attributes[i];

    /* set object attributes */
    if (attributes.size() > 0) {
      size_t index = object->attr_map_offset;

      for (AttributeRequest &req : attributes.requests) {
        uint64_t id;
        if (req.std == ATTR_STD_NONE) {
          id = scene->shader_manager->get_attribute_id(req.name);
        }
        else {
          id = scene->shader_manager->get_attribute_id(req.std);
        }

        emit_attribute_mapping(attr_map, index, id, req);
        index += ATTR_PRIM_TYPES;
      }

      emit_attribute_map_terminator(attr_map, index, true, object->geometry->attr_map_offset);
    }
  }

  /* Copy to device, unless the device already holds exactly this map.
   *
   * `attributes_map` is a MEM_GLOBAL buffer, so the upload is a synchronous copy on the default
   * stream and queues behind whatever tracing is in flight - the same cost that showed up as 4.9 ms
   * of "upload" in the light manager, and for the same reason. The map itself is stable across
   * playback: it says which attribute lives at which offset, and a deforming mesh moves its
   * positions without moving its table entries. It is rebuilt unconditionally just above because
   * the packing can shift underneath it, but rebuilt to the same bytes almost every frame. */
  const size_t map_bytes = dscene->attributes_map.size() * sizeof(AttributeMap);
  if (dscene->attributes_map.device_pointer != 0 && shadow_attributes_map_.size() == map_bytes &&
      std::memcmp(shadow_attributes_map_.data(), attr_map, map_bytes) == 0)
  {
    return;
  }

  dscene->attributes_map.copy_to_device();

  shadow_attributes_map_.resize(map_bytes);
  std::memcpy(shadow_attributes_map_.data(), attr_map, map_bytes);
}

template<typename T> struct AttributeTableEntry {
  device_vector<T> &data;
  size_t offset;
  size_t size;

  /* Which slices this pass actually rewrote, so an unchanged table does not have to be uploaded in
   * full for the sake of a few attributes. Entries are appended in increasing offset order, so
   * neighbours merge as they arrive. */
  struct DirtyRange {
    size_t offset;
    size_t count;
  };
  vector<DirtyRange> dirty;

  /* Where each entry landed on the previous pass, in packing order. An entry that did not change
   * still has to be rewritten when it moved, otherwise its data stays at the old address while the
   * descriptor already points at the new one - and the geometry reads whatever its neighbour left
   * there. Sizes can shift without the table resizing at all: one mesh growing while another
   * shrinks keeps the total identical, which is exactly what Geometry Nodes do frame to frame. */
  vector<size_t> *resident_layout = nullptr;
  vector<size_t> layout;

  void reserve(const size_t attr_size)
  {
    size += attr_size;
  }

  /* Templated on U since we'll want to assign float3 values to a packed_float3 device_vector. */
  template<typename U> size_t add(const U *attr_data, const size_t attr_size, const bool modified)
  {
    assert(data.size() >= offset + attr_size);
    size_t start_offset = offset;

    /* A freshly allocated table holds nothing, so every entry has to be written into it whether or
     * not the attribute itself changed. The caller cannot decide this: it derives "modified" from
     * `attributes_need_realloc[]`, which only covers the six generic tables - positions live in
     * `tri_verts`, `curve_keys` and `points`, and a reallocation of those was invisible to it. */
    const size_t entry_index = layout.size();
    layout.push_back(start_offset);
    const bool moved = resident_layout == nullptr || entry_index >= resident_layout->size() ||
                       (*resident_layout)[entry_index] != start_offset;

    const bool write = modified || data.need_realloc() || moved;

    if (write) {
      for (size_t k = 0; k < attr_size; k++) {
        data[offset + k] = attr_data[k];
      }
      data.tag_modified();

      if (!dirty.empty() && dirty.back().offset + dirty.back().count == start_offset) {
        dirty.back().count += attr_size;
      }
      else {
        dirty.push_back({start_offset, attr_size});
      }
    }
    offset += attr_size;
    return start_offset;
  }

  void alloc()
  {
    data.alloc(size);
  }
};

class AttributeTableBuilder {
 public:
  AttributeTableBuilder(DeviceScene *dscene, vector<size_t> *resident_layouts)
      : attr_float{dscene->attributes_float, 0, 0},
        attr_float2{dscene->attributes_float2, 0, 0},
        attr_float3{dscene->attributes_float3, 0, 0},
        attr_float4{dscene->attributes_float4, 0, 0},
        attr_uchar4{dscene->attributes_uchar4, 0, 0},
        attr_normal{dscene->attributes_normal, 0, 0},
        tri_verts{dscene->tri_verts, 0, 0},
        curve_keys{dscene->curve_keys, 0, 0},
        points{dscene->points, 0, 0}
  {
    /* Same order as `ATTRIBUTE_TABLE_COUNT` in the manager. */
    attr_float.resident_layout = &resident_layouts[0];
    attr_float2.resident_layout = &resident_layouts[1];
    attr_float3.resident_layout = &resident_layouts[2];
    attr_float4.resident_layout = &resident_layouts[3];
    attr_uchar4.resident_layout = &resident_layouts[4];
    attr_normal.resident_layout = &resident_layouts[5];
    tri_verts.resident_layout = &resident_layouts[6];
    curve_keys.resident_layout = &resident_layouts[7];
    points.resident_layout = &resident_layouts[8];
  }

  AttributeTableEntry<float> attr_float;
  AttributeTableEntry<float2> attr_float2;
  AttributeTableEntry<packed_float3> attr_float3;
  AttributeTableEntry<float4> attr_float4;
  AttributeTableEntry<uchar4> attr_uchar4;
  AttributeTableEntry<packed_normal> attr_normal;

  /* Positions in dedicated arrays, gives better BVH2 performance. */
  AttributeTableEntry<packed_float3> tri_verts;
  AttributeTableEntry<float4> curve_keys;
  AttributeTableEntry<float4> points;

  void add(Geometry *geom,
           Attribute *mattr,
           AttributePrimitive prim,
           TypeDesc &type,
           AttributeDescriptor &desc)
  {
    if (mattr == nullptr) {
      /* attribute not found */
      desc.element = ATTR_ELEMENT_NONE;
      desc.offset = 0;
      return;
    }

    /* For hair and pointcloud, pack combined position + radius as float4. */
    if (mattr->std == ATTR_STD_POSITION && (geom->is_hair() || geom->is_pointcloud())) {
      add_position_radius(geom, mattr, type, desc);
      return;
    }
    if (mattr->std == ATTR_STD_RADIUS && (geom->is_hair() || geom->is_pointcloud())) {
      desc.element = ATTR_ELEMENT_NONE;
      desc.offset = 0;
      return;
    }

    /* Store element and type. */
    desc.element = mattr->element;
    type = mattr->type;

    /* Store attribute data in arrays, including possible motion steps. */
    const size_t per_step = Attribute::element_size(geom, mattr->element, prim);
    const int num_motion = mattr->motion.size();

    const AttributeElement &element = desc.element;
    int &offset = desc.offset;

    if (mattr->element & ATTR_ELEMENT_VOXEL) {
      /* store slot in offset value */
      const ImageHandle &handle = mattr->data_voxel();
      offset = handle.kernel_id();
    }
    else if (mattr->element & ATTR_ELEMENT_IS_BYTE) {
      offset = attr_uchar4.add(mattr->data<uchar4>(), per_step, mattr->modified);
      for (int step = 1; step <= num_motion; step++) {
        attr_uchar4.add(mattr->data<uchar4>(step), per_step, mattr->modified);
      }
    }
    else if (mattr->element & ATTR_ELEMENT_IS_NORMAL) {
      offset = attr_normal.add(mattr->data<packed_normal>(), per_step, mattr->modified);
      for (int step = 1; step <= num_motion; step++) {
        attr_normal.add(mattr->data<packed_normal>(step), per_step, mattr->modified);
      }
    }
    else if (mattr->type == TypeFloat) {
      offset = attr_float.add(mattr->data<float>(), per_step, mattr->modified);
      for (int step = 1; step <= num_motion; step++) {
        attr_float.add(mattr->data<float>(step), per_step, mattr->modified);
      }
    }
    else if (mattr->type == TypeFloat2) {
      offset = attr_float2.add(mattr->data<float2>(), per_step, mattr->modified);
      for (int step = 1; step <= num_motion; step++) {
        attr_float2.add(mattr->data<float2>(step), per_step, mattr->modified);
      }
    }
    else if (mattr->type == TypeMatrix) {
      offset = attr_float4.add(
          (const float4 *)mattr->data<Transform>(), per_step * 3, mattr->modified);
      for (int step = 1; step <= num_motion; step++) {
        attr_float4.add(
            (const float4 *)mattr->data<Transform>(step), per_step * 3, mattr->modified);
      }
    }
    else if (mattr->type == TypeFloat4 || mattr->type == TypeRGBA) {
      offset = attr_float4.add(mattr->data<float4>(), per_step, mattr->modified);
      for (int step = 1; step <= num_motion; step++) {
        attr_float4.add(mattr->data<float4>(step), per_step, mattr->modified);
      }
    }
    else {
      AttributeTableEntry<packed_float3> &table = (mattr->std == ATTR_STD_POSITION) ? tri_verts :
                                                                                      attr_float3;
      offset = table.add(mattr->data<packed_float3>(), per_step, mattr->modified);
      for (int step = 1; step <= num_motion; step++) {
        table.add(mattr->data<packed_float3>(step), per_step, mattr->modified);
      }
    }

    /* Primitive index is global, not per object, so we sneak a correction
     * for that in here. Vertex index is per object. */
    if (geom->is_mesh() || geom->is_volume()) {
      Mesh *mesh = static_cast<Mesh *>(geom);
      if (element & ATTR_ELEMENT_FACE) {
        offset -= mesh->prim_offset;
      }
      else if (element & ATTR_ELEMENT_CORNER) {
        offset -= 3 * mesh->prim_offset;
      }
    }
    else if (geom->is_hair()) {
      Hair *hair = static_cast<Hair *>(geom);
      if (element & ATTR_ELEMENT_CURVE) {
        offset -= hair->prim_offset;
      }
    }
    else if (geom->is_pointcloud()) {
      if (element & ATTR_ELEMENT_VERTEX) {
        offset -= geom->prim_offset;
      }
    }
  }

  void reserve(Geometry *geom, Attribute *mattr, AttributePrimitive prim)
  {
    if (mattr == nullptr) {
      return;
    }

    /* Must match the number of steps written by add(), which is derived from
     * the attribute's own stored motion steps rather than geom->get_motion_steps(). */
    const int steps = mattr->num_motion_steps();
    const size_t size = Attribute::element_size(geom, mattr->element, prim) * steps;

    if (mattr->element & ATTR_ELEMENT_VOXEL) {
      /* pass */
    }
    else if (mattr->element & ATTR_ELEMENT_IS_BYTE) {
      attr_uchar4.reserve(size);
    }
    else if (mattr->element & ATTR_ELEMENT_IS_NORMAL) {
      attr_normal.reserve(size);
    }
    else if (mattr->type == TypeFloat) {
      attr_float.reserve(size);
    }
    else if (mattr->type == TypeFloat2) {
      attr_float2.reserve(size);
    }
    else if (mattr->type == TypeMatrix) {
      attr_float4.reserve(size * 3);
    }
    else if (mattr->type == TypeFloat4 || mattr->type == TypeRGBA) {
      attr_float4.reserve(size);
    }
    else {
      AttributeTableEntry<packed_float3> &table = (mattr->std == ATTR_STD_POSITION) ? tri_verts :
                                                                                      attr_float3;
      table.reserve(size);
    }
  }

  /* Pack combined position + radius for hair and point cloud. */
  void add_position_radius(Geometry *geom,
                           Attribute *attr_P,
                           TypeDesc &type,
                           AttributeDescriptor &desc)
  {
    Attribute *attr_R = geom->attributes.find(ATTR_STD_RADIUS);
    const size_t base_size = attr_P->size;
    const int steps = attr_P->has_motion() ? geom->get_motion_steps() : 1;
    const size_t total_size = base_size * steps;

    vector<float4> combined(total_size);
    for (int step = 0; step < steps; step++) {
      const packed_float3 *P = attr_P->data<packed_float3>(step);
      const float *R = attr_R->data<float>(step);
      const size_t dst_offset = size_t(step) * base_size;
      for (size_t i = 0; i < base_size; i++) {
        combined[dst_offset + i] = make_float4(P[i], R[i]);
      }
    }

    desc.element = attr_P->element;
    type = TypeFloat4;

    int &offset = desc.offset;
    const bool modified = attr_P->modified || attr_R->modified;
    AttributeTableEntry<float4> &table = geom->is_hair() ? curve_keys : points;
    offset = table.add(combined.data(), total_size, modified);

    /* Pointcloud uses global primitive index. */
    if (geom->is_pointcloud()) {
      offset -= geom->prim_offset;
    }
  }

  void reserve_position_radius(Geometry *geom, Attribute *attr_P)
  {
    const int steps = attr_P->has_motion() ? geom->get_motion_steps() : 1;
    const size_t total_size = attr_P->size * steps;
    AttributeTableEntry<float4> &table = geom->is_hair() ? curve_keys : points;
    table.reserve(total_size);
  }

  /* Flag every table whose size changed for reallocation.
   *
   * `device_vector::alloc()` frees and reallocates when the size differs, but does not raise
   * `need_realloc_` itself - so without this, the pass below would see a fresh, uninitialised buffer
   * and only rewrite the attributes that happen to be marked modified, leaving the rest as garbage.
   * Tables whose size held still keep their allocation and their contents. */
  void tag_resized_tables()
  {
    const auto tag = [](auto &entry) {
      if (entry.data.size() != entry.size) {
        entry.data.tag_realloc();
      }
    };
    tag(attr_float);
    tag(attr_float2);
    tag(attr_float3);
    tag(attr_float4);
    tag(attr_uchar4);
    tag(attr_normal);
    tag(tri_verts);
    tag(curve_keys);
    tag(points);
  }

  void alloc()
  {
    attr_float.alloc();
    attr_float2.alloc();
    attr_float3.alloc();
    attr_float4.alloc();
    attr_uchar4.alloc();
    attr_normal.alloc();
    tri_verts.alloc();
    curve_keys.alloc();
    points.alloc();
  }

  void copy_to_device_if_modified()
  {
    /* A table that kept its allocation still holds last frame's contents everywhere the pass above
     * did not write, so only the rewritten slices have to travel. A freshly allocated table has no
     * contents to keep and goes up whole. */
    const auto copy = [](auto &entry) {
      if (!entry.data.is_modified()) {
        return;
      }

      if (entry.data.need_realloc()) {
        entry.data.copy_to_device();
        return;
      }

      for (const auto &range : entry.dirty) {
        entry.data.copy_to_device_range(range.offset, range.count);
      }
    };

    /* Recorded only once the table reached the device, so a cancelled update cannot leave a layout
     * claiming that entries are resident when they never travelled. */
    const auto commit = [](auto &entry) {
      if (entry.resident_layout != nullptr) {
        *entry.resident_layout = entry.layout;
      }
    };

    copy(attr_float);
    copy(attr_float2);
    copy(attr_float3);
    copy(attr_float4);
    copy(attr_uchar4);
    copy(attr_normal);
    copy(tri_verts);
    copy(curve_keys);
    copy(points);

    commit(attr_float);
    commit(attr_float2);
    commit(attr_float3);
    commit(attr_float4);
    commit(attr_uchar4);
    commit(attr_normal);
    commit(tri_verts);
    commit(curve_keys);
    commit(points);
  }
};

void GeometryManager::device_update_attributes(Device *device,
                                               DeviceScene *dscene,
                                               Scene *scene,
                                               Progress &progress)
{
  progress.set_status("Updating Mesh", "Computing attributes");

  /* Where the attribute update spends its time. It measures 7.4 ms of the main scene update and
   * 5.9 ms of the motion-history pass, and the shape of the fix depends entirely on which of the
   * passes below that is - the loops over 33858 objects, or the data copies. `with_values` is the
   * number of objects that carry any attribute of their own: if it is near zero, the per-object
   * loops are pure overhead and can go without any caching at all.
   *
   * Reported from a destructor because the function has four early exits. */
  static const bool report_attr = getenv("CYCLES_DEBUG_VIEWPORT_PHASES") != nullptr;
  const double attr_started = time_dt();
  double ms_geom_requests = 0.0, ms_object_requests = 0.0, ms_reserve = 0.0, ms_fill = 0.0;
  double ms_map = 0.0, ms_offsets = 0.0;
  int objects_with_values = 0, objects_with_requests = 0;
  auto stamp = [](double &slot, const double started) { slot += (time_dt() - started) * 1000.0; };
  struct AttrReport {
    const bool &enabled;
    const double &started;
    const double &geom_requests, &object_requests, &reserve, &fill, &map, &offsets;
    const int &with_values, &with_requests;
    ~AttrReport()
    {
      if (!enabled) {
        return;
      }
      const double total = (time_dt() - started) * 1000.0;
      fprintf(stderr,
              "ATTR_PARTS total=%.2f geom_req=%.2f obj_req=%.2f reserve=%.2f fill=%.2f map=%.2f "
              "offsets=%.2f unaccounted=%.2f with_values=%d with_requests=%d\n",
              total,
              geom_requests,
              object_requests,
              reserve,
              fill,
              map,
              offsets,
              total - geom_requests - object_requests - reserve - fill - map - offsets,
              with_values,
              with_requests);
      fflush(stderr);
    }
  } attr_report{report_attr,
                attr_started,
                ms_geom_requests,
                ms_object_requests,
                ms_reserve,
                ms_fill,
                ms_map,
                ms_offsets,
                objects_with_values,
                objects_with_requests};

  const double geom_requests_started = time_dt();

  /* gather per mesh requested attributes. as meshes may have multiple
   * shaders assigned, this merges the requested attributes that have
   * been set per shader by the shader manager */
  vector<AttributeRequestSet> geom_attributes(scene->geometry.size());
  AttributeRequestSet global_attributes;
  scene->need_global_attributes(global_attributes);

  for (size_t i = 0; i < scene->geometry.size(); i++) {
    Geometry *geom = scene->geometry[i];

    geom->index = i;
    geom_attributes[i].add(global_attributes);

    for (Node *node : geom->get_used_shaders()) {
      Shader *shader = static_cast<Shader *>(node);
      geom_attributes[i].add(shader->attributes);
    }

    for (const Attribute &attr : geom->attributes.attributes) {
      switch (attr.std) {
        case ATTR_STD_POSITION:
        case ATTR_STD_VERTEX_NORMAL:
        case ATTR_STD_CORNER_NORMAL:
        case ATTR_STD_SHADOW_TRANSPARENCY:
          geom_attributes[i].add(attr.std);
          break;
        default:
          break;
      }
    }
  }

  stamp(ms_geom_requests, geom_requests_started);
  const double object_requests_started = time_dt();

  /* convert object attributes to use the same data structures as geometry ones */
  vector<AttributeRequestSet> object_attributes(scene->objects.size());

  /* Sparse: only the objects that actually carry an attribute of their own get an entry. This used
   * to hold one `AttributeSet` per object, constructed and destroyed on every call - and an
   * `AttributeSet` owns a `std::list`, which allocates its sentinel node in the default
   * constructor. On this scene not one of the 33858 objects carries an attribute, and building
   * that vector measured 2.2 ms of the 7.4 the attribute update takes. */
  vector<int> object_value_index(scene->objects.size(), -1);
  vector<AttributeSet> object_attribute_values;

  for (size_t i = 0; i < scene->objects.size(); i++) {
    Object *object = scene->objects[i];
    if (object->attributes.size() == 0) {
      continue;
    }
    objects_with_values++;

    Geometry *geom = object->geometry;
    const size_t geom_idx = geom->index;

    assert(geom_idx < scene->geometry.size() && scene->geometry[geom_idx] == geom);

    AttributeRequestSet &geom_requests = geom_attributes[geom_idx];
    AttributeRequestSet &attributes = object_attributes[i];

    for (size_t j = 0; j < object->attributes.size(); j++) {
      const ParamValue &param = object->attributes[j];

      /* add attributes that are requested and not already handled by the mesh */
      if (geom_requests.find(param.name()) && !geom->attributes.find(param.name())) {
        attributes.add(param.name());

        if (object_value_index[i] < 0) {
          object_value_index[i] = int(object_attribute_values.size());
          object_attribute_values.push_back(AttributeSet(geom, ATTR_PRIM_GEOMETRY));
        }
        AttributeSet &values = object_attribute_values[object_value_index[i]];

        Attribute *attr = values.add(param.name(), param.type(), ATTR_ELEMENT_OBJECT);
        assert(param.nvalues() == attr->size);
        memcpy(attr->data_for_write(), param.data(), param.datasize());
      }
    }
  }

  stamp(ms_object_requests, object_requests_started);
  const double reserve_started = time_dt();

  /* mesh attribute are stored in a single array per data type. here we fill
   * those arrays, and set the offset and element type to create attribute
   * maps next */

  /* Pre-allocate attributes to avoid arrays re-allocation which would
   * take 2x of overall attribute memory usage.
   */
  AttributeTableBuilder builder(dscene, attribute_layouts_);

  for (size_t i = 0; i < scene->geometry.size(); i++) {
    Geometry *geom = scene->geometry[i];
    AttributeRequestSet &attributes = geom_attributes[i];
    for (AttributeRequest &req : attributes.requests) {
      Attribute *attr = geom->attributes.find(req);
      if (attr && (geom->is_hair() || geom->is_pointcloud())) {
        /* Special cases for packed position + radius. */
        if (attr->std == ATTR_STD_POSITION) {
          builder.reserve_position_radius(geom, attr);
          continue;
        }
        if (attr->std == ATTR_STD_RADIUS) {
          continue;
        }
      }
      builder.reserve(geom, attr, ATTR_PRIM_GEOMETRY);
    }
  }

  for (size_t i = 0; i < scene->objects.size(); i++) {
    if (object_value_index[i] < 0) {
      continue;
    }
    Object *object = scene->objects[i];

    for (Attribute &attr : object_attribute_values[object_value_index[i]].attributes) {
      builder.reserve(object->geometry, &attr, ATTR_PRIM_GEOMETRY);
    }
  }

  builder.tag_resized_tables();
  builder.alloc();

  stamp(ms_reserve, reserve_started);
  const double fill_started = time_dt();

  /* The order of those flags needs to match that of AttrKernelDataType. */
  const bool attributes_need_realloc[AttrKernelDataType::NUM] = {
      dscene->attributes_float.need_realloc(),
      dscene->attributes_float2.need_realloc(),
      dscene->attributes_float3.need_realloc(),
      dscene->attributes_float4.need_realloc(),
      dscene->attributes_uchar4.need_realloc(),
      dscene->attributes_normal.need_realloc(),
  };

  /* Fill in attributes. */
  for (size_t i = 0; i < scene->geometry.size(); i++) {
    Geometry *geom = scene->geometry[i];
    AttributeRequestSet &attributes = geom_attributes[i];

    /* todo: we now store std and name attributes from requests even if
     * they actually refer to the same mesh attributes, optimize */
    for (AttributeRequest &req : attributes.requests) {
      Attribute *attr = geom->attributes.find(req);

      if (attr) {
        /* force a copy if we need to reallocate all the data */
        attr->modified |= attributes_need_realloc[Attribute::kernel_type(*attr)];
      }

      builder.add(geom, attr, ATTR_PRIM_GEOMETRY, req.type, req.desc);

      if (progress.get_cancel()) {
        return;
      }
    }
  }

  for (size_t i = 0; i < scene->objects.size(); i++) {
    AttributeRequestSet &attributes = object_attributes[i];
    if (attributes.requests.empty()) {
      continue;
    }
    objects_with_requests++;

    Object *object = scene->objects[i];
    /* A request without a value cannot happen: the request is only added alongside its value
     * above. The lookup below still tolerates a null attribute, as it did before. */
    AttributeSet &values = object_attribute_values[object_value_index[i]];

    for (AttributeRequest &req : attributes.requests) {
      Attribute *attr = values.find(req);

      if (attr) {
        attr->modified |= attributes_need_realloc[Attribute::kernel_type(*attr)];
      }

      builder.add(object->geometry, attr, ATTR_PRIM_GEOMETRY, req.type, req.desc);

      if (progress.get_cancel()) {
        return;
      }
    }
  }

  stamp(ms_fill, fill_started);
  const double map_started = time_dt();

  /* create attribute lookup maps */
  if (scene->shader_manager->use_osl()) {
    update_osl_globals(device, scene);
  }

  update_svm_attributes(device, dscene, scene, geom_attributes, object_attributes);

  stamp(ms_map, map_started);

  if (progress.get_cancel()) {
    return;
  }

  /* copy to device */
  progress.set_status("Updating Mesh", "Copying Attributes to device");

  builder.copy_to_device_if_modified();

  if (progress.get_cancel()) {
    return;
  }

  /* After mesh attributes and patch tables have been copied to device memory,
   * we need to update offsets in the objects. */
  const double offsets_started = time_dt();
  scene->object_manager->device_update_geom_offsets(device, dscene, scene);
  stamp(ms_offsets, offsets_started);
}

CCL_NAMESPACE_END
