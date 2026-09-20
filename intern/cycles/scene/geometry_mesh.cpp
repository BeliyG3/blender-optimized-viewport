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
#include "scene/osl.h"
#include "scene/pointcloud.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_nodes.h"

#include "util/progress.h"

CCL_NAMESPACE_BEGIN

void GeometryManager::device_update_mesh(Device * /*unused*/,
                                         DeviceScene *dscene,
                                         Scene *scene,
                                         Progress &progress)
{
  /* Count. */
  size_t tri_size = 0;

  size_t curve_size = 0;
  size_t curve_segment_size = 0;

  size_t point_size = 0;

  for (Geometry *geom : scene->geometry) {
    if (geom->is_mesh() || geom->is_volume()) {
      Mesh *mesh = static_cast<Mesh *>(geom);

      tri_size += mesh->num_triangles();
    }
    else if (geom->is_hair()) {
      Hair *hair = static_cast<Hair *>(geom);

      curve_size += hair->num_curves();
      curve_segment_size += hair->num_segments();
    }
    else if (geom->is_pointcloud()) {
      PointCloud *pointcloud = static_cast<PointCloud *>(geom);
      point_size += pointcloud->num_points();
    }
  }

  /* Fill in all the arrays. */
  if (tri_size != 0) {
    progress.set_status("Updating Mesh", "Computing normals");

    uint *tri_shader = dscene->tri_shader.alloc(tri_size);
    packed_uint3 *tri_vindex = dscene->tri_vindex.alloc(tri_size);

    const bool copy_all_data = dscene->tri_shader.need_realloc() ||
                               dscene->tri_vindex.need_realloc();

    /* Which slices of the scene-wide arrays were actually rewritten. Without this the arrays are
     * uploaded whole whenever any mesh changed - on a 742-mesh scene with four changed meshes that
     * measured 180 ms per frame for data that was already on the device. */
    struct DirtyRange {
      size_t offset;
      size_t count;
    };
    vector<DirtyRange> dirty_shader;
    vector<DirtyRange> dirty_vindex;

    const auto push_range = [](vector<DirtyRange> &ranges, const size_t offset, const size_t count) {
      if (count == 0) {
        return;
      }
      /* Meshes are visited in offset order, so a new range either extends the last one or starts a
       * new one. Merging keeps the number of uploads down when neighbouring meshes both changed. */
      if (!ranges.empty() && ranges.back().offset + ranges.back().count == offset) {
        ranges.back().count += count;
        return;
      }
      ranges.push_back({offset, count});
    };

    for (Geometry *geom : scene->geometry) {
      if (geom->is_mesh() || geom->is_volume()) {
        Mesh *mesh = static_cast<Mesh *>(geom);
        const size_t num_triangles = mesh->num_triangles();

        /* `mesh->is_modified()` rather than the individual socket flags: tessellation writes the
         * triangles, shaders and smooth flags through direct pointers and is not obliged to leave
         * `triangles_is_modified()` set. That used to be harmless because the whole array was
         * re-uploaded anyway; with a range upload it would silently leave stale triangles. */
        const bool mesh_data_dirty = mesh->is_modified() || copy_all_data;

        if (mesh_data_dirty) {
          mesh->pack_shaders(scene, &tri_shader[mesh->prim_offset]);
          push_range(dirty_shader, mesh->prim_offset, num_triangles);

          mesh->pack_triangles(&tri_vindex[mesh->prim_offset]);
          push_range(dirty_vindex, mesh->prim_offset, num_triangles);
        }

        if (progress.get_cancel()) {
          return;
        }
      }
    }

    /* vertex coordinates */
    progress.set_status("Updating Mesh", "Copying Mesh to device");

    /* A resized allocation has no valid device copy to patch, so it goes up whole. */
    if (copy_all_data) {
      dscene->tri_shader.copy_to_device_if_modified();
      dscene->tri_vindex.copy_to_device_if_modified();
    }
    else {
      for (const DirtyRange &range : dirty_shader) {
        dscene->tri_shader.copy_to_device_range(range.offset, range.count);
      }
      for (const DirtyRange &range : dirty_vindex) {
        dscene->tri_vindex.copy_to_device_range(range.offset, range.count);
      }
    }
  }

  if (curve_segment_size != 0) {
    progress.set_status("Updating Mesh", "Copying Curves to device");

    KernelCurve *curves = dscene->curves.alloc(curve_size);
    KernelCurveSegment *curve_segments = dscene->curve_segments.alloc(curve_segment_size);

    const bool copy_all_data = dscene->curves.need_realloc() ||
                               dscene->curve_segments.need_realloc();

    for (Geometry *geom : scene->geometry) {
      if (geom->is_hair()) {
        Hair *hair = static_cast<Hair *>(geom);

        const bool curve_data_modified = hair->curve_shader_is_modified() ||
                                         hair->curve_first_key_is_modified();

        if (!curve_data_modified && !copy_all_data) {
          continue;
        }

        hair->pack_curves(
            scene, &curves[hair->prim_offset], &curve_segments[hair->curve_segment_offset]);
        if (progress.get_cancel()) {
          return;
        }
      }
    }

    dscene->curves.copy_to_device_if_modified();
    dscene->curve_segments.copy_to_device_if_modified();
  }

  if (point_size != 0) {
    progress.set_status("Updating Mesh", "Copying Point clouds to device");

    uint *points_shader = dscene->points_shader.alloc(point_size);

    for (Geometry *geom : scene->geometry) {
      if (geom->is_pointcloud()) {
        PointCloud *pointcloud = static_cast<PointCloud *>(geom);
        pointcloud->pack(scene, &points_shader[pointcloud->prim_offset]);
        if (progress.get_cancel()) {
          return;
        }
      }
    }

    dscene->points_shader.copy_to_device();
  }

}

CCL_NAMESPACE_END
