/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Walking the baked density octree of a volume.
 *
 * Split out of `shade_volume.h` so that the froxel grid can walk the same tree. What lives here is
 * the part that needs nothing but the kernel globals, a ray and a volume stack entry: the octree
 * itself, and finding the leaf a point falls in. Everything that needs the state of a path - the
 * extrema of a shader with a light path node in it, the walk across overlapping volumes - stays
 * where it was, because the grid has no path to give it.
 */

#pragma once

#include "kernel/globals.h"

#include "kernel/geom/object.h"

#include "kernel/types.h"

CCL_NAMESPACE_BEGIN

#ifdef __VOLUME__

/* We use both volume octree and volume stack, sometimes they disagree on whether a point is inside
 * a volume or not. We accept small numerical precision issues, above this threshold the volume
 * stack shall prevail. */
/* TODO(weizhen): tweak this value. */
#  define OVERLAP_EXP 5e-4f
/* Restrict the number of steps in case of numerical problems */
#  define VOLUME_MAX_STEPS 1024
/* Number of mantissa bits of floating-point numbers. */
#  define MANTISSA_BITS 23

/* -------------------------------------------------------------------- */
/** \name Hierarchical DDA for ray tracing the volume octree
 *
 * Following "Efficient Sparse Voxel Octrees" by Samuli Laine and Tero Karras,
 * and the implementation in https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/
 *
 * The ray segment is transformed into octree space [1, 2), with `ray->D` pointing all negative
 * directions. At each ray tracing step, we intersect the backface of the current active leaf node
 * to find `t.max`, then store a point `current_P` which lies in the adjacent leaf node. The next
 * leaf node is found by checking the higher bits of `current_P`.
 *
 * The paper suggests to keep a stack of parent nodes, in practice such a stack (even when the size
 * is just 8) slows down performance on GPU. Instead we store the parent index in the leaf node
 * directly, since there is sufficient space due to alignment.
 *
 * \{ */

struct OctreeTracing {
  /* Current active leaf node. */
  ccl_global const KernelOctreeNode *node = nullptr;

  /* Current active ray segment, typically spans from the front face to the back face of the
   * current leaf node. */
  Interval<float> t;

  /* Ray origin in octree coordinate space. */
  packed_float3 ray_P;

  /* Ray direction in octree coordinate space. */
  packed_float3 ray_D;

  /* Current active position in octree coordinate space. */
  uint3 current_P;

  /* Object and shader which the octree represents. */
  VolumeStack entry = {OBJECT_NONE, SHADER_NONE};

  /* Scale of the current active leaf node, relative to the smallest possible size representable by
   * float. Initialize to the number of float mantissa bits. */
  uint8_t scale = MANTISSA_BITS;
  uint8_t next_scale;
  /* Mark the dimension (x,y,z) to negate the ray so that we find the correct octant. */
  uint8_t octant_mask;

  /* Whether multiple volumes overlap in the ray segment. */
  bool no_overlap = false;

  /* Maximum and minimum of the densities in the current segment. */
  Extrema<float> sigma = 0.0f;

  ccl_device_inline_method OctreeTracing(const float tmin)
  {
    /* Initialize t.max to FLT_MAX so that any intersection with the node face is smaller. */
    t = {tmin, FLT_MAX};
  }

  enum Dimension { DIM_X = 1U << 0U, DIM_Y = 1U << 1U, DIM_Z = 1U << 2U };

  /* Given ray origin `P` and direction `D` in object space, convert them into octree space
   * [1.0, 2.0).
   * Returns false if ray is leaving the octree or octree has degenerate shape. */
  ccl_device_inline_method bool to_octree_space(ccl_private const float3 &P,
                                                ccl_private const float3 &D,
                                                const float3 scale,
                                                const float3 translation)
  {
    if (!isfinite_safe(scale)) {
      /* Octree with a degenerate shape. */
      return false;
    }

    /* Starting point of octree tracing. */
    float3 local_P = (P + D * t.min) * scale + translation;
    ray_D = D * scale;

    /* Select octant mask to mirror the coordinate system so that ray direction is negative along
     * each axis, and adjust `local_P` accordingly. */
    const auto positive = ray_D > 0.0f;
    octant_mask = (!!positive.x * DIM_X) | (!!positive.y * DIM_Y) | (!!positive.z * DIM_Z);
    local_P = select(positive, 3.0f - local_P, local_P);

    /* Clamp to the largest floating-point number smaller than 2.0f, for numerical stability. */
    local_P = min(local_P, make_float3(1.9999999f));
    current_P = float3_as_uint3(local_P);

    ray_D = -fabs(ray_D);

    /* Ray origin. */
    ray_P = local_P - ray_D * t.min;

    /* Returns false if point lies outside of the octree and the ray is leaving the octree. */
    return all(local_P > 1.0f);
  }

  /* Find the bounding box min of the node that `current_P` lies in within the current scale. */
  ccl_device_inline_method float3 floor_pos() const
  {
    /* Erase bits lower than scale. */
    const uint mask = ~0u << scale;
    return make_float3(__uint_as_float(current_P.x & mask),
                       __uint_as_float(current_P.y & mask),
                       __uint_as_float(current_P.z & mask));
  }

  /* Find arbitrary position inside the next node.
   * We use the end of the current segment offsetted by half of the minimal node size in the normal
   * direction of the last face intersection. */
  ccl_device_inline_method void find_next_pos(const float3 bbox_min,
                                              const float3 t,
                                              const float tmax)
  {
    constexpr float half_size = 1.0f / (2 << VOLUME_OCTREE_MAX_DEPTH);
    const uint3 next_P = float3_as_uint3(
        select(t == tmax, bbox_min - half_size, ray_D * tmax + ray_P));

    /* Find the nearest common ancestor of two positions by checking the shared higher bits. */
    const uint diff = (current_P.x ^ next_P.x) | (current_P.y ^ next_P.y) |
                      (current_P.z ^ next_P.z);

    current_P = next_P;
    next_scale = 32u - count_leading_zeros(diff);
  }

  /* See `ray_aabb_intersect()`. We only need to intersect the 3 back sides because the ray
   * direction is all negative. */
  ccl_device_inline_method float ray_voxel_intersect(const float ray_tmax)
  {
    const float3 bbox_min = floor_pos();

    /* Distances to the three surfaces. */
    float3 intersect_t = (bbox_min - ray_P) / ray_D;

    /* Select the smallest element that is larger than `t.min`, to avoid self intersection. */
    intersect_t = select(intersect_t > t.min, intersect_t, make_float3(FLT_MAX));

    /* The first intersection is given by the smallest t. */
    const float tmax = reduce_min(intersect_t);

    find_next_pos(bbox_min, intersect_t, tmax);

    return fminf(tmax, ray_tmax);
  }

  /* Returns the octant of `current_P` in the node at given scale. */
  ccl_device_inline_method int get_octant() const
  {
    const uint8_t x = (current_P.x >> scale) & 1u;
    const uint8_t y = ((current_P.y >> scale) & 1u) << 1u;
    const uint8_t z = ((current_P.z >> scale) & 1u) << 2u;
    return (x | y | z) ^ octant_mask;
  }
};

/* Check if an octree node is leaf node. */
ccl_device_inline bool volume_node_is_leaf(const ccl_global KernelOctreeNode *knode)
{
  return knode->first_child == -1;
}

/* Find the leaf node of the current position, and replace `octree.node` with that node. */
ccl_device void volume_voxel_get(KernelGlobals kg, ccl_private OctreeTracing &octree)
{
  while (!volume_node_is_leaf(octree.node)) {
    octree.scale -= 1;
    const int child_index = octree.node->first_child + octree.get_octant();
    octree.node = &kernel_data_fetch(volume_tree_nodes, child_index);
  }
}

/* Find the octree root node in the kernel array that corresponds to the volume stack entry. */
ccl_device_inline const ccl_global KernelOctreeRoot *volume_find_octree_root(
    KernelGlobals kg, const VolumeStack entry)
{
  int root = kernel_data_fetch(volume_tree_root_ids, entry.object);
  const ccl_global KernelOctreeRoot *kroot = &kernel_data_fetch(volume_tree_roots, root);
  while ((entry.shader & SHADER_MASK) != kroot->shader) {
    /* If one object has multiple shaders, we store the index of the last shader, and search
     * backwards for the octree with the corresponding shader. */
    kroot = &kernel_data_fetch(volume_tree_roots, --root);
  }
  return kroot;
}

#endif /* __VOLUME__ */

CCL_NAMESPACE_END
