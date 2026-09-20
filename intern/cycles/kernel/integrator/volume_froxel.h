/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Camera-aligned volume grid.
 *
 * One sample per pixel through a dense medium is noise, and no denoiser guide fixes that - the
 * signal simply is not there. While the viewport is being interacted with, the medium is evaluated
 * on a coarse grid instead: every cell holds what the fog emits and how much it blocks, the grid is
 * integrated front to back, and a ray segment then costs two lookups and a division. The answer is
 * approximate but has no variance at all, which is what a moving image needs.
 *
 * The grid is rebuilt from scratch every frame. That is deliberate: reprojecting it between frames
 * is what produces the light trails Unreal and Unity document for their volumetric fog, and trails
 * are exactly what must not appear here.
 *
 * Layout. Columns follow the render resolution - one column per `tile` pixels - so a cell is found
 * from the pixel index alone, with no projection back from world space. That keeps panoramic
 * cameras, depth of field and the reconstruction's sub-pixel jitter correct for free. Slices are
 * distributed exponentially in depth, so the near ones are thin where the eye can tell.
 */

#pragma once

#include "kernel/globals.h"

#include "kernel/camera/camera.h"

#include "kernel/bvh/bvh.h"
#include "kernel/geom/object.h"
#include "kernel/geom/shader_data.h"

#include "kernel/integrator/volume_octree.h"
#include "kernel/integrator/state.h"
#include "kernel/integrator/volume_shader.h"
#include "kernel/light/light.h"
#include "kernel/light/sample.h"
#include "kernel/util/colorspace.h"

#include "util/hash.h"

CCL_NAMESPACE_BEGIN

/* How many octree nodes a shadow walk may cross. Deep enough for the tree the path tracer bakes -
 * seven levels - and a bound rather than a budget: the walk ends at the boundary long before this
 * in anything but a pathological medium. */
#define VOLUME_FROXEL_SHADOW_STEPS 64


/* Where one medium covers the column, found once and reused by every slice.
 *
 * The alternative is asking "which volumes contain this point" per cell, and there is no such query
 * outside a traced path - the stack is something the integrator carries along a ray, not something
 * the scene can be asked about. Walking the hulls once and keeping the spans is the same answer for
 * a fraction of the work, and it handles overlapping volumes because spans simply add up. */
typedef struct VolumeFroxelSpan {
  float t_enter;
  float t_exit;
  int object;
  int shader;
} VolumeFroxelSpan;

/* Distance to the front face of slice `k`, exponential so near slices are thin.
 *
 * `t_near` cannot be zero: the ratio below is what spaces the slices, and a zero start would put
 * every one of them in the same place. */
ccl_device_inline float volume_froxel_slice_distance(KernelGlobals kg, const int k)
{
  const float t_near = kernel_data.froxel.t_near;
  const float t_far = kernel_data.froxel.t_far;
  const int res_z = kernel_data.froxel.res_z;

  if (k <= 0) {
    return t_near;
  }
  if (k >= res_z) {
    return t_far;
  }

  const float fraction = float(k) / float(res_z);
  return t_near * powf(t_far / t_near, fraction);
}

/* Which slice a distance falls in, the inverse of the above. Returns a fractional index so a
 * segment that ends inside a slice can take the rest of it analytically instead of snapping to the
 * boundary - snapping is what puts a halo around every silhouette. */
ccl_device_inline float volume_froxel_slice_coordinate(KernelGlobals kg, const float t)
{
  const float t_near = kernel_data.froxel.t_near;
  const float t_far = kernel_data.froxel.t_far;
  const int res_z = kernel_data.froxel.res_z;

  if (!(t > t_near)) {
    return 0.0f;
  }
  if (t >= t_far) {
    return float(res_z);
  }

  return float(res_z) * logf(t / t_near) / logf(t_far / t_near);
}

ccl_device_inline int volume_froxel_index(KernelGlobals kg, const int x, const int y, const int z)
{
  return (z * kernel_data.froxel.res_y + y) * kernel_data.froxel.res_x + x;
}

/* Emission and extinction of one cell.
 *
 * Filled by the injection kernel; `w` carries the extinction, `xyz` what the medium adds on its
 * own. Both are per unit length, and the segment lookup integrates them across the slices it
 * touches. */
ccl_device_inline float4 volume_froxel_read_scatter(KernelGlobals kg, const int index)
{
  return kernel_data_array(volume_froxel_scatter)[index];
}

/* The same slice read at a fractional column, blended between the four around it.
 *
 * Reading the nearest column instead makes every tile boundary visible as a step, and at one
 * column per eight pixels the picture comes out looking like a chequerboard - which is what this
 * grid is meant to replace, not to introduce. Extinction and emission are both linear in space, so
 * blending them is the same as blending what they describe. */
ccl_device_inline float4 volume_froxel_sample_scatter(KernelGlobals kg,
                                                      const float column_x,
                                                      const float column_y,
                                                      const int z)
{
  const int res_x = kernel_data.froxel.res_x;
  const int res_y = kernel_data.froxel.res_y;

  const float cx = clamp(column_x, 0.0f, float(res_x) - 1.0f);
  const float cy = clamp(column_y, 0.0f, float(res_y) - 1.0f);

  const int x0 = int(cx);
  const int y0 = int(cy);
  const int x1 = min(x0 + 1, res_x - 1);
  const int y1 = min(y0 + 1, res_y - 1);

  const float tx = cx - float(x0);
  const float ty = cy - float(y0);

  const float4 c00 = volume_froxel_read_scatter(kg, volume_froxel_index(kg, x0, y0, z));
  const float4 c10 = volume_froxel_read_scatter(kg, volume_froxel_index(kg, x1, y0, z));
  const float4 c01 = volume_froxel_read_scatter(kg, volume_froxel_index(kg, x0, y1, z));
  const float4 c11 = volume_froxel_read_scatter(kg, volume_froxel_index(kg, x1, y1, z));

  return (c00 * (1.0f - tx) + c10 * tx) * (1.0f - ty) + (c01 * (1.0f - tx) + c11 * tx) * ty;
}

ccl_device_inline void volume_froxel_write_scatter(KernelGlobals kg,
                                                   const int index,
                                                   const float4 value)
{
  kernel_data_array(volume_froxel_scatter)[index] = value;
}

/* Which media cover this column, and between which distances.
 *
 * A hull crossed from the inside that was never entered means the camera started inside that
 * medium, so the span opens at the camera rather than being discarded. */
ccl_device_inline int volume_froxel_collect_spans(KernelGlobals kg,
                                                  ccl_private Ray *ccl_restrict ray,
                                                  ccl_private VolumeFroxelSpan *spans,
                                                  const int max_spans)
{
  int num_spans = 0;

  /* A world volume has no hull to be found by, so nothing below would ever see it: the column
   * would come back empty and the fog would simply not be there. The path tracer puts it at the
   * bottom of its volume stack for the same reason, and it reaches as far as the grid does. */
  if ((kernel_data.froxel.fixes & VOLUME_FROXEL_FIX_WORLD_VOLUME) &&
      kernel_data.background.volume_shader != SHADER_NONE && max_spans > 0)
  {
    spans[num_spans].t_enter = ray->tmin;
    spans[num_spans].t_exit = ray->tmax;
    spans[num_spans].object = kernel_data.background.object_index;
    spans[num_spans].shader = kernel_data.background.volume_shader & SHADER_MASK;
    num_spans++;
  }

  Ray hull_ray = *ray;
  hull_ray.self.object = OBJECT_NONE;
  hull_ray.self.prim = PRIM_NONE;
  hull_ray.self.light_object = OBJECT_NONE;
  hull_ray.self.light_prim = PRIM_NONE;

  ShaderData sd;

#ifdef __VOLUME_RECORD_ALL__
  /* Where the single-hit form is unavailable, all crossings come back at once and have to be put
   * in order by hand - a span only makes sense if entry precedes exit. Insertion sort: this is at
   * most a few dozen crossings, and the array is nearly sorted already. */
  Intersection hits[2 * MAX_VOLUME_STACK_SIZE];
  const uint max_hits = uint(2 * max_spans);
  const uint num_hits = scene_intersect_volume(
      kg, &hull_ray, hits, max_hits, PATH_RAY_VISIBILITY_ALL);

  for (uint i = 1; i < num_hits; i++) {
    const Intersection key = hits[i];
    int j = int(i) - 1;
    while (j >= 0 && hits[j].t > key.t) {
      hits[j + 1] = hits[j];
      j--;
    }
    hits[j + 1] = key;
  }

  for (uint hit = 0; hit < num_hits; hit++) {
    const Intersection isect = hits[hit];
    shader_setup_from_ray(kg, &sd, &hull_ray, &isect);

    if (sd.flag & SD_HAS_VOLUME) {
#else
  Intersection isect;
  int step = 0;

  while (step < 2 * max_spans &&
         scene_intersect_volume(kg, &hull_ray, &isect, PATH_RAY_VISIBILITY_ALL))
  {
    shader_setup_from_ray(kg, &sd, &hull_ray, &isect);

    if (sd.flag & SD_HAS_VOLUME) {
#endif
      if (sd.flag & SD_BACKFACING) {
        /* Leaving a medium: close its span, or open one at the camera if we were already inside
         * when the column started. */
        int found = -1;
        for (int i = 0; i < num_spans; i++) {
          if (spans[i].object == sd.object && spans[i].shader == (sd.shader & SHADER_MASK) &&
              spans[i].t_exit <= spans[i].t_enter)
          {
            found = i;
          }
        }
        if (found >= 0) {
          spans[found].t_exit = isect.t;
        }
        else if (num_spans < max_spans) {
          spans[num_spans].t_enter = ray->tmin;
          spans[num_spans].t_exit = isect.t;
          spans[num_spans].object = sd.object;
          spans[num_spans].shader = sd.shader & SHADER_MASK;
          num_spans++;
        }
      }
      else if (num_spans < max_spans) {
        /* Entering: the span stays open until the matching exit, and `t_exit <= t_enter` is what
         * marks it as open. */
        spans[num_spans].t_enter = isect.t;
        spans[num_spans].t_exit = isect.t;
        spans[num_spans].object = sd.object;
        spans[num_spans].shader = sd.shader & SHADER_MASK;
        num_spans++;
      }
    }

#ifdef __VOLUME_RECORD_ALL__
  }
#else
    hull_ray.tmin = intersection_t_offset(isect.t);
    hull_ray.self.object = isect.object;
    hull_ray.self.prim = isect.prim;
    step++;
  }
#endif

  /* A medium the ray entered but never left reaches as far as the grid does. */
  for (int i = 0; i < num_spans; i++) {
    if (spans[i].t_exit <= spans[i].t_enter) {
      spans[i].t_exit = kernel_data.froxel.t_far;
    }
  }

  return num_spans;
}

/* What one cell receives from the lights, already folded onto the direction of the camera.
 *
 * The seed is a function of the cell and of nothing else. The grid is rebuilt from scratch every
 * frame, so a seed that moved with the frame would make the fog boil where it should stand still;
 * one tied to the cell gives the same answer for the same cell, and a still camera gives a still
 * image. The columns are laid out in screen space, so the seed follows the pixel rather than the
 * world - which is what keeps it from crawling while the camera moves, too.
 *
 * Shadows are binary. A stochastic transmittance through the medium is exactly the noise this grid
 * exists to avoid, and glass turning into an opaque blocker is the price: shafts of light come out
 * darker-edged than the path tracer draws them, but they hold still.
 *
 * A textured light or an emissive mesh needs a second kernel to be shaded and there is none here.
 * Rather than drop such a light entirely, it contributes the average emission the light tree
 * already keeps for it - see `volume_froxel_light_eval`. */
/* How far the medium reaches from `P` towards a light, taken as the distance to the first hull
 * crossed. For a medium that fills its hull evenly this is the whole of what shadows the light,
 * and without it the light arrives at every cell undimmed - which at this scene's optical depth is
 * a hundredfold overestimate and reads as fog lit from inside. */
ccl_device_inline float volume_froxel_medium_length(KernelGlobals kg,
                                                    ccl_private Ray *ccl_restrict ray,
                                                    ccl_private bool *ccl_restrict bounded)
{
  /* Whether a boundary was found at all, as opposed to the ray running to its end inside the
   * medium. It matters because the end for a distant light is infinitely far away, and a medium
   * assumed to reach that far leaves the cell perfectly black. */
  *bounded = false;

#ifdef __VOLUME_RECORD_ALL__
  Intersection hits[2 * MAX_VOLUME_STACK_SIZE];
  const uint num_hits = scene_intersect_volume(
      kg, ray, hits, uint(2 * MAX_VOLUME_STACK_SIZE), PATH_RAY_VISIBILITY_ALL);

  float nearest = ray->tmax;
  for (uint i = 0; i < num_hits; i++) {
    if (hits[i].t < nearest) {
      nearest = hits[i].t;
      *bounded = true;
    }
  }
  return nearest;
#else
  Intersection isect;
  if (scene_intersect_volume(kg, ray, &isect, PATH_RAY_VISIBILITY_ALL) && isect.t < ray->tmax) {
    *bounded = true;
    return isect.t;
  }
  return ray->tmax;
#endif
}

/* What a light emits, taken as the average that the light tree already keeps for it when the
 * emission is not a constant.
 *
 * `light_sample_shader_eval_nee_constant` answers only for a constant emission, and answering
 * "no" here means the cell receives nothing whatsoever from that light. That is a far wider hole
 * than it sounds: the flag is set only when both the colour and the strength of an Emission node
 * are plain numbers with nothing plugged into them, and a Principled BSDF is never constant even
 * with plain numbers at all. Lava driven by a texture, an emissive surface on a Principled shader
 * and any textured world light the fog not at all.
 *
 * The average is in the shader table either way - `constant_emission` is written from
 * `emission_estimate` whether or not the emission turned out to be constant - and it is the same
 * number the light tree picks this light by. Fog lit by the average of an emitter is nearer the
 * truth than fog not lit by it; what it does not carry is the pattern across the emitter, so a
 * band of lava comes out with the right power and a flat colour. */
ccl_device_inline bool volume_froxel_light_eval(KernelGlobals kg,
                                                const int shader_id,
                                                const int prim,
                                                const bool is_light,
                                                ccl_private Spectrum &eval)
{
  const int shader_index = shader_id & SHADER_MASK;
  const int shader_flag = kernel_data_fetch(shaders, shader_index).flags;

  if (!(shader_flag & SD_HAS_CONSTANT_EMISSION) &&
      !(kernel_data.froxel.fixes & VOLUME_FROXEL_FIX_EMITTER_ESTIMATE))
  {
    return false;
  }

  eval = rgb_to_spectrum(
      make_float3(kernel_data_fetch(shaders, shader_index).constant_emission[0],
                  kernel_data_fetch(shaders, shader_index).constant_emission[1],
                  kernel_data_fetch(shaders, shader_index).constant_emission[2]));

  if (is_light) {
    const ccl_global KernelLight *klight = &kernel_data_fetch(lights, prim);
    eval *= rgb_to_spectrum(
        make_float3(klight->strength[0], klight->strength[1], klight->strength[2]));
  }

  return true;
}

/* Whether anything solid stands between a cell and a light.
 *
 * One opaque shadow trace answers "was anything hit", and the boundary of the medium is something:
 * a container carrying a volume material stops the trace on its own face, so a light outside the
 * container reaches the fog not at all - not dimmed, not shadowed, simply absent. It is not a
 * problem the path tracer has, because a scene holding any volume at all switches to transparent
 * shadows, where every hit along the ray is examined instead of the first one ending it.
 *
 * So examine them. A hull whose shader is nothing but a volume is a boundary rather than a
 * blocker, and the ray carries on through it; the first hit that is anything else is the shadow.
 * The cheap trace comes first, so a light in plain view still costs one ray. */
/* The optical depth from a cell towards a light, taken by walking the baked density octree of one
 * medium.
 *
 * What this replaces takes the extinction found at the cell and applies it over the whole distance
 * the medium reaches, which is exact only for a medium that fills its hull evenly. A plume does
 * not. The cell that carries most of the visible light is a dense one, and it is shadowed as
 * though the entire container were as dense as its densest part - at the container sizes a smoke
 * simulation uses, that is orders of magnitude too dark, and it is the reason lit smoke reads as
 * a silhouette rather than as smoke.
 *
 * The walk costs no rays: the tree is already baked for the path tracer, its nodes carry the
 * extrema of the density inside them, and it answers the same way every frame - which is the whole
 * point of the grid. The midpoint of a node's extrema is an estimate rather than a bound; running
 * it with the minimum and with the maximum instead brackets the truth, which is a way to check it.
 */
ccl_device_inline float volume_froxel_optical_depth(KernelGlobals kg,
                                                    const ccl_private Ray *ccl_restrict ray,
                                                    const VolumeStack entry)
{
  /* A world volume has no object for a density to be baked against, so it has no octree either;
   * the caller falls back to the analytic form, which for a medium filling all of space is the
   * right answer anyway. Recognise it by its shader as well as its object index: with no world
   * volume in the scene that index is an ordinary object's, and testing it alone turned the walk
   * off for every medium in the scene. */
  const bool is_world = (kernel_data.background.volume_shader != SHADER_NONE) &&
                        (entry.object == kernel_data.background.object_index) &&
                        (entry.shader == (kernel_data.background.volume_shader & SHADER_MASK));
  if (entry.object == OBJECT_NONE || is_world) {
    return -1.0f;
  }

  const ccl_global KernelOctreeRoot *kroot = volume_find_octree_root(kg, entry);

  float3 local_P = ray->P;
  float3 local_D = ray->D;
  if (!(kernel_data_fetch(object_flag, entry.object) & SD_OBJECT_TRANSFORM_APPLIED)) {
    const Transform itfm = object_fetch_transform(kg, entry.object, OBJECT_INVERSE_TRANSFORM);
    local_P = transform_point(&itfm, ray->P);
    local_D = transform_direction(&itfm, ray->D);
  }

  OctreeTracing octree(ray->tmin);
  octree.node = &kernel_data_fetch(volume_tree_nodes, kroot->id);
  octree.entry = entry;

  if (!octree.to_octree_space(local_P, local_D, kroot->scale, kroot->translation)) {
    /* The ray is leaving the octree, or the octree has no shape to speak of: no medium along it. */
    return 0.0f;
  }

  volume_voxel_get(kg, octree);
  octree.t.max = octree.ray_voxel_intersect(ray->tmax);

  const float density = object_volume_density(kg, entry.object);

  float optical_depth = 0.0f;
  for (int step = 0; step < VOLUME_FROXEL_SHADOW_STEPS; step++) {
    const Extrema<float> sigma = octree.node->sigma * density;
    const float toward_max = kernel_data.froxel.shadow_sigma_mix;
    optical_depth += mix(sigma.min, sigma.max, toward_max) * (octree.t.max - octree.t.min);

    if (octree.t.max >= ray->tmax) {
      break;
    }

    if (octree.next_scale > MANTISSA_BITS) {
      /* Out through the far side of the tree, and the tree is the medium: what lies beyond it on
       * the way to the light is empty. The path tracer's own walk carries the root's extrema past
       * this point, but it is bounded by the volume segment it was handed, whereas this ray runs
       * all the way to the light - carrying the medium along with it made a plume thirty times its
       * own opacity. */
      break;
    }

    /* Climb to the ancestor the two leaves share, then descend into the next one. */
    for (; octree.scale < octree.next_scale; octree.scale++) {
      if (octree.node->parent == -1) {
        return optical_depth;
      }
      octree.node = &kernel_data_fetch(volume_tree_nodes, octree.node->parent);
    }
    volume_voxel_get(kg, octree);
    octree.t.min = octree.t.max;
    octree.t.max = octree.ray_voxel_intersect(ray->tmax);
  }

  return optical_depth;
}

ccl_device_inline bool volume_froxel_shadowed(KernelGlobals kg,
                                              const ccl_private Ray *ccl_restrict shadow_ray)
{
  Ray ray = *shadow_ray;
  if (!scene_intersect_shadow(kg, &ray, PATH_RAY_VISIBILITY_SHADOW_OPAQUE)) {
    return false;
  }
  if (!(kernel_data.froxel.fixes & VOLUME_FROXEL_FIX_HULL_SHADOW)) {
    return true;
  }

  Intersection isect;
  for (int step = 0; step < 2 * MAX_VOLUME_STACK_SIZE; step++) {
    if (!scene_intersect(kg, &ray, PATH_RAY_VISIBILITY_SHADOW_OPAQUE, &isect)) {
      return false;
    }
    if (!(intersection_get_shader_flags(kg, isect.prim, isect.type) & SD_HAS_ONLY_VOLUME)) {
      return true;
    }
    ray.tmin = intersection_t_offset(isect.t);
    ray.self.object = isect.object;
    ray.self.prim = isect.prim;
  }

  /* More boundaries than the stack can hold: say shadowed, which is what the trace said before. */
  return true;
}

ccl_device_inline float3 volume_froxel_in_scatter(KernelGlobals kg,
                                                  ccl_private ShaderData *sd,
                                                  const VolumeStack entry,
                                                  const float3 P,
                                                  const float sigma_t,
                                                  const uint seed)
{
  if (!kernel_data.integrator.use_direct_light || !(sd->flag & SD_SCATTER)) {
    return zero_float3();
  }

  const float3 rand = make_float3(
      hash_uint2_to_float(seed, 0u), hash_uint2_to_float(seed, 1u), hash_uint2_to_float(seed, 2u));

  LightSample ls ccl_optional_struct_init;
  if (!light_sample_from_position(kg,
                                  rand,
                                  sd->time,
                                  P,
                                  zero_float3(),
                                  light_link_receiver_nee(kg, sd),
                                  SD_BSDF_HAS_TRANSMISSION,
                                  0,
                                  PATH_RAY_FLAG_NONE,
                                  &ls))
  {
    return zero_float3();
  }

  if (ls.shader & SHADER_EXCLUDE_SCATTER) {
    return zero_float3();
  }

  Spectrum light_eval ccl_optional_struct_init;
  if (!volume_froxel_light_eval(kg, ls.shader, ls.prim, ls.type != LIGHT_TRIANGLE, light_eval)) {
    return zero_float3();
  }

  /* Every scattering closure weighed by its own phase towards the camera. `sd->wi` points at the
   * viewer, which is what the phase functions expect, so this is the fog seen from where the
   * column was cast and not from anywhere else. */
  Spectrum in_scatter = zero_spectrum();
  /* The scattering coefficient on its own, without the phase. Against the extinction it gives the
   * single-scattering albedo of this cell, which is what says how much light there is in the
   * orders the grid does not follow. */
  Spectrum sigma_s = zero_spectrum();
  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];
    /* Scattering closures only - absorption has no phase to evaluate. */
    if (!CLOSURE_IS_VOLUME_SCATTER(sc->type)) {
      continue;
    }

    float phase_pdf = 0.0f;
    const Spectrum phase = volume_phase_eval(
        sd, (const ccl_private ShaderVolumeClosure *)sc, ls.D, &phase_pdf);
    in_scatter += sc->weight * phase;
    sigma_s += sc->weight;
  }

  if (is_zero(in_scatter) || !(ls.pdf > 0.0f)) {
    return zero_float3();
  }

  Ray shadow_ray ccl_optional_struct_init;
  shadow_ray.P = P;
  shadow_ray.D = ls.D;
  shadow_ray.tmin = 0.0f;
  shadow_ray.tmax = ls.t;
  shadow_ray.time = sd->time;
  shadow_ray.self.object = OBJECT_NONE;
  shadow_ray.self.prim = PRIM_NONE;
  shadow_ray.self.light_object = ls.object;
  shadow_ray.self.light_prim = ls.prim;
#ifdef __RAY_DIFFERENTIALS__
  shadow_ray.dP = differential_zero_compact();
  shadow_ray.dD = differential_zero_compact();
#endif

  if (volume_froxel_shadowed(kg, &shadow_ray)) {
    return zero_float3();
  }

  /* What the medium itself takes out of the light on its way in. Taken analytically from the
   * extinction at the cell over the distance the medium reaches, rather than by marching: one
   * value per cell, the same every frame, and no variance to reconstruct. It is only exact for a
   * medium that fills its hull evenly - a wisp of smoke is shadowed as if it were a solid block of
   * it - and that is the trade the grid makes everywhere else too. */
  float medium_transmittance = 1.0f;
  float optical_depth = 0.0f;
  if (sigma_t > 0.0f) {
    Ray hull_ray = shadow_ray;
    hull_ray.self.light_object = OBJECT_NONE;
    hull_ray.self.light_prim = PRIM_NONE;
    const float marched = (kernel_data.froxel.fixes & VOLUME_FROXEL_FIX_MARCHED_SHADOW) ?
                              volume_froxel_optical_depth(kg, &hull_ray, entry) :
                              -1.0f;
    if (marched >= 0.0f) {
      optical_depth = marched;
      medium_transmittance = expf(-optical_depth);
    }
    else {
      bool bounded = false;
      const float medium_length = volume_froxel_medium_length(kg, &hull_ray, &bounded);

      /* No boundary means the ray never left the medium as far as the trace could tell, and the
       * distance stands in for one: for a light at a finite distance that is the whole way to it,
       * and for a distant light it is infinity, which makes the cell black however thin the smoke
       * is. Treating what was not found as empty is the lesser error of the two. */
      if (bounded || !(kernel_data.froxel.fixes & VOLUME_FROXEL_FIX_UNBOUNDED_SHADOW)) {
        optical_depth = sigma_t * medium_length;
        medium_transmittance = expf(-optical_depth);
      }
    }
  }

  /* What the light does after it has scattered here once. The grid follows one scattering event
   * and stops, and in anything but soot that is where much of the brightness lives: the orders
   * beyond the first carry a factor of the albedo each, so they sum as a geometric series. A
   * medium too thin to scatter twice gets none of it, which is what the optical depth towards the
   * light says - the same depth the attenuation above is taken from, so one walk pays for both.
   *
   * This is a stand-in, not transport. Every real-time volumetric uses one, because the honest
   * answer needs a second light transport and there is none here. */
  if ((kernel_data.froxel.fixes & VOLUME_FROXEL_FIX_MULTI_SCATTER) && sigma_t > 0.0f) {
    const float albedo = min(reduce_max(sigma_s) / sigma_t, 0.999f);

    /* How much brighter a medium of this albedo is once every order of scattering is counted
     * rather than only the first. For a semi-infinite isotropic medium that ratio is
     * Chandrasekhar's H function squared, and its standard rational approximation is within a
     * percent across the range: at an albedo of 0.5 it gives 1.55 against a tabulated 1.57, at 0.9
     * it gives 3.38 against 3.42. Which is why this is not fitted - the physics is cheaper than
     * the fit, and it does not have to be redone for every medium. */
    const float h = 3.0f / (1.0f + 2.0f * sqrtf(1.0f - albedo));
    const float saturated = h * h;

    /* A medium too thin to scatter twice gets none of it. The depth towards the light is the one
     * the attenuation above already walked for, so this costs nothing. */
    const float depth = 1.0f - expf(-2.0f * optical_depth);
    in_scatter *= 1.0f + (saturated - 1.0f) * depth * kernel_data.froxel.multi_scatter_return;
  }

  return spectrum_to_rgb(in_scatter * light_eval * (medium_transmittance * ls.eval_fac / ls.pdf));
}

/* Injection: what the medium is inside every cell of one column.
 *
 * One thread per column rather than per cell, because the spans above are found by walking the
 * column once. */
ccl_device void volume_froxel_inject(KernelGlobals kg, const int column_index)
{
  const int res_x = kernel_data.froxel.res_x;
  const int res_y = kernel_data.froxel.res_y;
  const int res_z = kernel_data.froxel.res_z;

  const int y = column_index / res_x;
  const int x = column_index - y * res_x;
  if (x >= res_x || y >= res_y) {
    return;
  }

  const int stage = kernel_data.froxel.stage;

  /* Stage 1 writes a fixed medium and touches nothing else, which is what says whether the arrays
   * are bound at all. Later stages add one step each, so a fault can be attributed rather than
   * guessed at. */
  if (stage == 1) {
    for (int z = 0; z < res_z; z++) {
      volume_froxel_write_scatter(
          kg, volume_froxel_index(kg, x, y, z), make_float4(zero_float3(), 0.05f));
    }
    return;
  }

  /* The centre of the tile this column stands for. No jitter: the grid has to be the same from one
   * frame to the next, or the fog crawls. */
  const int tile = kernel_data.froxel.tile;
  const int pixel_x = x * tile + tile / 2;
  const int pixel_y = y * tile + tile / 2;

  Ray ray;
  int cache_miss = 0;
  camera_sample(kg,
                pixel_x,
                pixel_y,
                make_float2(0.5f, 0.5f),
                0.5f,
                make_float2(0.5f, 0.5f),
                &ray,
                cache_miss);
  ray.tmin = kernel_data.froxel.t_near;
  ray.tmax = kernel_data.froxel.t_far;

  if (stage == 2) {
    for (int z = 0; z < res_z; z++) {
      volume_froxel_write_scatter(
          kg, volume_froxel_index(kg, x, y, z), make_float4(zero_float3(), 0.05f));
    }
    return;
  }

  VolumeFroxelSpan spans[MAX_VOLUME_STACK_SIZE];
  const int num_spans = volume_froxel_collect_spans(kg, &ray, spans, MAX_VOLUME_STACK_SIZE);

  if (stage == 3) {
    /* The hulls were walked; the medium inside them is stood in for rather than asked about, so
     * anything that goes wrong from here on belongs to the shader evaluation. */
    for (int z = 0; z < res_z; z++) {
      const float t_mid = 0.5f * (volume_froxel_slice_distance(kg, z) +
                                  volume_froxel_slice_distance(kg, z + 1));
      float sigma_t = 0.0f;
      for (int i = 0; i < num_spans; i++) {
        if (t_mid >= spans[i].t_enter && t_mid < spans[i].t_exit) {
          sigma_t += 0.05f;
        }
      }
      volume_froxel_write_scatter(
          kg, volume_froxel_index(kg, x, y, z), make_float4(zero_float3(), sigma_t));
    }
    return;
  }

  for (int z = 0; z < res_z; z++) {
    const int index = volume_froxel_index(kg, x, y, z);

    const float t_front = volume_froxel_slice_distance(kg, z);
    const float t_back = volume_froxel_slice_distance(kg, z + 1);
    const float t_mid = 0.5f * (t_front + t_back);

    float3 emission = zero_float3();
    float sigma_t = 0.0f;

    for (int i = 0; i < num_spans; i++) {
      /* How much of this slice the medium actually covers, and where in it to ask. Deciding by
       * whether the midpoint lands inside instead - which is what this did - makes a slice either
       * wholly full or wholly empty: a sheet of mist thinner than a slice disappears, and the
       * slices are 11 percent apart, so far from the camera they are metres thick. The segment
       * lookup on the other end of the grid already weighs its ends by their overlap. */
      float coverage = 1.0f;
      float t_sample = t_mid;
      if (kernel_data.froxel.fixes & VOLUME_FROXEL_FIX_SLICE_OVERLAP) {
        const float t_lo = max(t_front, spans[i].t_enter);
        const float t_hi = min(t_back, spans[i].t_exit);
        if (!(t_hi > t_lo)) {
          continue;
        }
        coverage = (t_hi - t_lo) / max(t_back - t_front, 1e-12f);
        t_sample = 0.5f * (t_lo + t_hi);
      }
      else if (t_mid < spans[i].t_enter || t_mid >= spans[i].t_exit) {
        continue;
      }

      /* Coefficients where the medium is, with no path to carry them: the volume shader takes the
       * stack entry directly, which is what the density bake already relies on. */
      Ray point_ray = ray;
      point_ray.P = ray.P + ray.D * t_sample;
      point_ray.tmin = 0.0f;
      point_ray.tmax = 0.0f;

      ShaderData sd;
      shader_setup_from_volume(&sd, &point_ray, spans[i].object);
      /* `shader_setup_from_volume` leaves the closure count alone - the stock path resets it in
       * `volume_shader_eval`, which is bypassed here. Left uninitialised, the shader writes its
       * closures at whatever index the stack happened to hold, which is an illegal address named
       * after the whole kernel and traceable to nothing. */
      sd.num_closure = 0;
      sd.num_closure_left = kernel_data.max_closures;
      sd.flag = SD_IS_VOLUME_SHADER_EVAL;
      sd.object_flag = 0;
      sd.closure_transparent_extinction = zero_float3();
      sd.closure_emission_background = zero_float3();

      const VolumeStack entry = {spans[i].object, spans[i].shader};
      ConstIntegratorBakeState state;
      volume_shader_eval_entry<false,
                               KERNEL_FEATURE_NODE_MASK_VOLUME & ~KERNEL_FEATURE_NODE_LIGHT_PATH>(
          kg, state, &sd, entry, PATH_RAY_VISIBILITY_CAMERA, PATH_RAY_FLAG_NONE);

      /* The extinction where the medium is, and the same averaged over the whole slice. Only the
       * first describes the medium, so the first is what shadows the light; the second is what the
       * slice as a whole carries, and it is what the grid stores. */
      const float entry_sigma_t = reduce_max(sd.closure_transparent_extinction);
      sigma_t += entry_sigma_t * coverage;
      emission += sd.closure_emission_background * coverage;

      /* Light gathered here is added to what the medium emits: both are radiance per unit length
       * along the column, and the segment lookup integrates them the same way. */
      const int light_samples = kernel_data.froxel.light_samples;
      for (int s = 0; s < light_samples; s++) {
        const uint seed = hash_uint4(uint(x), uint(y), uint(z), uint(s) + uint(i) * 8u);
        emission += volume_froxel_in_scatter(kg, &sd, entry, point_ray.P, entry_sigma_t, seed) *
                    (coverage / float(light_samples));
      }
    }

    volume_froxel_write_scatter(kg, index, make_float4(emission, sigma_t));
  }
}

/* What the grid says a camera ray segment carries: the light it gathers and what it lets through.
 *
 * Marched across the slices the segment touches, each contributing only its overlap, taken
 * analytically from the extinction stored there - snapping to slice boundaries instead is what puts
 * a halo around every silhouette.
 *
 * A prefix sum along the column would make this two lookups instead of a march, and it was written
 * that way first. It divides by the transmittance at the near end of the segment, and inside a
 * dense medium that is precisely the quantity that has gone to nothing: at this scene's optical
 * depth it is already 0.009, and deeper fog drives it under what a float can carry. The march never
 * performs that division. The grid is 2.4 MB and sits in L2, so what it costs is cache hits. */
ccl_device_inline void volume_froxel_segment(KernelGlobals kg,
                                             const float x,
                                             const float y,
                                             const float t_start,
                                             const float t_end,
                                             ccl_private float3 *scattered,
                                             ccl_private float3 *transmittance)
{
  *scattered = zero_float3();
  *transmittance = one_float3();

  const int res_z = kernel_data.froxel.res_z;
  if (!(t_end > t_start)) {
    return;
  }

  /* Walk the slices the segment touches. Partial slices at either end contribute only their
   * overlap, which is what keeps the answer continuous as the segment's end moves. */
  const float coord_start = volume_froxel_slice_coordinate(kg, t_start);
  const float coord_end = volume_froxel_slice_coordinate(kg, t_end);

  const int first = max(int(coord_start), 0);
  const int last = min(int(coord_end), res_z - 1);

  float3 accumulated = zero_float3();
  float3 through = one_float3();

  for (int z = first; z <= last; z++) {
    const float slice_front = volume_froxel_slice_distance(kg, z);
    const float slice_back = volume_froxel_slice_distance(kg, z + 1);

    const float from = max(slice_front, t_start);
    const float to = min(slice_back, t_end);
    if (!(to > from)) {
      continue;
    }

    const float4 cell = volume_froxel_sample_scatter(kg, x, y, z);
    const float sigma_t = cell.w;
    const float3 emission = make_float3(cell.x, cell.y, cell.z);
    const float length = to - from;

    if (sigma_t > 0.0f) {
      const float slice_transmittance = expf(-sigma_t * length);
      accumulated += through * emission * ((1.0f - slice_transmittance) / sigma_t);
      through *= slice_transmittance;
    }
    else {
      accumulated += through * emission * length;
    }
  }

  *scattered = accumulated;
  *transmittance = through;
}

/* Which column of the grid a pixel belongs to. */
/* Where a pixel falls among the columns, as a fraction rather than a choice of one: column `x`
 * stands for the pixel at its centre, and everything between two columns belongs partly to each.
 */
ccl_device_inline void volume_froxel_column(KernelGlobals kg,
                                            const int pixel_x,
                                            const int pixel_y,
                                            ccl_private float *x,
                                            ccl_private float *y)
{
  const int tile = max(kernel_data.froxel.tile, 1);
  *x = float(pixel_x - tile / 2) / float(tile);
  *y = float(pixel_y - tile / 2) / float(tile);
}

/* Let the grid answer for this segment instead of tracing the medium.
 *
 * Returns false when the grid does not apply, and the caller falls through to the honest path. When
 * it does apply, the light the medium adds is written and the path's throughput is attenuated -
 * then the caller returns `VOLUME_PATH_ATTENUATED`, the path carries on to the surface behind the
 * fog, and that surface gets multiplied by the same transmittance. The pixel therefore ends up as
 * `surface * T + S`, with the surface itself still traced, so its silhouette stays as sharp as it
 * would be with no fog at all.
 *
 * Only the camera segment: a grid built along camera columns says nothing about where a bounced ray
 * goes, and using it there would be inventing light rather than approximating it. */
ccl_device_inline bool volume_froxel_shade(KernelGlobals kg,
                                           IntegratorState state,
                                           const ccl_private Ray *ccl_restrict ray,
                                           ccl_global float *ccl_restrict render_buffer)
{
  if (!kernel_data.froxel.enabled) {
    return false;
  }
  if (INTEGRATOR_STATE(state, path, bounce) != 0) {
    return false;
  }

  /* Handing back over to the path tracer once the view settles: each pixel picks one estimate or
   * the other, rather than the two being averaged. Averaging would mean paying for both, and the
   * expectation of this choice is already `blend * grid + (1 - blend) * traced` - an interpolation
   * between two estimates rather than a blend of two pictures. The draw is redrawn every sample,
   * so over the fade the picture crosses over smoothly, and at `blend` of zero nothing here runs.
   */
  const float blend = kernel_data.froxel.blend;
  if (blend < 1.0f) {
    const uint pixel = INTEGRATOR_STATE(state, path, render_pixel_index);
    const uint sample = INTEGRATOR_STATE(state, path, sample);
    if (!(hash_uint2_to_float(pixel, sample) < blend)) {
      return false;
    }
  }
  const int width = max(int(kernel_data.cam.width), 1);
  const uint pixel_index = INTEGRATOR_STATE(state, path, render_pixel_index);
  const int pixel_x = int(pixel_index) % width;
  const int pixel_y = int(pixel_index) / width;

  float x;
  float y;
  volume_froxel_column(kg, pixel_x, pixel_y, &x, &y);

  const float t_start = max(ray->tmin, kernel_data.froxel.t_near);
  const float t_end = min(ray->tmax, kernel_data.froxel.t_far);

  float3 scattered;
  float3 transmittance;
  volume_froxel_segment(kg, x, y, t_start, t_end, &scattered, &transmittance);

  const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput);

  if (!is_zero(scattered)) {
    /* `film_write_volume_emission` does not apply throughput itself. */
    film_write_volume_emission(
        kg, state, throughput * rgb_to_spectrum(scattered), render_buffer, LIGHTGROUP_NONE);
  }

  INTEGRATOR_STATE_WRITE(state, path, throughput) = throughput * rgb_to_spectrum(transmittance);

  return true;
}

CCL_NAMESPACE_END
