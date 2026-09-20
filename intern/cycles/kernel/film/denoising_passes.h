/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/closure/bsdf.h"

#include "kernel/film/write.h"

CCL_NAMESPACE_BEGIN

#ifdef __DENOISING_FEATURES__

/* Whether the depth pass describes one place per pixel instead of the length of a wandering path.
 *
 * `..._FOLLOW_REFLECTIONS` makes depth accumulate ray length rather than camera z, and keeps
 * accumulating past the first bounce. Down a mirror that is what a denoiser wants. Through a dense
 * medium it is not: a scattered sample bounces around inside the fog and every further crossing of
 * the hull adds its own length, so the number a pixel ends up with depends on where the scatter
 * happened to go. Measured on the owner's scene, that alone left 66% of pixels with a different
 * depth from one iteration to the next while the camera stood still.
 *
 * NGX is told the depth is linear (`NVSDK_NGX_DLSS_Depth_Type_Linear`), and linear depth means
 * camera z - ray length only equals it down the centre of the frame. */
ccl_device_forceinline bool denoising_depth_is_primary_only(KernelGlobals kg)
{
  return (kernel_data.film.denoising_pass_options_flag &
          DENOISING_PASS_VOLUME_DEPTH_SEGMENT_END) != 0;
}

/* Whether the medium already described this pixel's depth, so nothing the path reaches afterwards
 * should add to it.
 *
 * `PATH_RAY_VOLUME_PRIMARY_TRANSMIT` marks a primary segment that entered a medium and is dropped
 * by the first scatter, so it is exactly the transmitted branch - the one that would otherwise walk
 * on and write the depth of whatever it finds, while the scattered branch stops in the fog. Both
 * branches then land on the medium's own point instead. */
ccl_device_forceinline bool denoising_depth_owned_by_medium(KernelGlobals kg,
                                                            const uint32_t path_flag)
{
  return (kernel_data.film.denoising_pass_options_flag &
          DENOISING_PASS_VOLUME_DEPTH_REPRESENTATIVE) != 0 &&
         (path_flag & PATH_RAY_VOLUME_PRIMARY_TRANSMIT) != 0;
}

ccl_device_forceinline float denoising_depth_compute(KernelGlobals kg,
                                                     IntegratorState state,
                                                     const ccl_private ShaderData *sd,
                                                     const Spectrum denoising_feature_throughput,
                                                     bool follow_reflections)
{
  if (denoising_depth_is_primary_only(kg)) {
    follow_reflections = false;
  }
  float depth;
  const float d = sd->ray_length - INTEGRATOR_STATE(state, ray, tmin);
  if (follow_reflections) {
    /* Write the ray length minus tmin. */
    depth = d;
  }
  else {
    /* Write the camera z depth. */
    const float3 prev_P = sd->P + sd->wi * d;
    const float prev_depth = camera_z_depth(kg, prev_P);
    const float new_depth = camera_z_depth(kg, sd->P);
    depth = new_depth - prev_depth;
  }

  return ensure_finite(depth * average(denoising_feature_throughput));
}

ccl_device_forceinline void film_write_denoising_features_surface(KernelGlobals kg,
                                                                  IntegratorState state,
                                                                  const ccl_private ShaderData *sd,
                                                                  ccl_global float *ccl_restrict
                                                                      render_buffer)
{
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  if (!(path_flag & PATH_RAY_DENOISING_FEATURES)) {
    return;
  }

  /* Don't write denoising passes for paths that were split off for shadow catchers
   * to avoid double-counting. */
  if (path_flag & PATH_RAY_SHADOW_CATCHER_PASS) {
    return;
  }

  const bool use_albedo_roughness_weighting = (kernel_data.film.denoising_pass_options_flag &
                                               DENOISING_PASS_USE_ALBEDO_ROUGHNESS_WEIGHTING) != 0;

  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);

  float3 normal = zero_float3();
  Spectrum diffuse_albedo = zero_spectrum();
  Spectrum specular_albedo = zero_spectrum();
  Spectrum transparent_albedo = zero_spectrum();
  float specular_roughness = 0.0f;
  float sum_weight = 0.0f;
  float sum_nonspecular_weight = 0.0f;

  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (!CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      continue;
    }

    /* Transparency always passes through. */
    if (CLOSURE_IS_BSDF_TRANSPARENT(sc->type)) {
      transparent_albedo += sc->weight;
      continue;
    }

    const Spectrum closure_albedo = bsdf_albedo(kg, sd, sc, true, true);
    const float closure_weight = average(closure_albedo);

    /* All closures contribute to the normal feature, but only diffuse-like ones to the albedo. */
    /* If far-field hair, use fiber tangent as feature instead of normal. */
    normal += (sc->type == CLOSURE_BSDF_HAIR_HUANG_ID ? safe_normalize(sd->dPdu) : sc->N) *
              closure_weight;

    /* bsdf_get_specular_roughness_squared returns GGX alpha squared (alpha_x*alpha_y). Use sqrtf
     * to get GGX alpha. */
    const float roughness = sqrtf(bsdf_get_specular_roughness_squared(sc));

    /* Transition smoothly from specular to diffuse between 0.0 and 0.15 roughness. */
    const float diffuse_weight = (sc->type == CLOSURE_BSDF_HAIR_HUANG_ID) ?
                                     1.0f :
                                     smoothstep(0.0f, 0.15f, roughness);

    if (use_albedo_roughness_weighting) {
      diffuse_albedo += closure_albedo * diffuse_weight;
      specular_albedo += closure_albedo * (1.0f - diffuse_weight);
    }
    else if (CLOSURE_IS_BSDF_DIFFUSE(sc->type) || CLOSURE_IS_BSSRDF(sc->type)) {
      diffuse_albedo += closure_albedo;
    }
    else if (CLOSURE_IS_BSDF_GLOSSY(sc->type) || CLOSURE_IS_GLASS(sc->type)) {
      specular_albedo += closure_albedo;
    }
    /* Apply sqrtf again to convert GGX alpha to perceptual roughness. */
    specular_roughness += sqrtf(roughness) * closure_weight;

    sum_weight += closure_weight;
    sum_nonspecular_weight += closure_weight * diffuse_weight;
  }

  /* Fraction of non-transparent closures, for smooth blending at transparent surfaces. */
  const float transparent_weight = average(transparent_albedo);
  const float total_weight = sum_weight + transparent_weight;

  /* Blend between writing features at this bounce vs. deferring to the next bounce based
   * on the proportion of diffuse closures. Smoothly transition between 0.0 and 0.5 diffuse
   * fraction. */
  float feature_weight = 0.0f;
  /* A surface can reach here with no BSDF at all - a shader with only an Emission node is the
   * common case, and an emissive mesh of many triangles is exactly where this matters. Emission
   * never enters `sd->closure[]` (see `emission_setup`), so the loop above ran zero times: the
   * normal is still zero and the roughness is still zero. Zero roughness is how a mirror is
   * described, and a zero normal is not a direction at all, so reconstruction reads such a pixel
   * as a specular surface with invalid geometry, switches to specular reprojection and filters it
   * weakly - which shows up as blotches over the emissive geometry.
   *
   * The ordinary Normal render pass has never had this problem: `surface_shader_average_normal`
   * falls back to `sd->N` and `surface_shader_average_roughness` falls back to 1.0. Do the same
   * here. `sd->N` is already unit length, in world space, and flipped for backfacing hits. */
  const int emissive_features = kernel_data.integrator.denoising_emissive_features;
  const bool has_bsdf = (sum_weight > 0.0f);
  const bool write_emissive_normal = !has_bsdf && (emissive_features & 1) != 0;
  if (has_bsdf) {
    normal /= sum_weight;
    specular_roughness /= sum_weight;

    feature_weight = smoothstep(0.0f, 0.5f, sum_nonspecular_weight / sum_weight);
  }
  else {
    if (write_emissive_normal) {
      normal = sd->N;
    }
    if ((emissive_features & 2) != 0) {
      /* Fully rough - a phase-function-like answer rather than a mirror. The write below
       * multiplies this by the feature throughput, the same as for a surface that does have
       * closures, so partially covered pixels stay in step with their neighbours. */
      specular_roughness = 1.0f;
    }
  }

  float deferred_feature_weight = 1.0f - feature_weight;

  /* Whether to defer features to the next bounce for individual passes. */
  const bool follow_reflections = (kernel_data.film.denoising_pass_options_flag &
                                   DENOISING_PASS_FOLLOW_REFLECTIONS) != 0;
  if (!follow_reflections) {
    feature_weight = 1.0f;
  }

  const bool is_first_bounce = INTEGRATOR_STATE(state, path, bounce) == 0;

  const Spectrum denoising_feature_throughput = INTEGRATOR_STATE(
      state, path, denoising_feature_throughput);

  if (is_first_bounce || follow_reflections) {
    /* Depth stops at the primary segment when it has to describe one place - see
     * `denoising_depth_is_primary_only`. The normal and albedo below keep following reflections
     * either way: they are what a surface behind the fog contributes on the rare transmitted
     * sample, and that is the only place they come from at all. */
    if (kernel_data.film.pass_denoising_depth != PASS_UNUSED &&
        (is_first_bounce || !denoising_depth_is_primary_only(kg)) &&
        !denoising_depth_owned_by_medium(kg, path_flag))
    {
      const float denoising_depth = denoising_depth_compute(
          kg, state, sd, denoising_feature_throughput, follow_reflections);
      film_write_pass_float(buffer + kernel_data.film.pass_denoising_depth, denoising_depth);
    }

    /* `write_emissive_normal` admits the fallback above: `feature_weight` is zero there because it
     * measures the diffuse fraction of the closures, and an emissive surface has none. */
    if (kernel_data.film.pass_denoising_normal != PASS_UNUSED &&
        (feature_weight > 0.0f || write_emissive_normal))
    {
      /* Transform normal into camera space. */
      const Transform worldtocamera = kernel_data.cam.worldtocamera;
      float3 denoising_normal = transform_direction(&worldtocamera, normal);
      const float opaque_fraction = (total_weight > 0.0f) ? (sum_weight / total_weight) : 1.0f;
      /* The fallback normal stands on its own, so it is not scaled down by a weight describing
       * closures that are not there. */
      const float normal_weight = has_bsdf ? feature_weight : 1.0f;

      denoising_normal = ensure_finite(denoising_normal * opaque_fraction * normal_weight *
                                       average(denoising_feature_throughput));
      film_write_pass_float3(buffer + kernel_data.film.pass_denoising_normal, denoising_normal);
    }

    if (kernel_data.film.pass_denoising_albedo != PASS_UNUSED && feature_weight > 0.0f) {
      const Spectrum denoising_albedo = ensure_finite(diffuse_albedo * feature_weight *
                                                      denoising_feature_throughput);
      film_write_pass_spectrum(buffer + kernel_data.film.pass_denoising_albedo, denoising_albedo);
    }
  }

  if (is_first_bounce) {
    if (kernel_data.film.pass_denoising_roughness != PASS_UNUSED) {
      const float denoising_roughness = ensure_finite(specular_roughness *
                                                      average(denoising_feature_throughput));
      film_write_pass_float(buffer + kernel_data.film.pass_denoising_roughness,
                            denoising_roughness);
    }

    if (kernel_data.film.pass_denoising_specular_albedo != PASS_UNUSED) {
      const Spectrum denoising_specular_albedo = ensure_finite(specular_albedo *
                                                               denoising_feature_throughput);
      film_write_pass_spectrum(buffer + kernel_data.film.pass_denoising_specular_albedo,
                               denoising_specular_albedo);
    }

    if (kernel_data.film.pass_denoising_backward_motion != PASS_UNUSED) {
      const float3 backward_motion = primitive_motion_vector_backward_depth_delta(kg, sd);
      film_write_pass_float3(buffer + kernel_data.film.pass_denoising_backward_motion,
                             backward_motion);
    }
  }
  else if (INTEGRATOR_STATE(state, path, bounce) == 1 && (path_flag & PATH_RAY_REFLECT)) {
    if (kernel_data.film.pass_denoising_specular_motion != PASS_UNUSED) {
      const float3 reflector_P = INTEGRATOR_STATE(state, ray, P);
      const float3 reflector_N = INTEGRATOR_STATE(state, path, mis_origin_n);

      const float4 denoising_specular_motion = primitive_motion_vector_reflection(
          kg, reflector_P, reflector_N, sd);
      film_write_pass_float(buffer + kernel_data.film.pass_denoising_specular_motion + 0,
                            denoising_specular_motion.x);
      film_write_pass_float(buffer + kernel_data.film.pass_denoising_specular_motion + 1,
                            denoising_specular_motion.y);
    }
  }

  /* Portion deferred to the next bounce. Specularity uses the feature weight, transparent
   * always passes through. */
  Spectrum deferred_albedo = specular_albedo * deferred_feature_weight + transparent_albedo;

  /* When not following reflections, but the specular motion pass is enabled, still need to
   * continue to the first bounce, but with no weight for the albedo pass. */
  if (!follow_reflections && kernel_data.film.pass_denoising_specular_motion == PASS_UNUSED) {
    deferred_albedo = transparent_albedo;
  }
  if (reduce_max(fabs(deferred_albedo)) > 1e-4f) {
    if (!follow_reflections) {
      deferred_albedo = transparent_albedo;
    }
    INTEGRATOR_STATE_WRITE(state, path, denoising_feature_throughput) *= deferred_albedo;
  }
  else {
    INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_DENOISING_FEATURES;
  }
}

ccl_device_forceinline void film_write_denoising_features_surface_volume(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_private ShaderData *sd,
    ccl_global float *ccl_restrict render_buffer)
{
  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);

  const bool follow_reflections = (kernel_data.film.denoising_pass_options_flag &
                                   DENOISING_PASS_FOLLOW_REFLECTIONS) != 0;
  const bool is_first_bounce = INTEGRATOR_STATE(state, path, bounce) == 0;

  const Spectrum denoising_feature_throughput = INTEGRATOR_STATE(
      state, path, denoising_feature_throughput);

  /* The entry crossing still writes: the flag is set inside the volume kernel, which runs after
   * this, so only the crossing on the way out is silenced. */
  if ((is_first_bounce || follow_reflections) &&
      (is_first_bounce || !denoising_depth_is_primary_only(kg)) &&
      !denoising_depth_owned_by_medium(kg, INTEGRATOR_STATE(state, path, flag)))
  {
    if (kernel_data.film.pass_denoising_depth != PASS_UNUSED) {
      const float denoising_depth = denoising_depth_compute(
          kg, state, sd, denoising_feature_throughput, follow_reflections);
      film_write_pass_float(buffer + kernel_data.film.pass_denoising_depth, denoising_depth);
    }
  }
}

ccl_device_forceinline void film_write_denoising_features_volume(KernelGlobals kg,
                                                                 IntegratorState state,
                                                                 const Spectrum albedo,
                                                                 const bool scatter,
                                                                 const float3 reference_P,
                                                                 const float depth_delta,
                                                                 const float3 reference_N,
                                                                 const float coverage,
                                                                 ccl_global float *ccl_restrict
                                                                     render_buffer)
{
  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);

  const Spectrum denoising_feature_throughput = INTEGRATOR_STATE(
      state, path, denoising_feature_throughput);

  /* Where the radiance comes from scattering, give the reconstruction a motion vector.
   *
   * Without one the pixel arrives at NGX as if it had not moved: a path that scatters loses
   * `PATH_RAY_TRANSPARENT_BACKGROUND` (`path_state.h`), and without that flag neither the surface
   * behind the volume nor the background reaches `film_write_data_passes`, which is the only place
   * a motion vector is written. Temporal reconstruction then holds the previous frame over the
   * volume, which is the smear seen on fog while the camera moves. Measured on a 20 m fog cube:
   * 28.5% of the frame carried no motion at all, against 0% with the fog hidden.
   *
   * `reference_P` is where the ray entered the medium (`ray->P + ray->D * ray->tmin` - crossing a
   * volume boundary moves only `tmin`, see `integrate_surface_volume_only_bounce`, so `ray->P` is
   * still the camera), not where this sample happened to scatter.
   * The scatter point is drawn at random each sample, so a vector built from it jitters frame to
   * frame; reconstruction then cannot hold a stable history and the fog turns grainy - which is
   * exactly what the first version of this did. The entry point is the same every sample, and it
   * agrees with the depth the volume's bounding surface already writes
   * (`film_write_denoising_features_surface_volume`), so depth and motion describe one place.
   *
   * That last sentence stops being true under `..._SEGMENT_END`, which moves the vector to the end
   * of the segment while the depth stays on the front face. `depth_delta` is what pays the
   * difference back - see `volume_denoising_reference_depth_delta`.
   *
   * Written only at the first bounce, like the surface features, and only when the denoiser asked
   * for it - the Vector pass keeps its usual contents for everyone else, including a render that
   * merely stores the denoising passes for output.
   *
   * The motion is that point reprojected by the camera alone. A volume carried by a simulation
   * moves its medium too, and that needs the velocity field rather than this. */
  const bool write_volume_motion = (kernel_data.film.denoising_pass_options_flag &
                                    DENOISING_PASS_VOLUME_MOTION) != 0;

  /* With `..._ALWAYS` the medium contributes its vector whether or not this sample scattered, so a
   * pixel that also sees a surface gets a blend of the two rather than one or the other at random.
   * The surface still writes its own vector when the path reaches it, and the weights sum in the
   * same pass - `pass_motion_weight` divides them out. */
  const bool always_write_volume_motion = (kernel_data.film.denoising_pass_options_flag &
                                           DENOISING_PASS_VOLUME_MOTION_ALWAYS) != 0;

  /* Only describe the pixel when nothing follows this segment. `isect.prim` is PRIM_NONE exactly
   * when the segment runs off into the background - the same test `integrator_shade_volume_setup`
   * uses to set `ray->tmax`. A lamp counts as a surface here: it will shade and write its own
   * vector. The state is still the one the segment was set up with; the volume kernel only
   * overwrites it later, in the phase scatter. */
  const bool background_only = (kernel_data.film.denoising_pass_options_flag &
                                DENOISING_PASS_VOLUME_MOTION_BACKGROUND_ONLY) != 0;
  const bool own_medium_only = (kernel_data.film.denoising_pass_options_flag &
                                DENOISING_PASS_VOLUME_MOTION_OWN_MEDIUM_ONLY) != 0;

  /* A scattered sample describes the pixel with whatever ends the segment - the caller has already
   * picked that point and passed it as `reference_P`. Nothing is withheld, so the weight is never
   * zero and reconstruction is never told the pixel stood still. */
  const bool segment_end = (kernel_data.film.denoising_pass_options_flag &
                            DENOISING_PASS_VOLUME_MOTION_SEGMENT_END) != 0;

  bool segment_describes_pixel = true;
  if (!segment_end && (background_only || own_medium_only)) {
    const bool ends_in_background = INTEGRATOR_STATE(state, isect, prim) == PRIM_NONE;
    if (own_medium_only) {
      /* The medium's own hull does not end the pixel - more medium or the background follows, and
       * the entry point stays the right thing to describe it with. A different object does end
       * it, and it will write its own vector once the path reaches it. */
      const bool ends_on_own_hull = INTEGRATOR_STATE(state, isect, object) ==
                                    INTEGRATOR_STATE_ARRAY(state, volume_stack, 0, object);
      segment_describes_pixel = ends_in_background || ends_on_own_hull;
    }
    else {
      segment_describes_pixel = ends_in_background;
    }
  }

  if ((scatter || always_write_volume_motion) && write_volume_motion && segment_describes_pixel &&
      INTEGRATOR_STATE(state, path, bounce) == 0 && kernel_data.film.pass_motion != PASS_UNUSED)
  {
    /* Weight by how much of the pixel the medium accounts for. A scattered sample has nothing else
     * to offer, so it keeps full weight; a transmitted one contributes only its share, and the
     * surface behind it supplies the rest with a weight of one. A thin haze in front of an object
     * therefore barely moves the vector, while dense fog still owns it. */
    const float motion_weight = (scatter || !always_write_volume_motion) ? 1.0f : coverage;
    if (motion_weight > 0.0f) {
      const float4 motion = camera_motion_vector(kg, reference_P, reference_P, reference_P);
      film_write_pass_float4(buffer + kernel_data.film.pass_motion, motion * motion_weight);
      film_write_pass_float(buffer + kernel_data.film.pass_motion_weight, motion_weight);
    }
  }

  /* Pay back the depth the scattered sample never travelled, so the guide describes the point the
   * vector above was built from. Weighted like `denoising_depth_compute` does, so this delta and
   * the one the bounding surface already wrote are on the same scale, and added rather than
   * overwritten - an overwrite would erase the entry point's depth this is a remainder of. */
  const bool medium_owns_depth = (kernel_data.film.denoising_pass_options_flag &
                                  DENOISING_PASS_VOLUME_DEPTH_REPRESENTATIVE) != 0;
  if ((scatter || medium_owns_depth) && depth_delta != 0.0f &&
      INTEGRATOR_STATE(state, path, bounce) == 0 &&
      kernel_data.film.pass_denoising_depth != PASS_UNUSED)
  {
    film_write_pass_float(buffer + kernel_data.film.pass_denoising_depth,
                          ensure_finite(depth_delta * average(denoising_feature_throughput)));
  }

  if (scatter && kernel_data.film.pass_denoising_normal != PASS_UNUSED) {
    /* Assume scatter is sufficiently diffuse to stop writing denoising features. */
    INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_DENOISING_FEATURES;

    /* The caller picked this: the surface that ends the segment where there is one, the view
     * direction where the pixel is nothing but medium. */
    film_write_pass_float3(buffer + kernel_data.film.pass_denoising_normal, reference_N);

    /* Say the pixel is fully rough. Nothing writes roughness for a scattering event, so it stays
     * at zero, and zero is how a mirror is described - temporal reconstruction treats such pixels
     * as needing specular reprojection and filters them weakly, which is the opposite of what a
     * participating medium wants. A phase function is far closer to a diffuse lobe. */
    if (kernel_data.film.pass_denoising_roughness != PASS_UNUSED) {
      film_write_pass_float(buffer + kernel_data.film.pass_denoising_roughness,
                            average(denoising_feature_throughput));
    }
  }

  if (kernel_data.film.pass_denoising_albedo != PASS_UNUSED) {
    /* Write albedo. */
    const Spectrum denoising_albedo = ensure_finite(denoising_feature_throughput * albedo);
    film_write_pass_spectrum(buffer + kernel_data.film.pass_denoising_albedo, denoising_albedo);
  }
}

ccl_device_forceinline void film_write_denoising_features_background(
    KernelGlobals kg, IntegratorState state, ccl_global float *ccl_restrict render_buffer)
{
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  if (!(path_flag & PATH_RAY_DENOISING_FEATURES)) {
    return;
  }

  /* Do not write default background denoising data for secondary paths. */
  if (INTEGRATOR_STATE(state, path, bounce) != 0) {
    return;
  }

  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);

  if (kernel_data.film.pass_denoising_depth != PASS_UNUSED &&
      !denoising_depth_owned_by_medium(kg, path_flag))
  {
    /* A pixel of fog with sky behind it gets this on a transmitted sample and the fog's own depth
     * on a scattered one. `FLT_MAX` makes that swing infinite, so temporal reconstruction reads it
     * as geometry appearing and vanishing rather than as one distant surface. The volume pays its
     * remainder up to the same value, so under this mode both branches agree. */
    const bool finite_background = (kernel_data.film.denoising_pass_options_flag &
                                    DENOISING_PASS_VOLUME_DEPTH_SEGMENT_END) != 0;
    film_overwrite_pass_float(buffer + kernel_data.film.pass_denoising_depth,
                              finite_background ? VOLUME_DENOISING_FAR_DEPTH : FLT_MAX);
  }

  /* 'pass_denoising_albedo' is written by 'film_write_emission_or_background_pass' */
}
#endif /* __DENOISING_FEATURES__ */

CCL_NAMESPACE_END
