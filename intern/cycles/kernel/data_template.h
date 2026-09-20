/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef KERNEL_STRUCT_BEGIN
#  define KERNEL_STRUCT_BEGIN(name, parent)
#endif
#ifndef KERNEL_STRUCT_END
#  define KERNEL_STRUCT_END(name)
#endif
#ifndef KERNEL_STRUCT_MEMBER
#  define KERNEL_STRUCT_MEMBER(parent, type, name)
#endif
#ifndef KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
#  define KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
#endif

/* Background. */

KERNEL_STRUCT_BEGIN(KernelBackground, background)
/* xyz store direction, w the angle. float4 instead of float3 is used
 * to ensure consistent padding/alignment across devices. */
KERNEL_STRUCT_MEMBER(background, float4, sun)
KERNEL_STRUCT_MEMBER(background, int, use_sun_guiding)
/* Only shader index. */
KERNEL_STRUCT_MEMBER(background, int, surface_shader)
KERNEL_STRUCT_MEMBER(background, int, volume_shader)
KERNEL_STRUCT_MEMBER(background, int, transparent)
KERNEL_STRUCT_MEMBER(background, float, transparent_roughness_squared_threshold)
/* Sun sampling. */
KERNEL_STRUCT_MEMBER(background, float, sun_weight)
/* Importance map sampling. */
KERNEL_STRUCT_MEMBER(background, float, map_weight)
KERNEL_STRUCT_MEMBER(background, float, portal_weight)
KERNEL_STRUCT_MEMBER(background, int, map_res_x)
KERNEL_STRUCT_MEMBER(background, int, map_res_y)
/* Ray differential used for generating the importance map. */
KERNEL_STRUCT_MEMBER(background, float, map_dD)
/* Multiple importance sampling. */
KERNEL_STRUCT_MEMBER(background, int, use_mis)
/* Light-group. */
KERNEL_STRUCT_MEMBER(background, int, lightgroup)
/* Object Index. */
KERNEL_STRUCT_MEMBER(background, int, object_index)
/* Padding. */
KERNEL_STRUCT_MEMBER(background, int, pad1)
KERNEL_STRUCT_END(KernelBackground)

/* BVH: own BVH2 if no native device acceleration struct used. */

KERNEL_STRUCT_BEGIN(KernelBVH, bvh)
KERNEL_STRUCT_MEMBER(bvh, int, root)
KERNEL_STRUCT_MEMBER(bvh, int, have_motion)
KERNEL_STRUCT_MEMBER(bvh, int, have_curves)
KERNEL_STRUCT_MEMBER(bvh, int, have_points)
KERNEL_STRUCT_MEMBER(bvh, int, have_volumes)
KERNEL_STRUCT_MEMBER(bvh, int, bvh_layout)
KERNEL_STRUCT_MEMBER(bvh, int, use_bvh_steps)
KERNEL_STRUCT_MEMBER(bvh, int, curve_subdivisions)
KERNEL_STRUCT_END(KernelBVH)

/* Film. */

KERNEL_STRUCT_BEGIN(KernelFilm, film)
/* XYZ to rendering color space transform. float4 instead of float3 to
 * ensure consistent padding/alignment across devices. */
KERNEL_STRUCT_MEMBER(film, float4, xyz_to_r)
KERNEL_STRUCT_MEMBER(film, float4, xyz_to_g)
KERNEL_STRUCT_MEMBER(film, float4, xyz_to_b)
KERNEL_STRUCT_MEMBER(film, float4, rgb_to_y)
KERNEL_STRUCT_MEMBER(film, float4, white_xyz)
/* Rec709 to rendering color space. */
KERNEL_STRUCT_MEMBER(film, float4, rec709_to_r)
KERNEL_STRUCT_MEMBER(film, float4, rec709_to_g)
KERNEL_STRUCT_MEMBER(film, float4, rec709_to_b)
KERNEL_STRUCT_MEMBER(film, int, is_rec709)
/* Exposure. */
KERNEL_STRUCT_MEMBER(film, float, exposure)
/* Passed used. */
KERNEL_STRUCT_MEMBER(film, int, pass_flag)
KERNEL_STRUCT_MEMBER(film, int, denoising_pass_flag)
KERNEL_STRUCT_MEMBER(film, int, light_pass_flag)
/* Pass offsets. */
KERNEL_STRUCT_MEMBER(film, int, pass_stride)
KERNEL_STRUCT_MEMBER(film, int, pass_combined)
KERNEL_STRUCT_MEMBER(film, int, pass_depth)
KERNEL_STRUCT_MEMBER(film, int, pass_position)
KERNEL_STRUCT_MEMBER(film, int, pass_normal)
KERNEL_STRUCT_MEMBER(film, int, pass_roughness)
KERNEL_STRUCT_MEMBER(film, int, pass_motion)
KERNEL_STRUCT_MEMBER(film, int, pass_motion_weight)
KERNEL_STRUCT_MEMBER(film, int, pass_uv)
KERNEL_STRUCT_MEMBER(film, int, pass_object_id)
KERNEL_STRUCT_MEMBER(film, int, pass_material_id)
KERNEL_STRUCT_MEMBER(film, int, pass_diffuse_color)
KERNEL_STRUCT_MEMBER(film, int, pass_glossy_color)
KERNEL_STRUCT_MEMBER(film, int, pass_transmission_color)
KERNEL_STRUCT_MEMBER(film, int, pass_diffuse_indirect)
KERNEL_STRUCT_MEMBER(film, int, pass_glossy_indirect)
KERNEL_STRUCT_MEMBER(film, int, pass_transmission_indirect)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_indirect)
KERNEL_STRUCT_MEMBER(film, int, pass_diffuse_direct)
KERNEL_STRUCT_MEMBER(film, int, pass_glossy_direct)
KERNEL_STRUCT_MEMBER(film, int, pass_transmission_direct)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_direct)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_scatter)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_scatter_denoised)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_transmit)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_transmit_denoised)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_majorant)
KERNEL_STRUCT_MEMBER(film, int, pass_volume_majorant_sample_count)
KERNEL_STRUCT_MEMBER(film, int, pass_emission)
KERNEL_STRUCT_MEMBER(film, int, pass_background)
KERNEL_STRUCT_MEMBER(film, int, pass_ao)
KERNEL_STRUCT_MEMBER(film, float, pass_alpha_threshold)
KERNEL_STRUCT_MEMBER(film, int, pass_shadow_catcher)
KERNEL_STRUCT_MEMBER(film, int, pass_shadow_catcher_sample_count)
KERNEL_STRUCT_MEMBER(film, int, pass_shadow_catcher_matte)
KERNEL_STRUCT_MEMBER(film, int, pass_render_time)
/* Cryptomatte. */
KERNEL_STRUCT_MEMBER(film, int, cryptomatte_passes)
KERNEL_STRUCT_MEMBER(film, int, cryptomatte_depth)
KERNEL_STRUCT_MEMBER(film, int, pass_cryptomatte)
/* Adaptive sampling. */
KERNEL_STRUCT_MEMBER(film, int, pass_adaptive_aux_buffer)
KERNEL_STRUCT_MEMBER(film, int, pass_sample_count)
/* Mist. */
KERNEL_STRUCT_MEMBER(film, int, pass_mist)
KERNEL_STRUCT_MEMBER(film, float, mist_start)
KERNEL_STRUCT_MEMBER(film, float, mist_inv_depth)
KERNEL_STRUCT_MEMBER(film, float, mist_falloff)
/* Denoising. */
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_albedo)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_specular_albedo)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_normal)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_roughness)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_depth)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_backward_motion)
KERNEL_STRUCT_MEMBER(film, int, pass_denoising_specular_motion)
KERNEL_STRUCT_MEMBER(film, int, denoising_pass_options_flag)
/* AOVs. */
KERNEL_STRUCT_MEMBER(film, int, pass_aov_color)
KERNEL_STRUCT_MEMBER(film, int, pass_aov_value)
/* Light groups. */
KERNEL_STRUCT_MEMBER(film, int, pass_lightgroup)
/* Baking. */
KERNEL_STRUCT_MEMBER(film, int, pass_bake_primitive)
KERNEL_STRUCT_MEMBER(film, int, pass_bake_seed)
KERNEL_STRUCT_MEMBER(film, int, pass_bake_differential)
/* Shadow catcher. */
KERNEL_STRUCT_MEMBER(film, int, use_approximate_shadow_catcher)
/* Path Guiding */
KERNEL_STRUCT_MEMBER(film, int, pass_guiding_color)
KERNEL_STRUCT_MEMBER(film, int, pass_guiding_probability)
KERNEL_STRUCT_MEMBER(film, int, pass_guiding_avg_roughness)
/* Padding. */
KERNEL_STRUCT_MEMBER(film, int, pad1)
KERNEL_STRUCT_END(KernelFilm)

/* Integrator. */

KERNEL_STRUCT_BEGIN(KernelIntegrator, integrator)
/* Emission. */
KERNEL_STRUCT_MEMBER(integrator, int, use_direct_light)
KERNEL_STRUCT_MEMBER(integrator, int, use_light_mis)
KERNEL_STRUCT_MEMBER(integrator, int, use_light_tree)
KERNEL_STRUCT_MEMBER(integrator, int, num_lights)
KERNEL_STRUCT_MEMBER(integrator, int, num_distant_lights)
KERNEL_STRUCT_MEMBER(integrator, int, num_background_lights)
/* Portal sampling. */
KERNEL_STRUCT_MEMBER(integrator, int, num_portals)
KERNEL_STRUCT_MEMBER(integrator, int, portal_offset)
/* Flat light distribution. */
KERNEL_STRUCT_MEMBER(integrator, int, num_distribution)
KERNEL_STRUCT_MEMBER(integrator, float, distribution_pdf_triangles)
KERNEL_STRUCT_MEMBER(integrator, float, distribution_pdf_lights)
KERNEL_STRUCT_MEMBER(integrator, float, light_inv_rr_threshold)
/* Bounces. */
KERNEL_STRUCT_MEMBER(integrator, int, min_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_diffuse_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_glossy_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_transmission_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, max_volume_bounce)
/* AO bounces. */
KERNEL_STRUCT_MEMBER(integrator, int, ao_bounces)
KERNEL_STRUCT_MEMBER(integrator, float, ao_bounces_distance)
KERNEL_STRUCT_MEMBER(integrator, float, ao_bounces_factor)
KERNEL_STRUCT_MEMBER(integrator, float, ao_additive_factor)
/* Transparency. */
KERNEL_STRUCT_MEMBER(integrator, int, transparent_min_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, transparent_max_bounce)
KERNEL_STRUCT_MEMBER(integrator, int, transparent_shadows)
/* Caustics. */
KERNEL_STRUCT_MEMBER(integrator, int, caustics_reflective)
KERNEL_STRUCT_MEMBER(integrator, int, caustics_refractive)
KERNEL_STRUCT_MEMBER(integrator, float, filter_glossy)
/* Seed. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, seed)
/* Clamp. */
KERNEL_STRUCT_MEMBER(integrator, float, sample_clamp_direct)
KERNEL_STRUCT_MEMBER(integrator, float, sample_clamp_indirect)
/* Caustics. */
KERNEL_STRUCT_MEMBER(integrator, int, use_caustics)
/* Sampling pattern. */
KERNEL_STRUCT_MEMBER(integrator, int, sampling_pattern)
KERNEL_STRUCT_MEMBER(integrator, float, scrambling_distance)
/* Sobol pattern. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, tabulated_sobol_sequence_size)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, sobol_index_mask)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, blue_noise_sequence_length)
/* Volume render. */
KERNEL_STRUCT_MEMBER(integrator, int, use_volumes)
KERNEL_STRUCT_MEMBER(integrator, int, volume_ray_marching)
KERNEL_STRUCT_MEMBER(integrator, int, volume_max_steps)
/* Shadow catcher. */
KERNEL_STRUCT_MEMBER(integrator, int, has_shadow_catcher)
/* Closure filter. */
KERNEL_STRUCT_MEMBER(integrator, int, filter_closures)
/* MIS debugging. */
KERNEL_STRUCT_MEMBER(integrator, int, direct_light_sampling_type)
/* Path Guiding */
KERNEL_STRUCT_MEMBER(integrator, float, surface_guiding_probability)
KERNEL_STRUCT_MEMBER(integrator, float, volume_guiding_probability)
KERNEL_STRUCT_MEMBER(integrator, int, guiding_distribution_type)
KERNEL_STRUCT_MEMBER(integrator, int, guiding_directional_sampling_type)
KERNEL_STRUCT_MEMBER(integrator, float, guiding_roughness_threshold)
KERNEL_STRUCT_MEMBER(integrator, int, use_guiding)
KERNEL_STRUCT_MEMBER(integrator, int, train_guiding)
KERNEL_STRUCT_MEMBER(integrator, int, use_surface_guiding)
KERNEL_STRUCT_MEMBER(integrator, int, use_volume_guiding)
KERNEL_STRUCT_MEMBER(integrator, int, use_guiding_direct_light)
KERNEL_STRUCT_MEMBER(integrator, int, use_guiding_mis_weights)

/* Volume scattering probability guiding (VSPG), driven by environment variables so a whole sweep
 * costs one build. See Integrator::device_update for the variables and their defaults.
 *
 * These exist because under DLSS the guiding never has data: the render buffer is cleared before
 * every iteration and the guiding filter runs after the trace, so the guide passes read zero and
 * the probability collapses to a constant. Whether that constant is better or worse than the
 * analytic attenuation is a measurement, not an argument - hence the switches.
 *
 * DONT_SPECIALIZE: these change between runs, and specializing on them would rebuild the kernels
 * on every flip. The block must stay a multiple of 16 bytes, hence the pad. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, volume_scatter_guiding)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, float, volume_scatter_probability_override)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, float, volume_scatter_guiding_mix)
/* Which denoising features a surface with no BSDF closures (pure emission) publishes:
 *   bit 0  the geometric normal
 *   bit 1  roughness, as fully rough
 *   bit 2  withhold emission and background from the denoising albedo pass, so that the hint
 *          stays reflectance rather than becoming radiance - see film_write_emission_or_
 *          background_pass for why that matters across a silhouette
 *   bit 3  compress that contribution instead of withholding it, so emitters of different
 *          brightness stop arriving as the same fully saturated hue
 * Zero is upstream behaviour.
 * A knob rather than a constant because the passes accumulate across samples: in a pixel where
 * some paths scatter in the fog and others reach the emitter behind it, the two normals add, and
 * a guide buffer that is itself noisy is worse than one that is merely incomplete. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(integrator, int, denoising_emissive_features)

KERNEL_STRUCT_MEMBER(integrator, float2, pixel_jitter)
KERNEL_STRUCT_END(KernelIntegrator)

/* Volume froxel grid.
 *
 * A camera-aligned grid holding the volume's emission and extinction, integrated front to back so
 * that any segment of a camera ray costs two lookups. It replaces the stochastic volume while the
 * viewport is being interacted with: one sample per pixel through a dense medium is noise, and the
 * grid is the same answer without the variance.
 *
 * Every member changes with the camera or the viewport size, so none of them may be specialized
 * into the kernel. */
KERNEL_STRUCT_BEGIN(KernelVolumeFroxel, froxel)
/* Zero when the grid is not in use - then nothing below is read. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, enabled)
/* How much of the pixel the grid accounts for, 1 while interacting and falling to 0 once the camera
 * settles, so the honest path trace takes over without a step in the picture. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, float, blend)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, res_x)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, res_y)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, res_z)
/* Pixels per froxel along X and Y. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, tile)
/* The depth range the slices span, distributed exponentially between them. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, float, t_near)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, float, t_far)
/* How much of the injection to actually run, for bisecting a fault down to the step that causes
 * it: 0 all of it, 1 write a constant, 2 also generate the camera ray, 3 also walk the hulls. Set
 * from CYCLES_FROXEL_STAGE. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, stage)
/* What the user asked for, as opposed to `enabled`, which is what the frame can actually deliver -
 * the grid needs a viewport and a scene with a medium in it. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, requested)
/* Whether the grid keeps standing in once the view settles, instead of fading back to the traced
 * result over the following frames. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, always)
/* How far the slices reach if the user said so; zero takes it from the volume bounds below. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, float, user_distance)
/* Light samples per cell. Zero leaves the medium unlit - it still blocks and glows, but nothing
 * shines through it - which is what the grid did before and what the lit version is measured
 * against. Set from CYCLES_FROXEL_LIGHT_SAMPLES. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, light_samples)
/* Which corrections to the grid's lighting are switched on, one per bit. Each is a departure from
 * what the grid did originally, and each is measured against the traced result on its own - so
 * they have to be separable at run time rather than at build time. Set from CYCLES_FROXEL_FIX;
 * see `VolumeFroxelFix` in `kernel/types.h`. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, int, fixes)
/* Where between the least and the greatest density inside an octree node the walk towards a light
 * takes its answer. Half way is the obvious choice and not the right one: a node holding both a
 * wisp and the empty space around it averages to more than the ray actually crosses. Set from
 * CYCLES_FROXEL_SIGMA_MIX. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, float, shadow_sigma_mix)
/* How much of the multiple scattering the grid puts back, as a fraction of what the physics says.
 * One means all of it; the knob is there to turn it down, not to fit it. Set from
 * CYCLES_FROXEL_MULTI_SCATTER. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, float, multi_scatter_return)
/* Where the media of the scene are, so the slices can reach exactly that far. `w` is one when the
 * bounds are known and zero when the scene holds no volume at all. Written by the object manager,
 * which is the one place that sees the volume objects with their final bounds. */
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, float4, bounds_min)
KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
KERNEL_STRUCT_MEMBER(froxel, float4, bounds_max)
KERNEL_STRUCT_END(KernelVolumeFroxel)

/* Image. */

KERNEL_STRUCT_BEGIN(KernelImage, image)
KERNEL_STRUCT_MEMBER(image, float, mip_bias)

/* Padding. */
KERNEL_STRUCT_MEMBER(image, int, pad1)
KERNEL_STRUCT_MEMBER(image, int, pad2)
KERNEL_STRUCT_MEMBER(image, int, pad3)
KERNEL_STRUCT_END(KernelImage)

/* SVM. For shader specialization. */

KERNEL_STRUCT_BEGIN(KernelSVMUsage, svm_usage)
#define SHADER_NODE_TYPE(type) KERNEL_STRUCT_MEMBER(svm_usage, int, type)
#define SHADER_NODE_TYPE_DERIVATIVE(type) \
  SHADER_NODE_TYPE(type) \
  SHADER_NODE_TYPE(type##_DERIVATIVE)
#include "kernel/svm/node_types_template.h"
KERNEL_STRUCT_END(KernelSVMUsage)

#undef KERNEL_STRUCT_BEGIN
#undef KERNEL_STRUCT_MEMBER
#undef KERNEL_STRUCT_MEMBER_DONT_SPECIALIZE
#undef KERNEL_STRUCT_END
