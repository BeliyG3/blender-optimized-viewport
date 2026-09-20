/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* CUDA kernel entry points */

#ifdef __CUDA_ARCH__

#  include "kernel/device/cuda/compat.h"
#  include "kernel/device/cuda/config.h"
#  include "kernel/device/cuda/globals.h"

#  include "kernel/device/gpu/image.h"
#  include "kernel/device/gpu/kernel.h"

/* --------------------------------------------------------------------
 * Denoising.
 */

ccl_gpu_kernel(GPU_KERNEL_BLOCK_NUM_THREADS, GPU_KERNEL_MAX_REGISTERS)
    ccl_gpu_kernel_signature(filter_color_preprocess_to_surface,
                             cudaSurfaceObject_t color_surface,
                             ccl_global float *render_buffer,
                             const int full_x,
                             const int full_y,
                             const int width,
                             const int height,
                             const int offset,
                             const int stride,
                             const int pass_stride,
                             const int pass_denoised,
                             const int pass_sample_count,
                             const int num_samples,
                             const float exposure_scale)
{
  const int work_index = ccl_gpu_global_id_x();
  const int y = work_index / width;
  const int x = work_index - y * width;

  if (x >= width || y >= height) {
    return;
  }

  const uint64_t render_pixel_index = offset + (x + full_x) + (y + full_y) * stride;
  ccl_global float *buffer = render_buffer + render_pixel_index * pass_stride;
  ccl_global float *denoised_pixel = buffer + pass_denoised;

  /* The buffer holds a sum over the samples traced into it. NGX wants the radiance itself, so
   * divide it back out; the postprocess multiplies it in again. Previously both sides assumed
   * exactly one sample per reconstruction and skipped this. */
  float pixel_scale;
  if (pass_sample_count == PASS_UNUSED) {
    pixel_scale = 1.0f / max(num_samples, 1);
  }
  else {
    const uint sample_count = __float_as_uint(buffer[pass_sample_count]);
    pixel_scale = (sample_count > 0) ? 1.0f / sample_count : 1.0f;
  }

  float4 color_value;
  color_value.x = denoised_pixel[0] * pixel_scale;
  color_value.y = denoised_pixel[1] * pixel_scale;
  color_value.z = denoised_pixel[2] * pixel_scale;
  /* Accumulated the same way, so it needs the same scale. */
  const float alpha = denoised_pixel[3] * pixel_scale;
  color_value.w = alpha;

  if (alpha > 1e-8f) {
    color_value.x /= alpha;
    color_value.y /= alpha;
    color_value.z /= alpha;
  }

  /* Bring the radiance into the range the network was trained on. Emissive geometry in a Cycles
   * scene reaches the hundreds, and reconstruction does not reproduce it: measured against the
   * raw path on a settled frame, it loses two thirds of the lava's brightness and a quarter of the
   * frame, where OptiX stays within four percent. Scaling is linear and exactly undone by the
   * postprocess - unlike a tone curve, which would not be invertible and would change the noise
   * distribution the network sees. A scale of one leaves everything as it was. */
  color_value.x *= exposure_scale;
  color_value.y *= exposure_scale;
  color_value.z *= exposure_scale;

  surf2Dwrite(color_value, color_surface, x * sizeof(float4), y);
}
ccl_gpu_kernel_postfix

ccl_gpu_kernel(GPU_KERNEL_BLOCK_NUM_THREADS, GPU_KERNEL_MAX_REGISTERS)
    ccl_gpu_kernel_signature(filter_guiding_preprocess_to_surface,
                             cudaSurfaceObject_t depth_surface,
                             cudaSurfaceObject_t albedo_surface,
                             cudaSurfaceObject_t specular_albedo_surface,
                             cudaSurfaceObject_t normal_roughness_surface,
                             cudaSurfaceObject_t motion_surface,
                             cudaSurfaceObject_t specular_motion_surface,
                             const ccl_global float *render_buffer,
                             const int render_offset,
                             const int render_stride,
                             const int render_pass_stride,
                             const int render_pass_sample_count,
                             const int render_pass_depth,
                             const int render_pass_albedo,
                             const int render_pass_specular_albedo,
                             const int render_pass_normal,
                             const int render_pass_roughness,
                             const int render_pass_motion,
                             const int render_pass_motion_weight,
                             const int render_pass_specular_motion,
                             const int full_x,
                             const int full_y,
                             const int width,
                             const int height,
                             const int num_samples,
                             const int albedo_mode,
                             const int guide_scramble)
{
  const int work_index = ccl_gpu_global_id_x();
  const int y = work_index / width;
  const int x = work_index - y * width;

  if (x >= width || y >= height) {
    return;
  }

  const uint64_t render_pixel_index = render_offset + (x + full_x) + (y + full_y) * render_stride;
  const ccl_global float *buffer = render_buffer + render_pixel_index * render_pass_stride;

  const bool zero_motion = num_samples < 0;
  const int effective_num_samples = max(abs(num_samples), 1);
  float pixel_scale;
  if (render_pass_sample_count == PASS_UNUSED) {
    pixel_scale = 1.0f / effective_num_samples;
  }
  else {
    pixel_scale = 1.0f / __float_as_uint(buffer[render_pass_sample_count]);
  }

  /* `guide_scramble` (CYCLES_DLSS_GUIDE_SCRAMBLE) replaces a guide with a constant, which tells the
   * whole frame nothing. If the picture does not change, the network was not reading that guide,
   * and no amount of work on what Cycles writes into it can matter. Bit 0 depth, bit 1 normal and
   * roughness, bit 2 diffuse albedo, bit 3 specular albedo. Diagnostic only. */
  const bool scramble_depth = (guide_scramble & 1) != 0;
  const bool scramble_normal = (guide_scramble & 2) != 0;
  const bool scramble_albedo = (guide_scramble & 4) != 0;
  const bool scramble_specular = (guide_scramble & 8) != 0;

  /* Depth pass. */
  if (render_pass_depth != PASS_UNUSED) {
    const ccl_global float *depth_in = buffer + render_pass_depth;

    const float depth_value = scramble_depth ? 1.0f : depth_in[0] * pixel_scale;

    surf2Dwrite(depth_value, depth_surface, x * sizeof(float), y);
  }

  /* Diffuse albedo pass. */
  if (render_pass_albedo != PASS_UNUSED) {
    const ccl_global float *albedo_in = buffer + render_pass_albedo;

    float4 albedo_value;
    albedo_value.x = albedo_in[0] * pixel_scale;
    albedo_value.y = albedo_in[1] * pixel_scale;
    albedo_value.z = albedo_in[2] * pixel_scale;
    albedo_value.w = 1.0;

    /* Bring the albedo into range without moving its hue. Emission and background write radiance
     * into this pass rather than reflectance, so it can exceed one and something has to compress
     * it - but dividing each channel by its own value pulls the bright channel down harder than
     * the dim one, which greys the hint out. NGX asks for linear reflectance. One divisor for all
     * three keeps the ratios.
     *
     * Which divisor is a measurement, hence the mode (CYCLES_DLSS_ALBEDO_MODE):
     *
     *   bit 0  Compress with sqrt(p / (1 + p)) instead of clipping at the peak. Dividing by the
     *          peak maps every value above one onto the same answer, so a lava emitter at 300 and
     *          a dim glow at 2 both arrive as a fully saturated hue - the hint stops telling the
     *          network anything about brightness. This keeps them apart and still lands in [0,1).
     *   bit 1  Floor the hint where it is non-zero. Demodulation divides the colour by the albedo,
     *          and a nearly black surface amplifies its own noise along with the signal.
     */
    const float albedo_peak = fmaxf(fmaxf(albedo_value.x, albedo_value.y), albedo_value.z);
    if ((albedo_mode & 1) != 0) {
      if (albedo_peak > 0.0f) {
        const float compressed = sqrtf(albedo_peak / (1.0f + albedo_peak));
        const float albedo_scale = compressed / albedo_peak;
        albedo_value.x *= albedo_scale;
        albedo_value.y *= albedo_scale;
        albedo_value.z *= albedo_scale;
      }
    }
    else if (albedo_peak > 1.0f) {
      const float albedo_scale = 1.0f / albedo_peak;
      albedo_value.x *= albedo_scale;
      albedo_value.y *= albedo_scale;
      albedo_value.z *= albedo_scale;
    }

    if ((albedo_mode & 2) != 0) {
      const float floor_value = 0.03f;
      const float current_peak = fmaxf(fmaxf(albedo_value.x, albedo_value.y), albedo_value.z);
      if (current_peak > 0.0f && current_peak < floor_value) {
        const float albedo_scale = floor_value / current_peak;
        albedo_value.x *= albedo_scale;
        albedo_value.y *= albedo_scale;
        albedo_value.z *= albedo_scale;
      }
    }
    albedo_value.x = clamp(albedo_value.x, 0.0f, 1.0f);
    albedo_value.y = clamp(albedo_value.y, 0.0f, 1.0f);
    albedo_value.z = clamp(albedo_value.z, 0.0f, 1.0f);

    if (scramble_albedo) {
      albedo_value = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
    }

    surf2Dwrite(albedo_value, albedo_surface, x * sizeof(float4), y);
  }

  /* Specular albedo pass. */
  if (render_pass_specular_albedo != PASS_UNUSED) {
    const ccl_global float *albedo_in = buffer + render_pass_specular_albedo;

    float4 specular_albedo_value;
    specular_albedo_value.x = albedo_in[0] * pixel_scale;
    specular_albedo_value.y = albedo_in[1] * pixel_scale;
    specular_albedo_value.z = albedo_in[2] * pixel_scale;
    specular_albedo_value.w = 1.0;

    if (scramble_specular) {
      specular_albedo_value = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
    }

    surf2Dwrite(specular_albedo_value, specular_albedo_surface, x * sizeof(float4), y);
  }

  /* Normal and roughness pass. */
  if (render_pass_normal != PASS_UNUSED && render_pass_roughness != PASS_UNUSED) {
    const ccl_global float *normal_in = buffer + render_pass_normal;
    const ccl_global float *roughness_in = buffer + render_pass_roughness;

    float4 normal_roughness_value;
    normal_roughness_value.x = normal_in[0] * pixel_scale;
    normal_roughness_value.y = normal_in[1] * pixel_scale;
    normal_roughness_value.z = normal_in[2] * pixel_scale;
    normal_roughness_value.w = clamp(roughness_in[0] * pixel_scale, 0.0f, 1.0f);

    /* NGX asks for a normalized shading normal. What arrives here is a sum over samples weighted
     * by throughput and closure weight, so its length is arbitrary - and in a pixel that mixes a
     * volume with the surface behind it, it is short. */
    const float normal_length = sqrtf(normal_roughness_value.x * normal_roughness_value.x +
                                      normal_roughness_value.y * normal_roughness_value.y +
                                      normal_roughness_value.z * normal_roughness_value.z);
    if (normal_length > 1e-6f) {
      const float normal_scale = 1.0f / normal_length;
      normal_roughness_value.x *= normal_scale;
      normal_roughness_value.y *= normal_scale;
      normal_roughness_value.z *= normal_scale;
    }

    if (scramble_normal) {
      /* Facing the camera, fully rough: what a volume already writes, now claimed everywhere. */
      normal_roughness_value = make_float4(0.0f, 0.0f, -1.0f, 1.0f);
    }

    surf2Dwrite(normal_roughness_value, normal_roughness_surface, x * sizeof(float4), y);
  }

  /* Motion pass. */
  if (render_pass_motion != PASS_UNUSED) {
    const ccl_global float *motion_in = buffer + render_pass_motion;

    /* Divide by the pass's own weight rather than by the sample count, the way
     * `film_get_pass_pixel_motion` does. Only the samples that reached a surface, the background or
     * a volume wrote a vector, so with more than one sample per iteration the two differ and the
     * vector would come out short. */
    float motion_scale = pixel_scale;
    if (render_pass_motion_weight != PASS_UNUSED) {
      const float weight = buffer[render_pass_motion_weight];
      motion_scale = (weight > 0.0f) ? 1.0f / weight : 0.0f;
    }

    float2 motion_value;
    motion_value.x = zero_motion ? 0.0f : motion_in[0] * motion_scale;
    motion_value.y = zero_motion ? 0.0f : motion_in[1] * motion_scale;

    surf2Dwrite(motion_value, motion_surface, x * sizeof(float2), y);
  }

  /* Specular motion pass. */
  if (render_pass_specular_motion != PASS_UNUSED) {
    const ccl_global float *motion_in = buffer + render_pass_specular_motion;

    float2 motion_value;
    motion_value.x = zero_motion ? 0.0f : motion_in[0] * pixel_scale;
    motion_value.y = zero_motion ? 0.0f : motion_in[1] * pixel_scale;

    surf2Dwrite(motion_value, specular_motion_surface, x * sizeof(float2), y);
  }
}
ccl_gpu_kernel_postfix

ccl_gpu_kernel(GPU_KERNEL_BLOCK_NUM_THREADS, GPU_KERNEL_MAX_REGISTERS)
    ccl_gpu_kernel_signature(filter_color_postprocess_from_surface,
                             cudaSurfaceObject_t color_surface,
                             cudaTextureObject_t input_color_texture,
                             ccl_global float *render_buffer,
                             const int full_x,
                             const int full_y,
                             const int width,
                             const int height,
                             const int offset,
                             const int stride,
                             const int render_full_x,
                             const int render_full_y,
                             const int render_offset,
                             const int render_stride,
                             const int render_width,
                             const int render_height,
                             const int pass_stride,
                             const int num_samples,
                             const int pass_denoised,
                             const int pass_sample_count,
                             const int num_components,
                             const float upscale_factor,
                             const float exposure_scale)
{
  const int work_index = ccl_gpu_global_id_x();
  const int y = work_index / width;
  const int x = work_index - y * width;

  if (x >= width || y >= height) {
    return;
  }

  const float input_x = (float(x) + 0.5f) / upscale_factor - 0.5f;
  const float input_y = (float(y) + 0.5f) / upscale_factor - 0.5f;
  const int input_x0 = clamp(int(floorf(input_x)), 0, render_width - 1);
  const int input_y0 = clamp(int(floorf(input_y)), 0, render_height - 1);
  const int input_x1 = min(input_x0 + 1, render_width - 1);
  const int input_y1 = min(input_y0 + 1, render_height - 1);
  const float alpha_tx = clamp(input_x - floorf(input_x), 0.0f, 1.0f);
  const float alpha_ty = clamp(input_y - floorf(input_y), 0.0f, 1.0f);

  const uint64_t render_pixel_index = render_offset + (input_x0 + render_full_x) +
                                      (input_y0 + render_full_y) * render_stride;
  ccl_global float *buffer = render_buffer + render_pixel_index * pass_stride;

  float pixel_scale;
  if (pass_sample_count == PASS_UNUSED) {
    pixel_scale = num_samples;
  }
  else {
    pixel_scale = __float_as_uint(buffer[pass_sample_count]);
  }

  const uint64_t denoised_pixel_index = offset + (x + full_x) + (y + full_y) * stride;
  ccl_global float *denoised_pixel = render_buffer + denoised_pixel_index * pass_stride +
                                     pass_denoised;

  float4 color_value;
  surf2Dread(&color_value, color_surface, x * sizeof(float4), y);

  const float alpha00 = tex2D<float4>(input_color_texture,
                                      (float(input_x0) + 0.5f) / render_width,
                                      (float(input_y0) + 0.5f) / render_height)
                            .w;
  const float alpha10 = tex2D<float4>(input_color_texture,
                                      (float(input_x1) + 0.5f) / render_width,
                                      (float(input_y0) + 0.5f) / render_height)
                            .w;
  const float alpha01 = tex2D<float4>(input_color_texture,
                                      (float(input_x0) + 0.5f) / render_width,
                                      (float(input_y1) + 0.5f) / render_height)
                            .w;
  const float alpha11 = tex2D<float4>(input_color_texture,
                                      (float(input_x1) + 0.5f) / render_width,
                                      (float(input_y1) + 0.5f) / render_height)
                            .w;
  const float alpha0 = alpha00 * (1.0f - alpha_tx) + alpha10 * alpha_tx;
  const float alpha1 = alpha01 * (1.0f - alpha_tx) + alpha11 * alpha_tx;
  const float alpha = alpha0 * (1.0f - alpha_ty) + alpha1 * alpha_ty;

  denoised_pixel[0] = color_value.x * alpha;
  denoised_pixel[1] = color_value.y * alpha;
  denoised_pixel[2] = color_value.z * alpha;

  /* Put back the scale the preprocess divided out. Unconditional: the preprocess always divides,
   * so skipping this when upscaling left the image dark by exactly the sample count. At one sample
   * per reconstruction the scale is one and nothing changes either way. */
  denoised_pixel[0] *= pixel_scale;
  denoised_pixel[1] *= pixel_scale;
  denoised_pixel[2] *= pixel_scale;

  /* Undo the scale the preprocess applied before handing the colour to the network. Exact, so a
   * settled frame is unchanged by the round trip. */
  if (exposure_scale > 0.0f && exposure_scale != 1.0f) {
    const float inverse_exposure = 1.0f / exposure_scale;
    denoised_pixel[0] *= inverse_exposure;
    denoised_pixel[1] *= inverse_exposure;
    denoised_pixel[2] *= inverse_exposure;
  }

  if (num_components == 3) {
    /* Pass without alpha channel. */
  }
  else {
    /* DLSS reconstructs RGB only. Keep the separately scaled source alpha in the denoised pass so
     * the compositor receives correctly premultiplied RGBA. Cycles stores Combined's fourth
     * component as transparency rather than opacity. */
    denoised_pixel[3] = 1.0f - alpha;
  }
}
ccl_gpu_kernel_postfix

#endif
