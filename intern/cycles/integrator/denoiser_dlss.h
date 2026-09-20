/* SPDX-FileCopyrightText: 2025 NVIDIA Corporation
 * SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_DLSS

#  include "integrator/denoiser_gpu.h"

struct NVSDK_NGX_Handle;
struct NVSDK_NGX_CUDADevice;
struct NVSDK_NGX_Parameter;

CCL_NAMESPACE_BEGIN
class DLSSParameterSpy;
CCL_NAMESPACE_END

CCL_NAMESPACE_BEGIN

/* Implementation of denoising API which uses DLSS. */
class DLSSDenoiser : public DenoiserGPU {
 public:
  DLSSDenoiser(Device *denoiser_device, const DenoiseParams &params);
  ~DLSSDenoiser();

  static bool is_device_supported(const DeviceInfo &device);

  void reset_history() override;
  void set_zero_motion(bool zero_motion) override;
  void set_camera_transforms(const float world_to_view[16],
                             const float view_to_clip[16]) override;

  /* Take a set of images allocated elsewhere, which this denoiser then fills instead of allocating
   * its own. Handing over a set with no handles goes back to the ordinary path. */
  void set_external_images(const DenoiserExternalImages &images) override;
  bool external_images_active() const override;
  void set_external_evaluate(ExternalEvaluate evaluate) override;

 private:
  bool denoise_create_if_needed(DenoiseContext &context) override;

  bool denoise_configure_if_needed(DenoiseContext &context) override;

  bool denoise_filter_color_preprocess(const DenoiseContext &context,
                                       const DenoisePass &pass) override;
  bool denoise_filter_color_postprocess(const DenoiseContext &context,
                                        const DenoisePass &pass) override;

  bool denoise_filter_guiding_preprocess(DenoiseContext &context) override;

  bool denoise_run(const DenoiseContext &context, const DenoisePass &pass) override;

  /* Allocate the evaluation parameter block and fill in everything which stays constant for the
   * lifetime of the feature. Must be called while the feature is alive and the textures are
   * initialized. */
  bool create_eval_params();

  /* Must be called before the feature is released, so that a stale parameter block can never be
   * handed to `EvaluateFeature`. */
  void destroy_eval_params();

  NVSDK_NGX_Handle *handle_ = nullptr;
  NVSDK_NGX_CUDADevice *ngx_device_ = nullptr;

  /* Reused across evaluations: allocating a fresh block and re-setting ~20 string-keyed entries
   * per 1-spp viewport iteration is pure overhead. Only the entries that actually change per
   * evaluation are written in `denoise_run()`.
   *
   * NOTE: unset keys read back as 0/null, so with a reused block every key that matters must be
   * written either here or per evaluation - stale values would otherwise persist. */
  NVSDK_NGX_Parameter *eval_params_ = nullptr;

  /* A recording stand-in for the block above, made only when `CYCLES_DLSS_SPY_PARAMS` asks for it.
   * It forwards everything and keeps a list of what the library read out, which is the one way to
   * see what a model wanted and did not get. */
  DLSSParameterSpy *eval_params_spy_ = nullptr;

  struct CUDATexture {
    void init(Device *device, int width, int height, int num_components);

    /* Stand on an image that Vulkan allocated, rather than allocating one here.
     *
     * `handle` is what the system calls that memory - a HANDLE on Windows, a file descriptor
     * elsewhere - and ownership of it passes to this texture, which closes it when done. The
     * kernels that fill the texture see no difference: they are given the same surface object
     * either way. */
    bool init_external(Device *device,
                       uint64_t handle,
                       uint64_t memory_size,
                       uint64_t memory_offset,
                       int width,
                       int height,
                       int num_components);
    void destroy();

    void *array = nullptr;
    uint64_t texture_handle = 0;
    uint64_t surface_handle = 0;

    /* Set only for a texture standing on Vulkan memory, and released with it. */
    void *external_memory = nullptr;
    void *external_mipmap = nullptr;
    uint64_t external_handle = 0;
  };

  /* Write one value into every texel of a texture, for an input the renderer has no data for. */
  static void fill_texture_constant(
      CUDATexture &texture, int width, int height, int num_components, float value);
  CUDATexture tex_color_;
  CUDATexture tex_depth_;
  CUDATexture tex_diffuse_albedo_;
  CUDATexture tex_specular_albedo_;
  CUDATexture tex_normal_roughness_;
  CUDATexture tex_motion_;
  CUDATexture tex_specular_motion_;
  /* An input the model of DLSS 4.5 asks about but this renderer has no notion of - how quickly a
   * pixel is allowed to change. Filled with a constant, and only when CYCLES_DLSS_RESPONSIVITY
   * says to, while it is being established whether the new model needs it at all. */
  /* Images the Vulkan side allocated for the model to read, when the model runs there.
   *
   * Set from outside once per set; `external_pending_` says the handles are new and the textures
   * standing on them have to be remade. While these are in play the feature is never created here
   * and nothing is written back into the render buffer - the denoised frame is a Vulkan texture,
   * and it goes to the screen from there. */
  DenoiserExternalImages external_images_;
  ExternalEvaluate external_evaluate_;
  bool external_pending_ = false;
  bool external_ready_ = false;
  int external_width_ = 0;
  int external_height_ = 0;

  bool external_create_if_needed(const DenoiseContext &context);

  CUDATexture tex_responsivity_;
  /* How far the ray that lit a pixel travelled. Cycles knows this by nature but does not write
   * it out; these hold a constant while it is being established whether the model of DLSS 4.5
   * wants the input at all. Set by CYCLES_DLSS_HIT_DISTANCE. */
  CUDATexture tex_specular_hit_distance_;
  CUDATexture tex_diffuse_hit_distance_;
  CUDATexture tex_output_;

  int last_width_ = 0;
  int last_height_ = 0;
  float last_upscale_factor_ = 0.0f;
  bool reset_history_ = true;
  bool zero_motion_ = false;

  /* NGX takes these as pointers and reads them when the feature is evaluated, so they live here
   * rather than on the caller's stack. Row-major 4x4, as the SDK's own helpers pass them. */
  float world_to_view_[16] = {};
  float view_to_clip_[16] = {};
  bool have_camera_transforms_ = false;
};

/* Which model Ray Reconstruction actually ran with, in NGX's own words - for example
 * "Using DRS Overridden Preset Preset_F". Empty until a feature has been created.
 *
 * The application asks for a preset but does not get the last word: a driver profile can override
 * the choice, and nothing in the picture says whether that happened. These report what NGX did. */
string dlss_applied_preset();
void dlss_note_expected_quality(const char *label);

/* Called for every line NGX logs; picks out the one naming the model. */
void dlss_note_applied_preset(const char *message);

CCL_NAMESPACE_END

#endif
