/* SPDX-FileCopyrightText: 2025 NVIDIA Corporation
 * SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_DLSS

#  include "integrator/denoiser_dlss.h"
#  include "integrator/pass_accessor_gpu.h"

#  include "device/cuda/device_impl.h"

#  include "util/path.h"
#  include "util/thread.h"
#  include "util/vector.h"

#  include <cstdlib>
#  include <cstring>
#  include <ios>

#  include <nvsdk_ngx.h>
#  include <nvsdk_ngx_defs_dlssd.h>

#  ifdef _WIN32
#    include "util/windows.h"

#    define dynamic_library_open(path) LoadLibraryW(path)
#    define dynamic_library_close(lib) FreeLibrary(static_cast<HMODULE>(lib))
#    define dynamic_library_find(lib, symbol) \
      reinterpret_cast<t##symbol>(GetProcAddress(static_cast<HMODULE>(lib), #symbol))
#  else
#    include <dlfcn.h>

#    define dynamic_library_open(path) dlopen(path, RTLD_NOW)
#    define dynamic_library_close(lib) dlclose(lib)
#    define dynamic_library_find(lib, symbol) reinterpret_cast<t##symbol>(dlsym(lib, #symbol))
#  endif

struct NGXDriver {
  using tNVSDK_NGX_CUDA_Init_Ext1 = NVSDK_NGX_Result (*)(unsigned long long,
                                                         const wchar_t *,
                                                         NVSDK_NGX_CUDADevice *,
                                                         NVSDK_NGX_Version,
                                                         const NVSDK_NGX_FeatureCommonInfo *);
  using tNVSDK_NGX_CUDA_Shutdown1 = NVSDK_NGX_Result (*)(NVSDK_NGX_CUDADevice *, unsigned int &);
  using tNVSDK_NGX_CUDA_GetFeatureRequirements = decltype(&NVSDK_NGX_CUDA_GetFeatureRequirements);
  using tNVSDK_NGX_CUDA_CreateFeature1 = decltype(&NVSDK_NGX_CUDA_CreateFeature1);
  using tNVSDK_NGX_CUDA_EvaluateFeature = decltype(&NVSDK_NGX_CUDA_EvaluateFeature);
  using tNVSDK_NGX_CUDA_ReleaseFeature = decltype(&NVSDK_NGX_CUDA_ReleaseFeature);
  using tNVSDK_NGX_CUDA_AllocateParameters = decltype(&NVSDK_NGX_CUDA_AllocateParameters);
  using tNVSDK_NGX_CUDA_DestroyParameters = decltype(&NVSDK_NGX_CUDA_DestroyParameters);

  tNVSDK_NGX_CUDA_Init_Ext1 Init_Ext1 = nullptr;
  tNVSDK_NGX_CUDA_Shutdown1 Shutdown1 = nullptr;
  tNVSDK_NGX_CUDA_GetFeatureRequirements GetFeatureRequirements = nullptr;
  tNVSDK_NGX_CUDA_CreateFeature1 CreateFeature1 = nullptr;
  tNVSDK_NGX_CUDA_EvaluateFeature EvaluateFeature = nullptr;
  tNVSDK_NGX_CUDA_ReleaseFeature ReleaseFeature = nullptr;
  tNVSDK_NGX_CUDA_AllocateParameters AllocateParameters = nullptr;
  tNVSDK_NGX_CUDA_DestroyParameters DestroyParameters = nullptr;

  explicit operator bool() const
  {
    return Init_Ext1 != nullptr && Shutdown1 != nullptr && CreateFeature1 != nullptr &&
           EvaluateFeature != nullptr && ReleaseFeature != nullptr &&
           AllocateParameters != nullptr && DestroyParameters != nullptr;
  }

  bool init()
  {
    if (*this) {
      return true;
    }

#  ifdef _WIN32
    WCHAR ngx_path[MAX_PATH] = L"";
    {
      HKEY ngx_key = nullptr;
      LSTATUS result = RegOpenKeyExW(
          HKEY_LOCAL_MACHINE,
          L"System\\CurrentControlSet\\Services\\nvlddmkm\\Parameters\\NGXCore",
          0,
          KEY_READ,
          &ngx_key);
      if (result != ERROR_SUCCESS) {
        result = RegOpenKeyExW(HKEY_LOCAL_MACHINE,
                               L"System\\CurrentControlSet\\Services\\nvlddmkm\\NGXCore",
                               0,
                               KEY_READ,
                               &ngx_key);
      }
      if (result == ERROR_SUCCESS) {
        DWORD ngx_path_size = ARRAYSIZE(ngx_path);
        result = RegQueryValueExW(
            ngx_key, L"NGXPath", 0, nullptr, reinterpret_cast<LPBYTE>(ngx_path), &ngx_path_size);
        RegCloseKey(ngx_key);
      }
      if (result != ERROR_SUCCESS) {
        return false;
      }

      wcscat_s(ngx_path, L"\\_nvngx.dll");
    }
#  else
    const char *const ngx_path = "libnvidia-ngx.so.1";
#  endif

    void *const ngx_module = dynamic_library_open(ngx_path);
    if (ngx_module == nullptr) {
      return false;
    }

    Init_Ext1 = dynamic_library_find(ngx_module, NVSDK_NGX_CUDA_Init_Ext1);
    Shutdown1 = dynamic_library_find(ngx_module, NVSDK_NGX_CUDA_Shutdown1);
    GetFeatureRequirements = dynamic_library_find(ngx_module,
                                                  NVSDK_NGX_CUDA_GetFeatureRequirements);
    CreateFeature1 = dynamic_library_find(ngx_module, NVSDK_NGX_CUDA_CreateFeature1);
    EvaluateFeature = dynamic_library_find(ngx_module, NVSDK_NGX_CUDA_EvaluateFeature);
    ReleaseFeature = dynamic_library_find(ngx_module, NVSDK_NGX_CUDA_ReleaseFeature);
    AllocateParameters = dynamic_library_find(ngx_module, NVSDK_NGX_CUDA_AllocateParameters);
    DestroyParameters = dynamic_library_find(ngx_module, NVSDK_NGX_CUDA_DestroyParameters);

    if (*this) {
      return true;
    }
    else {
      dynamic_library_close(ngx_module);
      return false;
    }
  }
} NVSDK_NGX_CUDA;

CCL_NAMESPACE_BEGIN

static const int ApplicationId = 100334311;

/* The model the library actually used, and whether that is the one that was asked for.
 *
 * The choice is not the application's to make in the end: a driver profile can override it, and on
 * this machine one does - the profile named "Blender" asks for preset F, which the SDK itself
 * marks "Do not use". A run with that override applied denoises nothing, and there is no way to
 * tell from the picture whether the request went through. NGX does say which model it took, but
 * only in its log; this catches that line as it passes through the logging callback, so the answer
 * can be shown in the interface instead of dug out of a file afterwards. */
static thread_mutex dlss_preset_mutex;
static string dlss_applied_preset_text;

/* Which of the six lines is ours.
 *
 * The library reports the model it settled on once per quality mode, all six in a row, and
 * only one of them describes the render about to happen. Keeping the last was keeping
 * UltraQuality - a mode nothing here selects. */
static string dlss_expected_quality_label;

void dlss_note_expected_quality(const char *label)
{
  const thread_scoped_lock lock(dlss_preset_mutex);
  dlss_expected_quality_label = (label != nullptr) ? label : "";
  dlss_applied_preset_text.clear();
}

void dlss_note_applied_preset(const char *message)
{
  if (message == nullptr) {
    return;
  }

  /* The line reads, in full: "Info: (Quality) Using DRS Overridden Preset Preset_F". The part
   * worth keeping starts at "Using", which carries both the model and where the choice came from.
   */
  const char *using_at = strstr(message, "Using ");
  if (using_at == nullptr || strstr(message, "Preset_") == nullptr) {
    return;
  }

  string text = using_at;
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
    text.pop_back();
  }

  /* The quality mode the line is about, written just in front of the message as "(Quality)".
   *
   * Searched backwards from there rather than forwards from the start: the library prefixes its
   * lines with the call site, and that prefix ends in "operator ()" - whose empty brackets would
   * otherwise be read as the label, leaving nothing to match. */
  string label;
  for (const char *close = using_at; close-- != message;) {
    if (*close != ')') {
      continue;
    }
    for (const char *open = close; open-- != message;) {
      if (*open == '(') {
        label = string(open + 1, close - open - 1);
        break;
      }
    }
    if (!label.empty()) {
      break;
    }
  }

  const thread_scoped_lock lock(dlss_preset_mutex);
  if (!dlss_expected_quality_label.empty() && label != dlss_expected_quality_label) {
    return;
  }
  dlss_applied_preset_text = text;
}

string dlss_applied_preset()
{
  const thread_scoped_lock lock(dlss_preset_mutex);
  return dlss_applied_preset_text;
}

static std::wstring to_wide(const string &path)
{
#  ifdef _WIN32
  return string_to_wstring(path);
#  else
  std::wstring wide(path.size(), L' ');
  wide.resize(std::mbstowcs(wide.data(), path.c_str(), path.size()));
  return wide;
#  endif
}

/* The version the add-on installed last, as a directory name, or empty.
 *
 * A library is installed into a directory named after its version rather than written over the one
 * already there, because NGX loads it the moment anything asks whether the feature is supported -
 * which the preferences do on every redraw - and Windows does not allow a loaded library to be
 * written over. The add-on names the directory to use in this file; the rule for choosing it lives
 * there alone, so that this and the Vulkan side cannot disagree about it. */
static string dlss_active_version(const string &root)
{
  const string pointer_path = path_join(root, "dlssd.active");
  if (!path_exists(pointer_path)) {
    return "";
  }

  string version;
  if (!path_read_text(pointer_path, version)) {
    return "";
  }

  /* Only a version, and only one line of it - this file is written by the add-on, but it sits in a
   * directory the user can reach. */
  const size_t end = version.find_first_not_of("0123456789.");
  if (end != string::npos) {
    version.resize(end);
  }

  return version;
}

/* Where to look for `nvngx_dlssd.dll` besides the directory of the executable.
 *
 * This build does not ship that library. NVIDIA's licence does not permit redistributing it beside
 * a GPL application, so the add-on offers to copy the one the driver already installed, into the
 * user's own data directory - a Blender unpacked into Program Files is not writable, and requiring
 * administrator rights to enable a denoiser is a poor trade.
 *
 * What was here before was `path_get()`, the Cycles add-on script directory, which never holds an
 * NGX library and never could. That entry was dead, and DLSS worked only through the executable
 * directory NGX checks on its own.
 *
 * Handed to the capability check as well as to the initialisation, and that is not optional: the
 * check runs first, and if it does not find a library it reports the feature unsupported, Cycles
 * falls back to OptiX, and initialisation never happens. Passing the path to only one of the two
 * looks exactly like the library not being installed at all. */
static vector<std::wstring> dlss_library_search_paths()
{
  const string root = path_join(path_user_get(), "datafiles/dlss");

  vector<std::wstring> paths;

  const string version = dlss_active_version(root);
  if (!version.empty()) {
    paths.push_back(to_wide(path_join(root, version)));
  }

  /* The root as well, always: a library installed by an older version of the add-on sits directly
   * in it, and that installation has to go on working without the user doing anything. */
  paths.push_back(to_wide(root));

  return paths;
}

/* Write one value into every texel, for an input this renderer has no data for.
 *
 * Done from the host once, when the texture is made, rather than through the kernel that fills the
 * guides every frame: the value never changes, and adding it to that kernel would mean threading a
 * new surface through every back-end for something that exists to answer a question. */
void DLSSDenoiser::fill_texture_constant(
    CUDATexture &texture, int width, int height, int num_components, float value)
{
  if (texture.array == nullptr) {
    return;
  }

  vector<float> host(size_t(width) * height * num_components, value);

  CUDA_MEMCPY2D copy = {};
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = host.data();
  copy.srcPitch = size_t(width) * num_components * sizeof(float);
  copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copy.dstArray = static_cast<CUarray>(texture.array);
  copy.WidthInBytes = copy.srcPitch;
  copy.Height = height;

  cuMemcpy2D(&copy);
}

void DLSSDenoiser::CUDATexture::init(Device *device, int width, int height, int num_components)
{
  /* `CYCLES_DLSS_ARRAY_GATHER`: create the arrays so that a four-texel gather can read them, and
   * address them in texels rather than in normalised coordinates.
   *
   * A transformer of the second generation is expected to gather, and a gather needs the array to
   * have been made for it - CUDA refuses otherwise, in a way NGX reports as a platform error and
   * does not log. The older models do not gather, which is what would make this fail for preset F
   * alone. */
  static const bool array_gather = []() {
    const char *value = getenv("CYCLES_DLSS_ARRAY_GATHER");
    return (value != nullptr) && atoi(value) != 0;
  }();

  if (array_gather) {
    CUDA_ARRAY3D_DESCRIPTOR desc3d = {};
    desc3d.Width = width;
    desc3d.Height = height;
    desc3d.Depth = 0;
    desc3d.Format = CU_AD_FORMAT_FLOAT;
    desc3d.NumChannels = num_components;
    desc3d.Flags = CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_TEXTURE_GATHER;

    cuda_device_assert(device, cuArray3DCreate((CUarray *)&array, &desc3d));
  }
  else {
    CUDA_ARRAY_DESCRIPTOR desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = num_components;

    cuda_device_assert(device, cuArrayCreate((CUarray *)&array, &desc));
  }

  CUDA_TEXTURE_DESC tex_desc = {};
  tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  tex_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  tex_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  tex_desc.flags = array_gather ? 0 : CU_TRSF_NORMALIZED_COORDINATES;

  CUDA_RESOURCE_DESC res_desc = {};
  res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
  res_desc.res.array.hArray = (CUarray)array;

  cuda_device_assert(
      device, cuTexObjectCreate((CUtexObject *)&texture_handle, &res_desc, &tex_desc, nullptr));
  cuda_device_assert(device, cuSurfObjectCreate((CUsurfObject *)&surface_handle, &res_desc));
}
/* The flag that says an imported allocation holds one resource and nothing else.
 *
 * Declared here because the loader Cycles uses declares the structure and the entry points but not
 * this constant. The value is CUDA's own and has not changed since external memory was introduced.
 */
#  ifndef CUDA_EXTERNAL_MEMORY_DEDICATED
#    define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1
#  endif

bool DLSSDenoiser::CUDATexture::init_external(Device *device,
                                             const uint64_t handle,
                                             const uint64_t memory_size,
                                             const uint64_t memory_offset,
                                             const int width,
                                             const int height,
                                             const int num_components)
{
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle_desc = {};
#  ifdef _WIN32
  /* Windows keeps its own reference, so the handle stays ours to close. */
  handle_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
  handle_desc.handle.win32.handle = reinterpret_cast<void *>(handle);
#  else
  /* The import takes the descriptor over, and closes it in turn. */
  handle_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  handle_desc.handle.fd = int(handle);
#  endif
  handle_desc.size = memory_size;

  /* Whether to tell CUDA the allocation holds this image and nothing else.
   *
   * It is a claim about how Vulkan allocated the memory, not a request, and CUDA turns the import
   * down when the claim does not hold. The pool the images come from asks for dedicated
   * allocations, so it should - `CYCLES_DLSS_DEDICATED=0` is here to find out whether it actually
   * does. */
  static const bool dedicated = []() {
    const char *value = getenv("CYCLES_DLSS_DEDICATED");
    return (value == nullptr) || atoi(value) != 0;
  }();
  handle_desc.flags = dedicated ? CUDA_EXTERNAL_MEMORY_DEDICATED : 0;

  CUexternalMemory memory = nullptr;
  CUresult result = cuImportExternalMemory(&memory, &handle_desc);
  if (result != CUDA_SUCCESS) {
    LOG_ERROR << "Could not import the Vulkan image into CUDA: " << cuewErrorString(result);
    return false;
  }
  external_memory = memory;
  external_handle = handle;

  CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC array_desc = {};
  array_desc.offset = memory_offset;
  array_desc.numLevels = 1;
  array_desc.arrayDesc.Width = width;
  array_desc.arrayDesc.Height = height;
  array_desc.arrayDesc.Depth = 0;
  array_desc.arrayDesc.Format = CU_AD_FORMAT_FLOAT;
  array_desc.arrayDesc.NumChannels = num_components;
  array_desc.arrayDesc.Flags = CUDA_ARRAY3D_SURFACE_LDST;

  CUmipmappedArray mipmap = nullptr;
  result = cuExternalMemoryGetMappedMipmappedArray(&mipmap, memory, &array_desc);
  if (result != CUDA_SUCCESS) {
    LOG_ERROR << "Could not map the Vulkan image in CUDA: " << cuewErrorString(result) << " ("
              << width << "x" << height << ", " << num_components << " channels, "
              << memory_size << " bytes at offset " << memory_offset << ", dedicated "
              << (handle_desc.flags != 0) << ")";
    destroy();
    return false;
  }
  external_mipmap = mipmap;

  CUarray level = nullptr;
  result = cuMipmappedArrayGetLevel(&level, mipmap, 0);
  if (result != CUDA_SUCCESS) {
    LOG_ERROR << "Could not reach the Vulkan image in CUDA: " << cuewErrorString(result);
    destroy();
    return false;
  }
  array = level;

  CUDA_RESOURCE_DESC res_desc = {};
  res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
  res_desc.res.array.hArray = level;

  CUDA_TEXTURE_DESC tex_desc = {};
  tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  tex_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  tex_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;

  cuda_device_assert(
      device, cuTexObjectCreate((CUtexObject *)&texture_handle, &res_desc, &tex_desc, nullptr));
  cuda_device_assert(device, cuSurfObjectCreate((CUsurfObject *)&surface_handle, &res_desc));
  return true;
}

void DLSSDenoiser::CUDATexture::destroy()
{
  cuSurfObjectDestroy((CUsurfObject)surface_handle);
  surface_handle = 0;
  cuTexObjectDestroy((CUtexObject)texture_handle);
  texture_handle = 0;

  if (external_memory != nullptr) {
    /* The array belongs to the mapping, not to us: destroying the mipmap releases it, and calling
     * cuArrayDestroy on it as well would be a double free. */
    if (external_mipmap != nullptr) {
      cuMipmappedArrayDestroy((CUmipmappedArray)external_mipmap);
      external_mipmap = nullptr;
    }
    cuDestroyExternalMemory((CUexternalMemory)external_memory);
    external_memory = nullptr;
    array = nullptr;

#  ifdef _WIN32
    /* The import made its own reference; this one was ours to close. */
    if (external_handle != 0) {
      CloseHandle(HANDLE(external_handle));
    }
#  endif
    external_handle = 0;
    return;
  }

  cuArrayDestroy((CUarray)array);
  array = 0;
}

DLSSDenoiser::DLSSDenoiser(Device *denoiser_device, const DenoiseParams &params)
    : DenoiserGPU(denoiser_device, params)
{
  CUDADevice *const cuda_device = static_cast<CUDADevice *>(denoiser_device_);
  const CUDAContextScope scope(cuda_device);

  if (!NVSDK_NGX_CUDA.init()) {
    set_error("Failed to load NGX driver");
    return;
  }

  const vector<std::wstring> search_paths = dlss_library_search_paths();

#  ifdef _WIN32
  const wstring user_path = string_to_wstring(path_user_get());
#  else
  string user_path_narrow = path_user_get();
  std::wstring user_path(user_path_narrow.size(), L' ');
  user_path.resize(
      std::mbstowcs(user_path.data(), user_path_narrow.c_str(), user_path_narrow.size()));
#  endif

  vector<const wchar_t *> app_paths;
  for (const std::wstring &path : search_paths) {
    app_paths.push_back(path.c_str());
  }

  NVSDK_NGX_FeatureCommonInfo feature_info = {};
  feature_info.PathListInfo.Path = app_paths.data();
  feature_info.PathListInfo.Length = uint32_t(app_paths.size());
  feature_info.LoggingInfo.LoggingCallback =
      [](const char *message, NVSDK_NGX_Logging_Level loggingLevel, NVSDK_NGX_Feature) {
        dlss_note_applied_preset(message);

        switch (loggingLevel) {
          case NVSDK_NGX_LOGGING_LEVEL_OFF:
          case NVSDK_NGX_LOGGING_LEVEL_NUM:
            assert(false);
            break;
          case NVSDK_NGX_LOGGING_LEVEL_ON:
            LOG_INFO << message;
            break;
          case NVSDK_NGX_LOGGING_LEVEL_VERBOSE:
            LOG_INFO << message;
            break;
        }
      };
  feature_info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;

  ngx_device_ = new NVSDK_NGX_CUDADevice{
      cuda_device->cuContext, static_cast<CUDADeviceQueue *>(denoiser_queue_.get())->stream()};

  /* The API version this build was compiled against is 1.5.0, from SDK 310.7.0 - which predates
   * the model shipped as DLSS 4.5 Ray Reconstruction. If that model expects a newer contract, an
   * old version number is exactly the kind of thing it would answer by doing nothing and saying
   * nothing, which is what it does here. `CYCLES_DLSS_API_VERSION` tries other numbers to find
   * out. */
  static const int api_version_forced = []() {
    const char *value = getenv("CYCLES_DLSS_API_VERSION");
    return (value != nullptr) ? int(strtol(value, nullptr, 0)) : -1;
  }();
  const NVSDK_NGX_Version api_version = (api_version_forced > 0) ?
                                            NVSDK_NGX_Version(api_version_forced) :
                                            NVSDK_NGX_Version_API;

  const NVSDK_NGX_Result result = NVSDK_NGX_CUDA.Init_Ext1(
      ApplicationId, user_path.c_str(), ngx_device_, api_version, &feature_info);

  if (NVSDK_NGX_FAILED(result)) {
    set_error("Failed to initialize NGX driver");
  }
}

DLSSDenoiser::~DLSSDenoiser()
{
  CUDADevice *const cuda_device = static_cast<CUDADevice *>(denoiser_device_);
  const CUDAContextScope scope(cuda_device);

  tex_color_.destroy();
  tex_depth_.destroy();
  tex_diffuse_albedo_.destroy();
  tex_specular_albedo_.destroy();
  tex_normal_roughness_.destroy();
  tex_motion_.destroy();
  tex_specular_motion_.destroy();
  tex_responsivity_.destroy();
  tex_specular_hit_distance_.destroy();
  tex_diffuse_hit_distance_.destroy();
  tex_output_.destroy();

  if (!NVSDK_NGX_CUDA) {
    return;
  }

  destroy_eval_params();

  if (handle_ != nullptr) {
    NVSDK_NGX_CUDA.ReleaseFeature(handle_);
  }

  unsigned int n = 0;
  const NVSDK_NGX_Result result = NVSDK_NGX_CUDA.Shutdown1(ngx_device_, n);

  if (NVSDK_NGX_FAILED(result)) {
    set_error("Failed to shutdown NGX driver");
  }

  delete ngx_device_;
}

void DLSSDenoiser::reset_history()
{
  reset_history_ = true;
}

void DLSSDenoiser::set_zero_motion(const bool zero_motion)
{
  /* The model of DLSS 4.5 denoises a final render but not the viewport, and the one thing that
   * separates the two here is motion: an offline render runs every iteration but the first with
   * motion zeroed, while the viewport hands over real vectors. `CYCLES_DLSS_ZERO_MOTION` zeroes
   * them in the viewport too, to find out whether the vectors are what the new model cannot use.
   */
  static const bool zero_motion_forced = []() {
    const char *value = getenv("CYCLES_DLSS_ZERO_MOTION");
    return (value != nullptr) && atoi(value) != 0;
  }();

  zero_motion_ = zero_motion || zero_motion_forced;
}

void DLSSDenoiser::set_camera_transforms(const float world_to_view[16],
                                         const float view_to_clip[16])
{
  /* `CYCLES_DLSS_CAMERA_MATRICES`: 0 withholds them, 1 sends them, 2 sends deliberate rubbish.
   *
   * The third setting is the only way to find out whether the DLL reads them at all: the CUDA path
   * is documented far less than Streamline, the keys exist in the header either way, and an
   * identical frame from settings 1 and 2 means the entries are being ignored. */
  static const int mode = []() {
    const char *value = getenv("CYCLES_DLSS_CAMERA_MATRICES");
    return (value != nullptr) ? atoi(value) : 1;
  }();

  if (mode == 0) {
    have_camera_transforms_ = false;
    return;
  }

  if (mode == 2) {
    for (int i = 0; i < 16; i++) {
      world_to_view_[i] = (i % 5 == 0) ? 1.0f : 0.0f;
      view_to_clip_[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }
    world_to_view_[14] = 1000.0f;
    have_camera_transforms_ = true;
    return;
  }

  memcpy(world_to_view_, world_to_view, sizeof(world_to_view_));
  memcpy(view_to_clip_, view_to_clip, sizeof(view_to_clip_));
  have_camera_transforms_ = true;
}

bool DLSSDenoiser::is_device_supported(const DeviceInfo &device)
{
  if (device.type != DEVICE_CUDA && device.type != DEVICE_OPTIX) {
    return false;
  }

  /* 'NVSDK_NGX_CUDA_GetFeatureRequirements' is an expensive call, so cache the result (since
   * 'is_device_supported' is called a lot). */
  static NVSDK_NGX_Feature_Support_Result supported_cache[8] = {
      NVSDK_NGX_FeatureSupportResult_CheckNotPresent,
      NVSDK_NGX_FeatureSupportResult_CheckNotPresent,
      NVSDK_NGX_FeatureSupportResult_CheckNotPresent,
      NVSDK_NGX_FeatureSupportResult_CheckNotPresent,
      NVSDK_NGX_FeatureSupportResult_CheckNotPresent,
      NVSDK_NGX_FeatureSupportResult_CheckNotPresent,
      NVSDK_NGX_FeatureSupportResult_CheckNotPresent,
      NVSDK_NGX_FeatureSupportResult_CheckNotPresent};

  if (device.num >= 8) {
    return false;
  }
  if (supported_cache[device.num] != NVSDK_NGX_FeatureSupportResult_CheckNotPresent) {
    return supported_cache[device.num] == NVSDK_NGX_FeatureSupportResult_Supported;
  }

  if (!NVSDK_NGX_CUDA.init() || NVSDK_NGX_CUDA.GetFeatureRequirements == nullptr) {
    /* NVIDIA driver is too old (requires 590+). */
    return false;
  }

#  ifdef _WIN32
  const wstring user_path = string_to_wstring(path_user_get());
#  else
  string user_path_narrow = path_user_get();
  std::wstring user_path(user_path_narrow.size(), L' ');
  user_path.resize(
      std::mbstowcs(user_path.data(), user_path_narrow.c_str(), user_path_narrow.size()));
#  endif

  CUdevice cuDevice = 0;
  cuDeviceGet(&cuDevice, device.num);

  /* The same search paths the initialisation uses. Without them this check only ever looks beside
   * the executable, and on a build that ships no library it answers "unsupported" however
   * correctly the library is installed elsewhere.
   *
   * This call is also what loads the library into the process, long before anything renders: the
   * preferences ask about support on every redraw. That is why an install cannot write over the
   * file in place, and why each version is installed into a directory of its own. */
  const vector<std::wstring> search_path_storage = dlss_library_search_paths();
  vector<const wchar_t *> search_paths;
  for (const std::wstring &path : search_path_storage) {
    search_paths.push_back(path.c_str());
  }

  NVSDK_NGX_FeatureCommonInfo feature_info = {};
  feature_info.PathListInfo.Path = search_paths.data();
  feature_info.PathListInfo.Length = uint32_t(search_paths.size());

  NVSDK_NGX_FeatureDiscoveryInfo discovery_info = {};
  discovery_info.SDKVersion = NVSDK_NGX_Version_API;
  discovery_info.FeatureID = NVSDK_NGX_Feature_RayReconstruction;
  discovery_info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
  discovery_info.Identifier.v.ApplicationId = ApplicationId;
  discovery_info.ApplicationDataPath = user_path.c_str();
  discovery_info.FeatureInfo = &feature_info;

  NVSDK_NGX_FeatureRequirement requirement = {NVSDK_NGX_FeatureSupportResult_Supported};

  const NVSDK_NGX_Result result = NVSDK_NGX_CUDA.GetFeatureRequirements(
      cuDevice, &discovery_info, &requirement);

  if (NVSDK_NGX_SUCCEED(result)) {
    supported_cache[device.num] = requirement.FeatureSupported;
    return requirement.FeatureSupported == NVSDK_NGX_FeatureSupportResult_Supported;
  }
  else {
    return false;
  }
}

void DLSSDenoiser::set_external_images(const DenoiserExternalImages &images)
{
  external_images_ = images;
  external_pending_ = images.is_set();
  if (!images.is_set()) {
    external_ready_ = false;
  }
}

void DLSSDenoiser::set_external_evaluate(ExternalEvaluate evaluate)
{
  external_evaluate_ = std::move(evaluate);
}

bool DLSSDenoiser::external_images_active() const
{
  return external_images_.is_set() && external_ready_;
}

/* Stand the input textures on the images the Vulkan side allocated.
 *
 * No feature is created: the model runs over there, and creating one here would only take a slot
 * of a library that has a limit on them. */
bool DLSSDenoiser::external_create_if_needed(const DenoiseContext &context)
{
  const int width = context.buffer_params.width;
  const int height = context.buffer_params.height;

  if (external_ready_ && !external_pending_ && external_width_ == width &&
      external_height_ == height)
  {
    return true;
  }

  CUDADevice *const cuda_device = static_cast<CUDADevice *>(denoiser_device_);
  const CUDAContextScope scope(cuda_device);

  denoiser_queue_->synchronize();

  tex_color_.destroy();
  tex_depth_.destroy();
  tex_diffuse_albedo_.destroy();
  tex_specular_albedo_.destroy();
  tex_normal_roughness_.destroy();
  tex_motion_.destroy();
  tex_specular_motion_.destroy();

  external_ready_ = false;
  external_pending_ = false;
  external_width_ = width;
  external_height_ = height;

  const struct {
    CUDATexture *texture;
    const DenoiserExternalImage *image;
  } bindings[] = {
      {&tex_color_, &external_images_.color},
      {&tex_depth_, &external_images_.depth},
      {&tex_diffuse_albedo_, &external_images_.diffuse_albedo},
      {&tex_specular_albedo_, &external_images_.specular_albedo},
      {&tex_normal_roughness_, &external_images_.normal_roughness},
      {&tex_motion_, &external_images_.motion},
      {&tex_specular_motion_, &external_images_.specular_motion},
  };

  tex_output_.destroy();
  const DenoiserExternalImage &output_image = external_images_.output;
  if (output_image.handle == 0 ||
      !tex_output_.init_external(cuda_device,
                                 output_image.handle,
                                 output_image.memory_size,
                                 output_image.memory_offset,
                                 output_image.width,
                                 output_image.height,
                                 output_image.channels))
  {
    LOG_ERROR << "Could not map the shared denoiser output; denoising on the render device "
                 "instead.";
    external_images_ = DenoiserExternalImages();
    return false;
  }

  for (const auto &binding : bindings) {
    const DenoiserExternalImage &image = *binding.image;
    if (image.width != width || image.height != height) {
      set_error("The shared denoiser images do not match the render resolution");
      return false;
    }
    if (!binding.texture->init_external(cuda_device,
                                        image.handle,
                                        image.memory_size,
                                        image.memory_offset,
                                        image.width,
                                        image.height,
                                        image.channels))
    {
      /* Back to denoising here. Everything mapped so far goes, and the images are forgotten, so
       * that the next frame allocates textures of this denoiser's own rather than running on
       * half a set. */
      LOG_ERROR << "Could not map the shared denoiser images; denoising on the render device "
                   "instead.";
      tex_color_.destroy();
      tex_depth_.destroy();
      tex_diffuse_albedo_.destroy();
      tex_specular_albedo_.destroy();
      tex_normal_roughness_.destroy();
      tex_motion_.destroy();
      tex_specular_motion_.destroy();
      external_images_ = DenoiserExternalImages();
      return false;
    }
  }

  external_ready_ = true;
  reset_history_ = true;
  return !cuda_device->have_error();
}

bool DLSSDenoiser::denoise_create_if_needed(DenoiseContext &context)
{
  if (external_images_.is_set()) {
    return external_create_if_needed(context);
  }

  const bool recreate_denoiser = last_width_ != context.denoised_buffer_params.width ||
                                 last_height_ != context.denoised_buffer_params.height ||
                                 last_upscale_factor_ != context.denoise_params.upscale_factor;
  if (handle_ != nullptr && !recreate_denoiser) {
    return true;
  }

  reset_history_ = true;

  CUDADevice *const cuda_device = static_cast<CUDADevice *>(denoiser_device_);
  const CUDAContextScope scope(cuda_device);

  if (handle_ != nullptr) {
    denoiser_queue_->synchronize();

    /* Destroy the evaluation block first: it holds pointers into the textures destroyed below, and
     * must never outlive the feature it was built for. */
    destroy_eval_params();

    NVSDK_NGX_CUDA.ReleaseFeature(handle_);
    handle_ = nullptr;
  }

  tex_color_.destroy();
  tex_depth_.destroy();
  tex_diffuse_albedo_.destroy();
  tex_specular_albedo_.destroy();
  tex_normal_roughness_.destroy();
  tex_motion_.destroy();
  tex_specular_motion_.destroy();
  tex_responsivity_.destroy();
  tex_specular_hit_distance_.destroy();
  tex_diffuse_hit_distance_.destroy();
  tex_output_.destroy();

  if (context.buffer_params.width <= 128 || context.buffer_params.height <= 96) {
    last_width_ = 0;
    last_height_ = 0;
    return false;
  }

  NVSDK_NGX_Parameter *params = nullptr;
  if (NVSDK_NGX_FAILED(NVSDK_NGX_CUDA.AllocateParameters(&params))) {
    return false;
  }

  /* DLSS SDK 310.x requires the CUDA context and stream both on initialization
   * and in the feature-create parameter block. Older SDKs accepted the device
   * wrapper alone, but the current Ray Reconstruction module dereferences these
   * entries while creating the feature. */
  params->Set(NVSDK_NGX_Parameter_Input1, ngx_device_->cudaContext);
  params->Set(NVSDK_NGX_Parameter_Input2, ngx_device_->cudaStream);

  params->Set(NVSDK_NGX_Parameter_Width, uint(context.buffer_params.width));
  params->Set(NVSDK_NGX_Parameter_Height, uint(context.buffer_params.height));
  params->Set(NVSDK_NGX_Parameter_OutWidth, uint(context.denoised_buffer_params.width));
  params->Set(NVSDK_NGX_Parameter_OutHeight, uint(context.denoised_buffer_params.height));

  params->Set(NVSDK_NGX_Parameter_DLSS_Denoise_Mode, NVSDK_NGX_DLSS_Denoise_Mode_DLUnified);

  /* The model of DLSS 4.5 Ray Reconstruction - preset F - returns its input untouched here, and
   * its log says nothing about why. The public SDK predates that model, so what it expects is not
   * documented anywhere; these switches exist to find it by experiment rather than by guessing,
   * and come out once the answer is known. */
  static const int create_flags_forced = []() {
    const char *value = getenv("CYCLES_DLSS_CREATE_FLAGS");
    return (value != nullptr) ? int(strtol(value, nullptr, 0)) : -1;
  }();
  params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
              (create_flags_forced >= 0) ?
                  create_flags_forced :
                  (NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes));
  params->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
  const NVSDK_NGX_PerfQuality_Value quality =
      context.denoise_params.upscale_factor == 1.0f ?
          NVSDK_NGX_PerfQuality_Value_DLAA :
      context.denoise_params.upscale_factor <= 1.0f / 0.65f ?
          NVSDK_NGX_PerfQuality_Value_MaxQuality :
      context.denoise_params.upscale_factor <= 1.0f / 0.57f ?
          NVSDK_NGX_PerfQuality_Value_Balanced :
      context.denoise_params.upscale_factor <= 1.0f / 0.5f ? NVSDK_NGX_PerfQuality_Value_MaxPerf :
                                                             NVSDK_NGX_PerfQuality_Value_UltraPerformance;
  params->Set(NVSDK_NGX_Parameter_PerfQualityValue, quality);

  /* The names the library writes in front of its own lines, so that the one belonging to this
   * render can be told apart from the other five. */
  dlss_note_expected_quality(quality == NVSDK_NGX_PerfQuality_Value_DLAA       ? "DLAA" :
                             quality == NVSDK_NGX_PerfQuality_Value_MaxQuality ? "Quality" :
                             quality == NVSDK_NGX_PerfQuality_Value_Balanced   ? "Balanced" :
                             quality == NVSDK_NGX_PerfQuality_Value_MaxPerf    ? "Perf" :
                                                                                "UltraPerf");
  static const int depth_type_forced = []() {
    const char *value = getenv("CYCLES_DLSS_HW_DEPTH");
    return (value != nullptr) ? atoi(value) : -1;
  }();
  params->Set(
      NVSDK_NGX_Parameter_Use_HW_Depth,
      uint((depth_type_forced >= 0) ? depth_type_forced : int(NVSDK_NGX_DLSS_Depth_Type_Linear)));

  static const int roughness_mode_forced = []() {
    const char *value = getenv("CYCLES_DLSS_ROUGHNESS_MODE");
    return (value != nullptr) ? atoi(value) : -1;
  }();
  params->Set(NVSDK_NGX_Parameter_DLSS_Roughness_Mode,
              uint((roughness_mode_forced >= 0) ? roughness_mode_forced :
                                                  int(NVSDK_NGX_DLSS_Roughness_Mode_Packed)));

  /* These are Ray Reconstruction presets, not the similarly named Super
   * Resolution presets. SDK 310.6 marks RR K through O as "Do not use" and
   * identifies E as the latest supported transformer model, D as the default one.
   *
   * Which of the two suits a path tracer is a measurement: on a settled frame E discards the top
   * of the distribution over emissive geometry - bright pixels drop from 32% of the region to
   * 0.3%, and its mean brightness to a third of the raw path, where OptiX stays within four
   * percent. Neither the input range nor the upscale explains it, so the model itself is the
   * remaining variable. The scene chooses; CYCLES_DLSS_PRESET still forces one for measurements
   * driven from a script.
   *
   * Asking is not deciding. A driver profile can override the choice - on this machine the profile
   * named "Blender" asks for F, which the SDK marks "Do not use" - and NGX says which model it
   * settled on only in its log. `dlss_note_applied_preset` catches that line as it goes past. */
  NVSDK_NGX_RayReconstruction_Hint_Render_Preset preset =
      NVSDK_NGX_RayReconstruction_Hint_Render_Preset(params_.dlss_preset);

  static const int preset_forced = []() {
    const char *value = getenv("CYCLES_DLSS_PRESET");
    if (value == nullptr) {
      return -1;
    }
    const char letter = (*value >= 'a') ? char(*value - 'a' + 'A') : *value;
    if (letter < 'A' || letter > 'O') {
      return -1;
    }
    return int(letter - 'A') + 1;
  }();
  if (preset_forced >= 0) {
    preset = NVSDK_NGX_RayReconstruction_Hint_Render_Preset(preset_forced);
  }

  params->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA, preset);
  params->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, preset);
  params->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, preset);
  params->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, preset);
  params->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance, preset);
  params->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality, preset);

  const NVSDK_NGX_Result result = NVSDK_NGX_CUDA.CreateFeature1(
      ngx_device_, NVSDK_NGX_Feature_RayReconstruction, params, &handle_);

  NVSDK_NGX_CUDA.DestroyParameters(params);

  if (NVSDK_NGX_FAILED(result)) {
    set_error("Failed to create DLSS instance");
    return false;
  }

  /* Said once per feature, because the model is decided here and nothing downstream can tell what
   * it was. A driver override reads as a picture that is simply wrong - too soft, or not denoised
   * at all - and without this line there is nothing to connect that to its cause. */
  const string applied = dlss_applied_preset();
  if (!applied.empty()) {
    LOG_WARNING << "DLSS Ray Reconstruction model: " << applied;
  }

  tex_color_.init(cuda_device, context.buffer_params.width, context.buffer_params.height, 4);
  tex_depth_.init(cuda_device, context.buffer_params.width, context.buffer_params.height, 1);
  tex_diffuse_albedo_.init(
      cuda_device, context.buffer_params.width, context.buffer_params.height, 4);
  tex_specular_albedo_.init(
      cuda_device, context.buffer_params.width, context.buffer_params.height, 4);
  tex_normal_roughness_.init(
      cuda_device, context.buffer_params.width, context.buffer_params.height, 4);
  tex_motion_.init(cuda_device, context.buffer_params.width, context.buffer_params.height, 2);
  tex_specular_motion_.init(
      cuda_device, context.buffer_params.width, context.buffer_params.height, 2);

  /* `CYCLES_DLSS_RESPONSIVITY` gives a value between 0 and 1, or a negative number to leave the
   * input unset as before. Constant across the frame: Cycles has nothing that corresponds to a
   * per-pixel responsivity, and the point here is only to learn whether the model wants the input
   * at all. */
  static const float responsivity = []() {
    const char *value = getenv("CYCLES_DLSS_RESPONSIVITY");
    return (value != nullptr) ? float(atof(value)) : -1.0f;
  }();
  if (responsivity >= 0.0f) {
    tex_responsivity_.init(
        cuda_device, context.buffer_params.width, context.buffer_params.height, 1);
    fill_texture_constant(tex_responsivity_,
                          context.buffer_params.width,
                          context.buffer_params.height,
                          1,
                          responsivity);
  }

  /* `CYCLES_DLSS_HIT_DISTANCE` gives a distance in scene units, or a negative number to leave the
   * inputs unset. Constant for now - the question this answers is whether the model wants them at
   * all, and only then is it worth writing the pass that carries the real values. */
  static const float hit_distance = []() {
    const char *value = getenv("CYCLES_DLSS_HIT_DISTANCE");
    return (value != nullptr) ? float(atof(value)) : -1.0f;
  }();
  if (hit_distance >= 0.0f) {
    for (CUDATexture *texture : {&tex_specular_hit_distance_, &tex_diffuse_hit_distance_}) {
      texture->init(cuda_device, context.buffer_params.width, context.buffer_params.height, 1);
      fill_texture_constant(
          *texture, context.buffer_params.width, context.buffer_params.height, 1, hit_distance);
    }
  }

  tex_output_.init(
      cuda_device, context.denoised_buffer_params.width, context.denoised_buffer_params.height, 4);

  if (!create_eval_params()) {
    set_error("Failed to allocate DLSS evaluation parameters");
    return false;
  }

  last_width_ = context.denoised_buffer_params.width;
  last_height_ = context.denoised_buffer_params.height;
  last_upscale_factor_ = context.denoise_params.upscale_factor;

  return !cuda_device->have_error();
}

/* A parameter block that says out loud what the library asked it for.
 *
 * Preset F fails inside the library, reports one code, and writes nothing to its own log. What it
 * reads out of the parameter block is the only signal it emits about what it expects, and it reads
 * by name through this interface, so standing in front of the real block records exactly that: the
 * order of the requests, and which of them came back empty. That is the difference between knowing
 * which input is missing and guessing at inputs one at a time.
 *
 * The forwarding is complete and the recording is bounded, but it is still off unless
 * `CYCLES_DLSS_SPY_PARAMS` is set: this exists to answer a question, not to live in the hot path. */
class DLSSParameterSpy : public NVSDK_NGX_Parameter {
 public:
  explicit DLSSParameterSpy(NVSDK_NGX_Parameter *inner) : inner_(inner) {}

  void Set(const char *name, unsigned long long value) override { inner_->Set(name, value); }
  void Set(const char *name, float value) override { inner_->Set(name, value); }
  void Set(const char *name, double value) override { inner_->Set(name, value); }
  void Set(const char *name, unsigned int value) override { inner_->Set(name, value); }
  void Set(const char *name, int value) override { inner_->Set(name, value); }
  void Set(const char *name, ID3D11Resource *value) override { inner_->Set(name, value); }
  void Set(const char *name, ID3D12Resource *value) override { inner_->Set(name, value); }
  void Set(const char *name, void *value) override { inner_->Set(name, value); }

  NVSDK_NGX_Result Get(const char *name, unsigned long long *out) const override
  {
    const NVSDK_NGX_Result result = inner_->Get(name, out);
    return record(name, result, *out);
  }
  NVSDK_NGX_Result Get(const char *name, float *out) const override
  {
    const NVSDK_NGX_Result result = inner_->Get(name, out);
    return record(name, result, uint64_t(*out != 0.0f));
  }
  NVSDK_NGX_Result Get(const char *name, double *out) const override
  {
    const NVSDK_NGX_Result result = inner_->Get(name, out);
    return record(name, result, uint64_t(*out != 0.0));
  }
  NVSDK_NGX_Result Get(const char *name, unsigned int *out) const override
  {
    const NVSDK_NGX_Result result = inner_->Get(name, out);
    return record(name, result, *out);
  }
  NVSDK_NGX_Result Get(const char *name, int *out) const override
  {
    const NVSDK_NGX_Result result = inner_->Get(name, out);
    return record(name, result, uint64_t(*out));
  }
  NVSDK_NGX_Result Get(const char *name, ID3D11Resource **out) const override
  {
    const NVSDK_NGX_Result result = inner_->Get(name, out);
    return record(name, result, uint64_t(uintptr_t(*out)));
  }
  NVSDK_NGX_Result Get(const char *name, ID3D12Resource **out) const override
  {
    const NVSDK_NGX_Result result = inner_->Get(name, out);
    return record(name, result, uint64_t(uintptr_t(*out)));
  }
  NVSDK_NGX_Result Get(const char *name, void **out) const override
  {
    const NVSDK_NGX_Result result = inner_->Get(name, out);
    return record(name, result, uint64_t(uintptr_t(*out)));
  }

  void Reset() override { inner_->Reset(); }

  /* Print the list once, after the evaluation that produced it. Once, because the list is the same
   * every frame and there are hundreds of frames. */
  void report()
  {
    if (reported_) {
      return;
    }
    reported_ = true;

    LOG_WARNING << "DLSS read " << reads_.size() << " entries out of the parameter block:";
    for (const Read &read : reads_) {
      if (NVSDK_NGX_SUCCEED(read.result)) {
        LOG_WARNING << "  ok       " << read.name << " = " << std::hex << read.value << std::dec;
      }
      else {
        /* A refused read leaves the caller's variable alone, so there is no value to report. */
        LOG_WARNING << "  missing  " << read.name;
      }
    }
  }

 private:
  struct Read {
    string name;
    NVSDK_NGX_Result result;
    uint64_t value;
  };

  NVSDK_NGX_Result record(const char *name, NVSDK_NGX_Result result, uint64_t value) const
  {
    /* One frame's worth. A cap because a runaway would otherwise grow without bound in a viewport
     * that never stops redrawing. */
    if (reads_.size() < 1024) {
      reads_.push_back({name ? name : "(null)", result, value});
    }
    return result;
  }

  NVSDK_NGX_Parameter *inner_;
  mutable vector<Read> reads_;
  bool reported_ = false;
};

bool DLSSDenoiser::create_eval_params()
{
  destroy_eval_params();

  if (NVSDK_NGX_FAILED(NVSDK_NGX_CUDA.AllocateParameters(&eval_params_))) {
    eval_params_ = nullptr;
    return false;
  }

  static const bool spy_params = []() {
    const char *value = getenv("CYCLES_DLSS_SPY_PARAMS");
    return (value != nullptr) && atoi(value) != 0;
  }();
  if (spy_params) {
    eval_params_spy_ = new DLSSParameterSpy(eval_params_);
  }

  /* The CUDA context and stream, the same pair the creation block carries.
   *
   * They were only ever set at creation, which is what NVIDIA's own CUDA sample gets away with -
   * that one reuses a single parameter block for both calls, so the pair is still there at
   * evaluation time. This code allocates a separate block, and left it without them. The older
   * models did not mind; the model of DLSS 4.5 resolves the stream per evaluation, found nothing,
   * and failed with a platform error on every frame while logging nothing at all. */
  eval_params_->Set(NVSDK_NGX_Parameter_Input1, ngx_device_->cudaContext);
  eval_params_->Set(NVSDK_NGX_Parameter_Input2, ngx_device_->cudaStream);

  /* Everything below is constant for the lifetime of the feature. The texture and surface entries
   * store the address of a member, and `CUDATexture::init()` only ever runs from
   * `denoise_create_if_needed()`, so those addresses stay valid until the feature is released. */
  eval_params_->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
  eval_params_->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);

  /* Ray Reconstruction ignores exposure and tonemapper entries (SDK 310.6 integration guide 3.7),
   * but they are kept so that the parameter block is bit-for-bit what a freshly allocated one used
   * to be. */
  eval_params_->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
  eval_params_->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
  eval_params_->Set(NVSDK_NGX_Parameter_TonemapperType, uint(NVSDK_NGX_TONEMAPPER_STRING));

  eval_params_->Set(NVSDK_NGX_Parameter_Color, &tex_color_.texture_handle);
  eval_params_->Set(NVSDK_NGX_Parameter_Depth, &tex_depth_.texture_handle);
  eval_params_->Set(NVSDK_NGX_Parameter_DiffuseAlbedo, &tex_diffuse_albedo_.texture_handle);
  eval_params_->Set(NVSDK_NGX_Parameter_SpecularAlbedo, &tex_specular_albedo_.texture_handle);
  eval_params_->Set(NVSDK_NGX_Parameter_GBuffer_Normals, &tex_normal_roughness_.texture_handle);
  eval_params_->Set(NVSDK_NGX_Parameter_GBuffer_Roughness, &tex_normal_roughness_.texture_handle);
  eval_params_->Set(NVSDK_NGX_Parameter_MotionVectors, &tex_motion_.texture_handle);
  eval_params_->Set(NVSDK_NGX_Parameter_GBuffer_SpecularMvec, &tex_specular_motion_.texture_handle);

  /* Only when it was actually made - an unset key reads back as null, which is what the
   * model saw before and must go on seeing when the mask is not in play. */
  if (tex_responsivity_.array != nullptr) {
    eval_params_->Set(NVSDK_NGX_Parameter_DLSSD_ResponsivityMask,
                      &tex_responsivity_.texture_handle);
  }

  if (tex_specular_hit_distance_.array != nullptr) {
    eval_params_->Set(NVSDK_NGX_Parameter_DLSSD_SpecularHitDistance,
                      &tex_specular_hit_distance_.texture_handle);
    eval_params_->Set(NVSDK_NGX_Parameter_DLSSD_DiffuseHitDistance,
                      &tex_diffuse_hit_distance_.texture_handle);
  }
  eval_params_->Set(NVSDK_NGX_Parameter_Output, &tex_output_.surface_handle);

  eval_params_->Set(NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis, 0);
  eval_params_->Set(NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis, 1);

  return true;
}

void DLSSDenoiser::destroy_eval_params()
{
  delete eval_params_spy_;
  eval_params_spy_ = nullptr;

  if (eval_params_ == nullptr) {
    return;
  }

  NVSDK_NGX_CUDA.DestroyParameters(eval_params_);
  eval_params_ = nullptr;
}

bool DLSSDenoiser::denoise_configure_if_needed(DenoiseContext & /*context*/)
{
  return true;
}

/* How much the noisy colour is scaled down before it reaches the network, and scaled back up
 * afterwards. Emissive geometry in a Cycles scene reaches the hundreds, while reconstruction is
 * trained on a pre-exposed signal around one; measured on a settled frame it loses two thirds of
 * an emitter's brightness and a quarter of the frame, where OptiX stays within four percent. The
 * scaling is linear and exactly undone, so a settled frame survives the round trip - which a tone
 * curve would not. One reproduces the previous behaviour. */
static float dlss_exposure_scale()
{
  static const float scale = []() {
    const char *value = getenv("CYCLES_DLSS_EXPOSURE");
    const float parsed = (value != nullptr) ? (float)atof(value) : 1.0f;
    return (parsed > 0.0f) ? parsed : 1.0f;
  }();

  return scale;
}

bool DLSSDenoiser::denoise_filter_color_preprocess(const DenoiseContext &context,
                                                   const DenoisePass &pass)
{
  if (pass.type != PASS_COMBINED) {
    return false;
  }

  // Input params (with resolution divider applied)
  const BufferParams &buffer_params = context.buffer_params;

  const int work_size = buffer_params.width * buffer_params.height;

  const float exposure_scale = dlss_exposure_scale();

  const DeviceKernelArguments args(&tex_color_.surface_handle,
                                   &context.render_buffers->buffer.device_pointer,
                                   &buffer_params.full_x,
                                   &buffer_params.full_y,
                                   &buffer_params.width,
                                   &buffer_params.height,
                                   &buffer_params.offset,
                                   &buffer_params.stride,
                                   &buffer_params.pass_stride,
                                   &pass.denoised_offset,
                                   &context.pass_sample_count,
                                   &context.num_samples,
                                   &exposure_scale);

  return denoiser_queue_->enqueue(
      DEVICE_KERNEL_FILTER_COLOR_PREPROCESS_TO_SURFACE, work_size, args);
}
bool DLSSDenoiser::denoise_filter_color_postprocess(const DenoiseContext &context,
                                                    const DenoisePass &pass)
{
  if (pass.type != PASS_COMBINED) {
    return false;
  }


  // Output params
  const BufferParams &buffer_params = context.denoised_buffer_params;

  const int work_size = buffer_params.width * buffer_params.height;

  const float exposure_scale = dlss_exposure_scale();

  const DeviceKernelArguments args(&tex_output_.surface_handle,
                                   &tex_color_.texture_handle,
                                   &context.render_buffers->buffer.device_pointer,
                                   &buffer_params.full_x,
                                   &buffer_params.full_y,
                                   &buffer_params.width,
                                   &buffer_params.height,
                                   &buffer_params.offset,
                                   &buffer_params.stride,
                                   &context.buffer_params.full_x,
                                   &context.buffer_params.full_y,
                                   &context.buffer_params.offset,
                                   &context.buffer_params.stride,
                                   &context.buffer_params.width,
                                   &context.buffer_params.height,
                                   &buffer_params.pass_stride,
                                   &context.num_samples,
                                   &pass.denoised_offset,
                                   &context.pass_sample_count,
                                   &pass.num_components,
                                   &params_.upscale_factor,
                                   &exposure_scale);

  return denoiser_queue_->enqueue(
      DEVICE_KERNEL_FILTER_COLOR_POSTPROCESS_FROM_SURFACE, work_size, args);
}

bool DLSSDenoiser::denoise_filter_guiding_preprocess(DenoiseContext &context)
{
  const BufferParams &buffer_params = context.buffer_params;

  const int work_size = buffer_params.width * buffer_params.height;

  const int pass_depth = context.buffer_params.get_pass_offset(PASS_DENOISING_DEPTH);
  const int pass_specular_albedo = context.buffer_params.get_pass_offset(
      PASS_DENOISING_SPECULAR_ALBEDO);
  const int pass_roughness = context.buffer_params.get_pass_offset(PASS_DENOISING_ROUGHNESS);
  const int pass_specular_motion = context.buffer_params.get_pass_offset(
      PASS_DENOISING_SPECULAR_MOTION);
  const int motion_num_samples = zero_motion_ ? -context.num_samples : context.num_samples;

  /* How the albedo hint is brought into range before it reaches NGX - see the kernel for what the
   * bits mean. Emission and background write radiance into that pass rather than reflectance, and
   * which compression suits the network is a measurement rather than a derivation. Read once per
   * process: it selects a configuration for a run, not per-frame behaviour. */
  static const int albedo_mode = []() {
    const char *value = getenv("CYCLES_DLSS_ALBEDO_MODE");
    /* Mode 3 - a square-root compression with a small floor - rather than dividing by the peak
     * channel and clipping. Peak division sends every emitter above one to the same fully
     * saturated hue with its magnitude gone, which is what turned white-hot lava into flat orange.
     * On the rig it is worth another 0.004 of mean absolute Laplacian on top of the depth and
     * emissive fixes, and it costs nothing on ordinary reflectance. */
    return (value != nullptr) ? atoi(value) : 3;
  }();

  /* Replace a guide with a constant to find out whether the network reads it at all. A frame that
   * does not move when the guide carries no information says the work Cycles puts into writing that
   * guide cannot matter. Bit 0 depth, bit 1 normal and roughness, bit 2 diffuse albedo, bit 3
   * specular albedo. Diagnostic; the default sends the real thing. */
  static const int guide_scramble = []() {
    const char *value = getenv("CYCLES_DLSS_GUIDE_SCRAMBLE");
    return (value != nullptr) ? atoi(value) : 0;
  }();

  const DeviceKernelArguments args(&tex_depth_.surface_handle,
                                   &tex_diffuse_albedo_.surface_handle,
                                   &tex_specular_albedo_.surface_handle,
                                   &tex_normal_roughness_.surface_handle,
                                   &tex_motion_.surface_handle,
                                   &tex_specular_motion_.surface_handle,
                                   &context.render_buffers->buffer.device_pointer,
                                   &buffer_params.offset,
                                   &buffer_params.stride,
                                   &buffer_params.pass_stride,
                                   &context.pass_sample_count,
                                   &pass_depth,
                                   &context.pass_denoising_albedo,
                                   &pass_specular_albedo,
                                   &context.pass_denoising_normal,
                                   &pass_roughness,
                                   &context.pass_motion,
                                   &context.pass_motion_weight,
                                   &pass_specular_motion,
                                   &buffer_params.full_x,
                                   &buffer_params.full_y,
                                   &buffer_params.width,
                                   &buffer_params.height,
                                   &motion_num_samples,
                                   &albedo_mode,
                                   &guide_scramble);

  return denoiser_queue_->enqueue(
      DEVICE_KERNEL_FILTER_GUIDING_PREPROCESS_TO_SURFACE, work_size, args);
}

bool DLSSDenoiser::denoise_run(const DenoiseContext &context, const DenoisePass &pass)
{
  if (pass.type != PASS_COMBINED) {
    return false;
  }

  /* The model runs on the Vulkan side, on the very images the kernels above just filled. The
   * synchronize is what makes them written rather than merely queued. */
  if (external_images_.is_set()) {
    if (!denoiser_queue_->synchronize()) {
      return false;
    }

    if (external_evaluate_ && external_evaluate_(reset_history_)) {
      reset_history_ = false;
      return true;
    }

    /* The model did not run. Let go of the shared images so that the next frame is denoised here
     * in the ordinary way - failing the frame instead would cancel the whole render, which is how
     * a final render came to break off half way and write nothing. */
    LOG_ERROR << "The external DLSS evaluation did not run; denoising on the render device from "
                 "the next frame.";
    external_images_ = DenoiserExternalImages();
    external_ready_ = false;
    return false;
  }

  if (eval_params_ == nullptr) {
    return false;
  }

  CUDADevice *const cuda_device = static_cast<CUDADevice *>(denoiser_device_);
  const CUDAContextScope scope(cuda_device);

  /* Only the entries that change between evaluations. Everything else was written once in
   * `create_eval_params()` and stays valid for the lifetime of the feature. */
  eval_params_->Set(NVSDK_NGX_Parameter_Reset, reset_history_ ? 1 : 0);

  /* The jitter an offline render reports comes from the plan's own Halton sequence; the viewport's
   * comes from the scene. That is the last difference left between a render the model of DLSS 4.5
   * denoises and a viewport it does not, so `CYCLES_DLSS_ZERO_JITTER` takes it out of the picture
   * to see whether the value it is being told is the problem. */
  static const bool zero_jitter = []() {
    const char *value = getenv("CYCLES_DLSS_ZERO_JITTER");
    return (value != nullptr) && atoi(value) != 0;
  }();

  eval_params_->Set(NVSDK_NGX_Parameter_Jitter_Offset_X,
                    zero_jitter ? 0.0f : context.pixel_jitter.x);
  eval_params_->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y,
                    zero_jitter ? 0.0f : context.pixel_jitter.y);

  eval_params_->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                    uint(context.buffer_params.width));
  eval_params_->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                    uint(context.buffer_params.height));

  /* Depth and motion are screen-space numbers; without these the network has no way to know what
   * they mean in the world. Both keys are ordinary optional inputs of the CUDA eval params, not
   * part of the block the SDK marks as research-only. The pointers stay valid because the storage
   * is a member of this object. */
  if (have_camera_transforms_) {
    eval_params_->Set(NVSDK_NGX_Parameter_DLSS_WORLD_TO_VIEW_MATRIX,
                      static_cast<void *>(world_to_view_));
    eval_params_->Set(NVSDK_NGX_Parameter_DLSS_VIEW_TO_CLIP_MATRIX,
                      static_cast<void *>(view_to_clip_));
  }

  /* The third argument is a progress callback, not a CUDA stream - the stream comes from
   * `ngx_device_`, which is built from the denoiser queue's stream. That is why the preprocess
   * kernels enqueued on that same queue are correctly ordered against this evaluation without an
   * explicit synchronization here. */
  /* Whether the context is already in an error state before NGX is asked to do anything.
   *
   * A CUDA error raised asynchronously by an earlier Cycles kernel surfaces at the next
   * synchronising call, and would then be reported against whoever made that call - here, as a
   * platform error from the evaluation, every frame, with nothing in NGX's own log. Checked once,
   * because a synchronisation per frame is not free. */
  static bool context_checked = false;
  if (!context_checked) {
    context_checked = true;

    const CUresult sync_result = cuCtxSynchronize();
    if (sync_result != CUDA_SUCCESS) {
      const char *name = nullptr;
      cuGetErrorName(sync_result, &name);
      LOG_ERROR << "The CUDA context was already in an error state before DLSS was evaluated: "
                << (name ? name : "unknown")
                << ". Whatever DLSS reports after this belongs to an "
                   "earlier kernel, not to the denoiser.";
    }
  }

  NVSDK_NGX_Parameter *const eval_block = (eval_params_spy_ != nullptr) ?
                                              static_cast<NVSDK_NGX_Parameter *>(eval_params_spy_) :
                                              eval_params_;
  const NVSDK_NGX_Result result = NVSDK_NGX_CUDA.EvaluateFeature(handle_, eval_block, nullptr);

  if (eval_params_spy_ != nullptr) {
    eval_params_spy_->report();
  }

  /* What CUDA makes of it, once. If the library's refusal comes from a CUDA call of its own, the
   * context is left in an error state and that error has a name - which is more than the one
   * opaque code the library returns. A clean context says the refusal was decided in software,
   * before anything was submitted. */
  if (NVSDK_NGX_FAILED(result)) {
    static bool asked_cuda = false;
    if (!asked_cuda) {
      asked_cuda = true;

      const CUresult sync_result = cuCtxSynchronize();
      const char *name = nullptr;
      cuGetErrorName(sync_result, &name);
      LOG_ERROR << "CUDA after the failed evaluation: " << (name ? name : "unknown");
    }
  }

  if (NVSDK_NGX_SUCCEED(result)) {
    reset_history_ = false;
    return true;
  }

  /* Said once, with the code NGX returned. A failed evaluation reaches the user as an unchanged,
   * noisy frame - which is indistinguishable from a model that ran and did nothing - and the
   * caller only logs that the denoiser failed, not why. Finding out what the model of DLSS 4.5
   * objects to took a day of guessing that this line would have ended. */
  static bool reported = false;
  if (!reported) {
    reported = true;
    LOG_ERROR << "DLSS evaluation failed with NGX result 0x" << std::hex << uint32_t(result)
              << std::dec << " (model: " << dlss_applied_preset() << ")";
  }

  reset_history_ = true;
  return false;
}

CCL_NAMESPACE_END

#endif
