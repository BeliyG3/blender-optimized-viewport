/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "vk_ngx_runtime.hh"

#ifdef WITH_DLSS_FRAME_GENERATION

#  include <cstdio>
#  include <filesystem>
#  include <optional>
#  include <vector>

#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>

#  include <nvsdk_ngx_defs_dlssg.h>
#  include <nvsdk_ngx_params_dlssg.h>

#  include "BKE_appdir.hh"

#  include "vk_backend.hh"
#  include "vk_device.hh"
#  include "vk_image_view.hh"
#  include "vk_texture.hh"

namespace blender::gpu {

namespace {

HMODULE ngx_core_load()
{
  static HMODULE module = []() -> HMODULE {
    WCHAR ngx_path[MAX_PATH] = L"";
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
    if (result != ERROR_SUCCESS) {
      return nullptr;
    }

    DWORD ngx_path_size = sizeof(ngx_path);
    result = RegQueryValueExW(
        ngx_key, L"NGXPath", nullptr, nullptr, reinterpret_cast<LPBYTE>(ngx_path), &ngx_path_size);
    RegCloseKey(ngx_key);
    if (result != ERROR_SUCCESS || wcscat_s(ngx_path, L"\\_nvngx.dll") != 0) {
      return nullptr;
    }
    return LoadLibraryW(ngx_path);
  }();
  return module;
}

/* The version a pointer file names, or empty.
 *
 * Each installed library sits in a directory named after its own version, because NGX holds one
 * open from the moment support is queried and Windows will not let an open file be written over.
 * The Cycles add-on writes these pointer files and is the only thing that does. */
std::string active_version(const std::filesystem::path &root, const char *pointer_name)
{
  std::string version;
  std::FILE *pointer = std::fopen((root / pointer_name).string().c_str(), "rb");
  if (pointer == nullptr) {
    return version;
  }

  char buffer[64] = {};
  const size_t read = std::fread(buffer, 1, sizeof(buffer) - 1, pointer);
  std::fclose(pointer);
  version.assign(buffer, read);

  /* Only a version, and only one line of it: the file lives somewhere the user can reach. */
  const size_t end = version.find_first_not_of("0123456789.");
  if (end != std::string::npos) {
    version.resize(end);
  }
  return version;
}

}  // namespace

std::string ngx_error(const char *operation, const NVSDK_NGX_Result result)
{
  char buffer[192];
  std::snprintf(
      buffer, sizeof(buffer), "%s failed with NGX result 0x%08x", operation, uint32_t(result));
  return buffer;
}

bool NGXVulkanRuntime::ensure_initialized()
{
  std::scoped_lock lock(mutex_);
  if (initialized_) {
    return error_.empty();
  }
  initialized_ = true;

  HMODULE module = ngx_core_load();
  if (module == nullptr) {
    error_ = "NVIDIA NGX core (_nvngx.dll) was not found";
    return false;
  }

#  define LOAD_NGX(member, name) \
    member = reinterpret_cast<decltype(member)>(GetProcAddress(module, name))
  LOAD_NGX(init, "NVSDK_NGX_VULKAN_Init_Ext2");
  LOAD_NGX(get_capabilities, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
  LOAD_NGX(allocate_parameters, "NVSDK_NGX_VULKAN_AllocateParameters");
  LOAD_NGX(destroy_parameters, "NVSDK_NGX_VULKAN_DestroyParameters");
  LOAD_NGX(create_feature, "NVSDK_NGX_VULKAN_CreateFeature");
  LOAD_NGX(evaluate_feature, "NVSDK_NGX_VULKAN_EvaluateFeature");
  LOAD_NGX(release_feature, "NVSDK_NGX_VULKAN_ReleaseFeature");
  LOAD_NGX(shutdown, "NVSDK_NGX_VULKAN_Shutdown1");
#  undef LOAD_NGX

  if (init == nullptr || get_capabilities == nullptr || allocate_parameters == nullptr ||
      destroy_parameters == nullptr || create_feature == nullptr || evaluate_feature == nullptr ||
      release_feature == nullptr || shutdown == nullptr)
  {
    error_ = "The NVIDIA driver exposes an incomplete Vulkan NGX API";
    return false;
  }

  /* Where to look for the feature libraries besides the directory of the executable. This build
   * does not ship them - NVIDIA's licence does not allow that beside a GPL application - so the
   * Cycles add-on copies the ones the driver installed into the user's data directory. Without
   * this a feature reports itself unsupported however correctly the library is installed.
   *
   * Both features are named here, and the data directory itself comes last, so that a library
   * installed by an older add-on, sitting directly in it, goes on working untouched. */
  const std::optional<std::string> dlss_dir = BKE_appdir_folder_id(BLENDER_USER_DATAFILES, "dlss");
  std::vector<std::wstring> search_path_storage;
  std::vector<const wchar_t *> search_paths;
  NVSDK_NGX_FeatureCommonInfo feature_info = {};
  if (dlss_dir.has_value()) {
    const std::filesystem::path root(*dlss_dir);

    for (const char *pointer_name : {"dlssg.active", "dlssd.active"}) {
      const std::string version = active_version(root, pointer_name);
      if (!version.empty()) {
        search_path_storage.push_back((root / version).wstring());
      }
    }
    search_path_storage.push_back(root.wstring());

    for (const std::wstring &path : search_path_storage) {
      search_paths.push_back(path.c_str());
    }
    feature_info.PathListInfo.Path = search_paths.data();
    feature_info.PathListInfo.Length = static_cast<int>(search_paths.size());
  }

  const VKDevice &device = VKBackend::get().device;
  const NVSDK_NGX_Result result = init(ngx_application_id,
                                       L".",
                                       device.instance_get(),
                                       device.physical_device_get(),
                                       device.vk_handle(),
                                       vkGetInstanceProcAddr,
                                       vkGetDeviceProcAddr,
                                       NVSDK_NGX_Version_API,
                                       &feature_info);
  if (NVSDK_NGX_FAILED(result)) {
    error_ = ngx_error("Vulkan NGX initialization", result);
    return false;
  }
  ngx_initialized_ = true;
  return true;
}

bool NGXVulkanRuntime::frame_generation_available()
{
  if (!ensure_initialized()) {
    return false;
  }

  std::scoped_lock lock(mutex_);
  if (frame_generation_checked_) {
    return frame_generation_available_;
  }
  frame_generation_checked_ = true;

  NVSDK_NGX_Parameter *capabilities = nullptr;
  const NVSDK_NGX_Result capabilities_result = get_capabilities(&capabilities);
  if (NVSDK_NGX_FAILED(capabilities_result) || capabilities == nullptr) {
    error_ = ngx_error("Vulkan NGX capability query", capabilities_result);
    return false;
  }

  unsigned int available = 0;
  unsigned int needs_driver_update = 0;
  unsigned int max_generated_frames = 0;
  capabilities->Get(NVSDK_NGX_Parameter_FrameGeneration_Available, &available);
  capabilities->Get(NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, &needs_driver_update);
  capabilities->Get(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &max_generated_frames);
  destroy_parameters(capabilities);

  if (needs_driver_update != 0) {
    error_ = "The NVIDIA driver is too old for DLSS Frame Generation";
    return false;
  }
  if (available == 0) {
    error_ = "DLSS Frame Generation is unavailable on this GPU";
    return false;
  }

  max_generated_frames_ = max_generated_frames > 0 ? int(max_generated_frames) : 1;
  frame_generation_available_ = true;
  return true;
}

int NGXVulkanRuntime::max_generated_frames_get() const
{
  return max_generated_frames_;
}

void NGXVulkanRuntime::shutdown_runtime(const VkDevice device)
{
  std::scoped_lock lock(mutex_);
  if (!ngx_initialized_) {
    return;
  }
  shutdown(device);
  ngx_initialized_ = false;
}

const std::string &NGXVulkanRuntime::error_get() const
{
  return error_;
}

NGXVulkanRuntime &ngx_runtime()
{
  static NGXVulkanRuntime runtime;
  return runtime;
}

NVSDK_NGX_Resource_VK ngx_resource(VKTexture &texture,
                                   const int width,
                                   const int height,
                                   const bool read_write)
{
  const VKImageView &image_view = texture.image_view_get(VKImageViewArrayed::NOT_ARRAYED,
                                                         VKImageViewFlags::NO_SWIZZLING);
  NVSDK_NGX_Resource_VK resource = {};
  resource.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
  resource.ReadWrite = read_write;
  resource.Resource.ImageViewInfo.ImageView = image_view.vk_handle();
  resource.Resource.ImageViewInfo.Image = texture.vk_image_handle();
  resource.Resource.ImageViewInfo.SubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  resource.Resource.ImageViewInfo.SubresourceRange.baseMipLevel = 0;
  resource.Resource.ImageViewInfo.SubresourceRange.levelCount = 1;
  resource.Resource.ImageViewInfo.SubresourceRange.baseArrayLayer = 0;
  resource.Resource.ImageViewInfo.SubresourceRange.layerCount = 1;
  resource.Resource.ImageViewInfo.Format = image_view.vk_format();
  resource.Resource.ImageViewInfo.Width = width;
  resource.Resource.ImageViewInfo.Height = height;
  return resource;
}

}  // namespace blender::gpu

#endif /* WITH_DLSS_FRAME_GENERATION */
