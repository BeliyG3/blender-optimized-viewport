/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GHOST_NGXVK.hh"

#ifdef WITH_DLSS_FRAME_GENERATION

#  include <cstdio>

#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>

#  include <nvsdk_ngx.h>
#  include <nvsdk_ngx_vk.h>

namespace blender::ghost {

namespace {

constexpr unsigned long long application_id = 100334311;

HMODULE ngx_core_load()
{
  static HMODULE module = []() -> HMODULE {
    wchar_t ngx_path[MAX_PATH] = L"";
    HKEY ngx_key = nullptr;
    LSTATUS result = RegOpenKeyExW(HKEY_LOCAL_MACHINE,
                                   L"System\\CurrentControlSet\\Services\\nvlddmkm\\Parameters\\"
                                   L"NGXCore",
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

NVSDK_NGX_FeatureDiscoveryInfo discovery_info()
{
  NVSDK_NGX_FeatureDiscoveryInfo info = {};
  info.SDKVersion = NVSDK_NGX_Version_API;
  info.FeatureID = NVSDK_NGX_Feature_FrameGeneration;
  info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
  info.Identifier.v.ApplicationId = application_id;
  info.ApplicationDataPath = L".";
  return info;
}

void result_error(const char *operation, NVSDK_NGX_Result result, std::string &r_error);

bool required_extensions_get(Vector<std::string> &r_instance_extensions,
                             Vector<std::string> &r_device_extensions,
                             std::string &r_error)
{
  HMODULE module = ngx_core_load();
  if (module == nullptr) {
    r_error = "NVIDIA NGX core (_nvngx.dll) was not found";
    return false;
  }

  using Function = decltype(&NVSDK_NGX_VULKAN_RequiredExtensions);
  Function function = reinterpret_cast<Function>(
      GetProcAddress(module, "NVSDK_NGX_VULKAN_RequiredExtensions"));
  if (function == nullptr) {
    r_error = "The NVIDIA driver does not expose Vulkan NGX extension discovery";
    return false;
  }

  unsigned int instance_count = 0;
  unsigned int device_count = 0;
  const char **instance_extensions = nullptr;
  const char **device_extensions = nullptr;
  const NVSDK_NGX_Result result = function(
      &instance_count, &instance_extensions, &device_count, &device_extensions);
  if (NVSDK_NGX_FAILED(result)) {
    result_error("NGX required extension discovery", result, r_error);
    return false;
  }

  for (unsigned int index = 0; index < instance_count; index++) {
    r_instance_extensions.append(instance_extensions[index]);
  }
  for (unsigned int index = 0; index < device_count; index++) {
    r_device_extensions.append(device_extensions[index]);
  }
  return true;
}

void result_error(const char *operation, const NVSDK_NGX_Result result, std::string &r_error)
{
  char buffer[160];
  std::snprintf(
      buffer, sizeof(buffer), "%s failed with NGX result 0x%08x", operation, uint32_t(result));
  r_error = buffer;
}

}  // namespace

bool ngx_vk_instance_extensions_get(Vector<std::string> &r_extensions, std::string &r_error)
{
  r_extensions.clear();
  HMODULE module = ngx_core_load();
  if (module == nullptr) {
    r_error = "NVIDIA NGX core (_nvngx.dll) was not found";
    return false;
  }

  using Function = decltype(&NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements);
  Function function = reinterpret_cast<Function>(
      GetProcAddress(module, "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements"));
  if (function == nullptr) {
    r_error = "The NVIDIA driver does not expose Vulkan NGX extension discovery";
    return false;
  }

  const NVSDK_NGX_FeatureDiscoveryInfo info = discovery_info();
  uint32_t count = 0;
  VkExtensionProperties *properties = nullptr;
  const NVSDK_NGX_Result result = function(&info, &count, &properties);
  if (NVSDK_NGX_FAILED(result)) {
    Vector<std::string> device_extensions;
    return required_extensions_get(r_extensions, device_extensions, r_error);
  }

  for (uint32_t index = 0; index < count; index++) {
    r_extensions.append(properties[index].extensionName);
  }
  return true;
}

bool ngx_vk_device_extensions_get(const VkInstance instance,
                                  const VkPhysicalDevice physical_device,
                                  Vector<std::string> &r_extensions,
                                  std::string &r_error)
{
  r_extensions.clear();
  HMODULE module = ngx_core_load();
  if (module == nullptr) {
    r_error = "NVIDIA NGX core (_nvngx.dll) was not found";
    return false;
  }

  using Function = decltype(&NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements);
  Function function = reinterpret_cast<Function>(
      GetProcAddress(module, "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements"));
  if (function == nullptr) {
    r_error = "The NVIDIA driver does not expose Vulkan NGX device extension discovery";
    return false;
  }

  const NVSDK_NGX_FeatureDiscoveryInfo info = discovery_info();
  uint32_t count = 0;
  VkExtensionProperties *properties = nullptr;
  const NVSDK_NGX_Result result = function(instance, physical_device, &info, &count, &properties);
  if (NVSDK_NGX_FAILED(result)) {
    Vector<std::string> instance_extensions;
    return required_extensions_get(instance_extensions, r_extensions, r_error);
  }

  for (uint32_t index = 0; index < count; index++) {
    r_extensions.append(properties[index].extensionName);
  }
  return true;
}

}  // namespace blender::ghost

#endif
