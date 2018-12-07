#include "cuvk/context.hpp"
#include "cuvk/logger.hpp"
#include <array>
#include <exception>
#include <map>

L_CUVK_BEGIN_

constexpr const char* DEBUG_LAYER= "VK_LAYER_LUNARG_standard_validation";
constexpr const char* DEBUG_EXT = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

static const VkApplicationInfo APP_INFO = {
  VK_STRUCTURE_TYPE_APPLICATION_INFO,
  nullptr,
  VulkanRequirements::app_name, VulkanRequirements::app_version,
  nullptr, 0,
  VulkanRequirements::vulkan_version
};

static VKAPI_ATTR VkBool32 VKAPI_CALL validation_cb(
  VkDebugUtilsMessageSeverityFlagBitsEXT lv,
  VkDebugUtilsMessageTypeFlagsEXT ty,
  const VkDebugUtilsMessengerCallbackDataEXT* data,
  void* user_data) noexcept {

  LogLevel level;
  if (lv >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    level = "VALID_ERROR";
  } else if (lv >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    level = "VALID_WARNING";
  } else if (lv >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    level = "VALID_INFO";
  } else {
    level = "VALID_TRACE";
  }

  LOG.log(level, data->pMessage);
  return VK_FALSE;
}


Vulkan::Vulkan() noexcept :
  inst(VK_NULL_HANDLE),
  phys_dev_infos(),
  debug_msgr(VK_NULL_HANDLE) {}
bool Vulkan::make() noexcept {
  LOG.info("creating vulkan instance");

  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &APP_INFO;

  if (L_VK <- vkCreateInstance(&ici, nullptr, &inst)) {
    LOG.error("unable to create vulkan instance in debug mode");
    return false;
  }
  if (!enum_phys_dev()) { return false; }
  return true;
}
bool Vulkan::make_debug() noexcept {
  LOG.info("creating vulkan instance in debug mode");

  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &APP_INFO;
  ici.enabledLayerCount = 1;
  ici.ppEnabledLayerNames = &DEBUG_LAYER;
  ici.enabledExtensionCount = 1;
  ici.ppEnabledExtensionNames = &DEBUG_EXT;

  if (L_VK <- vkCreateInstance(&ici, nullptr, &inst)) {
    LOG.error("unable to create vulkan instance");
    return {};
  }
  
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
    vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
  if (func == nullptr) {
    LOG.error("`VK_EXT_debug_utils` extension present but "
      "`vkCreateDebugUtilsMessengerEXT` doesn't exist");
    return false;
  }

  VkDebugUtilsMessengerCreateInfoEXT dumci = {};
  dumci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  dumci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  dumci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  dumci.pfnUserCallback = validation_cb;

  if (L_VK <- func(inst, &dumci, nullptr, &debug_msgr)) {
    LOG.error("unable to create debug messenger");
    return false;
  }
  if (!enum_phys_dev()) { return false; }
  return true;
}
void Vulkan::drop() noexcept {
  if (debug_msgr) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
      vkGetInstanceProcAddr(inst, "vkDestroyDebugUtilsMessengerEXT");
    if (func == nullptr) {
      LOG.error("`vkCreateDebugUtilsMessengerEXT` present but "
        "`vkDestroyDebugUtilsMessengerEXT` doesn't exist");
      std::terminate();
    }
    func(inst, debug_msgr, nullptr);
    debug_msgr = VK_NULL_HANDLE;
  }
  phys_dev_infos.clear();
  if (inst) {
    vkDestroyInstance(inst, nullptr);
    inst = VK_NULL_HANDLE;
  }
}
Vulkan::~Vulkan() noexcept { drop(); }

// Translate device type enums to C-string.
constexpr const char* translate_dev_ty(VkPhysicalDeviceType ty) noexcept {
  switch (ty) {
  case VK_PHYSICAL_DEVICE_TYPE_OTHER: return "Other";
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "IntegratedGpu";
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "DiscreteGpu";
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "VirtulaGpu";
  case VK_PHYSICAL_DEVICE_TYPE_CPU: return "Cpu";
  }
  return "Unknown";
}
bool Vulkan::enum_phys_dev() noexcept {
  LOG.info("enumerating physical devices");
  uint32_t count;
  if (L_VK <- vkEnumeratePhysicalDevices(inst, &count, nullptr)) {
    LOG.error("unable to enumerate physical devices");
    return false;
  }
  std::vector<VkPhysicalDevice> phys_devs;
  phys_devs.resize(count);
  if (L_VK <- vkEnumeratePhysicalDevices(inst, &count, phys_devs.data())) {
    LOG.error("unable to enumerate physical devices");
    return false;
  }

  phys_dev_infos.reserve(count);
  auto filtered = 0;
  for (const auto& phys_dev : phys_devs) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_dev, &props);
    if (props.apiVersion < VulkanRequirements::vulkan_version) {
      ++filtered;
      continue;
    }
    LOG.info(fmt::format("found '{} ({})'",
      props.deviceName, translate_dev_ty(props.deviceType)));

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> qfps;
    qfps.resize(count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &count, qfps.data());

    phys_dev_infos.emplace_back(PhysicalDeviceInfo {
      phys_dev, props, std::move(qfps) });
  }
  LOG.info("found {} physical devices, {} are filtered out", count, filtered);
  return true;
}

Context::Context(
  const PhysicalDeviceInfo& phys_dev_info, 
  L_STATIC const VkPhysicalDeviceFeatures& phys_dev_feats,
  L_STATIC Span<VkQueueFlags> queue_caps) noexcept :
  req({ &phys_dev_info, phys_dev_feats, queue_caps }) {
  if (queue_caps.size() > MAX_DEV_QUEUE_COUNT) {
    LOG.error("too many queues to be created");
    std::terminate();
  }
}
bool Context::make() noexcept {
  auto& phys_dev_info = req.phys_dev_info;
  LOG.info(fmt::format("making context on '{} ({})'",
    phys_dev_info->phys_dev_props.deviceName,
    translate_dev_ty(phys_dev_info->phys_dev_props.deviceType)));

  auto phys_dev = phys_dev_info->phys_dev;

  std::array<VkDeviceQueueCreateInfo, MAX_DEV_QUEUE_COUNT> dqcis;
  uint32_t ndqci = 0;

  // Find matching queue families.
  auto& queue_caps = req.queue_caps;
  auto& queue_fam_props = phys_dev_info->queue_fam_props;
  // For each wanted queue capacity.
  for (auto i = 0; i < queue_caps.size(); ++i) {
    // For each queue family.
    for (auto j = 0; j < queue_fam_props.size(); ++j ) {
      // Check if capabilities are matching.
      if ((queue_fam_props[j].queueFlags & queue_caps[i]) == queue_caps[i]) {
        VkDeviceQueueCreateInfo dqci {};
        dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqci.queueFamilyIndex = j;
        dqci.queueCount = 1;
        dqci.pQueuePriorities = &DEFAULT_QUEUE_PRIORITY;

        dqcis[ndqci++] = std::move(dqci);
        break;
      }
    }
  }

  // Create device and queues.
  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pEnabledFeatures = &req.phys_dev_feats;
  dci.queueCreateInfoCount = ndqci;
  dci.pQueueCreateInfos = dqcis.data();

  if (L_VK <- vkCreateDevice(phys_dev, &dci, nullptr, &dev)) {
    LOG.error("unable to create device");
    return false;
  }

  // Collect queues.
  for (uint32_t i = 0; i < ndqci; ++i) {
    queues[i].queue_fam_idx = dqcis[i].queueFamilyIndex;
    vkGetDeviceQueue(dev, queues[i].queue_fam_idx, i, &queues[i].queue);
  }
  nqueue = ndqci;

  return true;
}
void Context::drop() noexcept {
  if (dev) {
    vkDestroyDevice(dev, nullptr);
    dev = VK_NULL_HANDLE;
  }
  queues = { VK_NULL_HANDLE };
}
Context::~Context() noexcept { drop(); }

Context::Context(Context&& right) noexcept :
  req(right.req),
  dev(std::exchange(right.dev, nullptr)),
  queues(std::exchange(right.queues, {})) {}

L_CUVK_END_
