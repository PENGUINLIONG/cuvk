#include "cuvk/context.hpp"
#include "cuvk/logger.hpp"
#include <array>
#include <exception>
#include <map>

const char* APP_NAME = "L_CUVK";

L_CUVK_BEGIN_

#ifndef NDEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL validation_cb(
  VkDebugUtilsMessageSeverityFlagBitsEXT lv,
  VkDebugUtilsMessageTypeFlagsEXT ty,
  const VkDebugUtilsMessengerCallbackDataEXT* data,
  void* user_data) {

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
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
  const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
  const VkAllocationCallbacks* pAllocator,
  VkDebugUtilsMessengerEXT* pCallback) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
    vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pCallback);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}
void DestroyDebugUtilsMessengerEXT(VkInstance instance,
  VkDebugUtilsMessengerEXT callback,
  const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
    vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, callback, pAllocator);
  }
}
#endif

Context::Context() :
  _inst(VK_NULL_HANDLE),
  _ver(0), _min_ver(VK_VERSION_1_0),
  _dev(VK_NULL_HANDLE),
  _comp_fam_idx(0), _graph_fam_idx(0),
  _comp(VK_NULL_HANDLE), _graph(VK_NULL_HANDLE),
  _comp_pool(VK_NULL_HANDLE), _graph_pool(VK_NULL_HANDLE),
  _mem_types(), _mem_heaps(),
  _rg() {
  LOG.info("constructing vulkan context");
  if (!create_inst()) {
    throw std::runtime_error("unable to create instance");
  }
}
Context::~Context() {
  invalidate();

#ifndef NDEBUG
  DestroyDebugUtilsMessengerEXT(_inst, _cb, nullptr);
#endif // !NDEBUG


  _ver = _min_ver = 0;
  if (_inst) {
    vkDestroyInstance(_inst, nullptr);
    _inst = VK_NULL_HANDLE;
  }
}
bool Context::require_min_ver(uint32_t ver) {
  LOG.info(fmt::format("required minimum '{}.{}.{}'", VK_VERSION_MAJOR(ver),
    VK_VERSION_MINOR(ver), VK_VERSION_PATCH(ver)));
  bool rv;
  if (rv = _ver < ver) {
    LOG.info("required version not supported");
  }
  _min_ver = ver;
  return rv;
}
// Translate device type enums to C-string.
const char* translate_dev_ty(VkPhysicalDeviceType ty) {
  switch (ty) {
  case VK_PHYSICAL_DEVICE_TYPE_OTHER: return "Other";
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "IntegratedGpu";
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "DiscreteGpu";
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "VirtulaGpu";
  case VK_PHYSICAL_DEVICE_TYPE_CPU: return "Cpu";
  }
  return "Unknown";
}
const std::vector<PhysicalDeviceInfo> Context::enum_phys_dev() const {
  LOG.info("enumerating physical devices");
  uint32_t count;
  if (L_VK <- vkEnumeratePhysicalDevices(_inst, &count, nullptr)) {
    return {};
  }
  std::vector<VkPhysicalDevice> phys_devs;
  phys_devs.resize(count);
  if (L_VK <- vkEnumeratePhysicalDevices(_inst, &count, phys_devs.data())) {
    return {};
  }

  std::vector<PhysicalDeviceInfo> rv;
  rv.reserve(count);
  auto filtered = 0;
  for (const auto& phys_dev : phys_devs) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_dev, &props);
    if (props.apiVersion < _min_ver) {
      ++filtered;
      continue;
    }
    LOG.info(fmt::format("found '{} ({})'",
      props.deviceName, translate_dev_ty(props.deviceType)));
    rv.emplace_back(PhysicalDeviceInfo{ phys_dev, props });
  }
  LOG.info("{} physical devices were found, {} are filtered out",
    count, filtered);
  return rv;
}
bool Context::select_phys_dev(const PhysicalDeviceInfo& pdi) {

  LOG.info(fmt::format("selected '{} ({})'",
    pdi.props.deviceName, translate_dev_ty(pdi.props.deviceType)));
  auto phys_dev = pdi.phys_dev;
  _limits = pdi.props.limits;
  
  // Clean up existing bound device.
  invalidate();

  // Ger memory properties.
  VkPhysicalDeviceMemoryProperties pdmp;
  vkGetPhysicalDeviceMemoryProperties(phys_dev, &pdmp);
  _mem_types.resize(pdmp.memoryTypeCount);
  std::memcpy(_mem_types.data(), pdmp.memoryTypes,
    pdmp.memoryTypeCount * sizeof(VkMemoryType));
  _mem_heaps.resize(pdmp.memoryHeapCount);
  std::memcpy(_mem_heaps.data(), pdmp.memoryHeaps,
    pdmp.memoryHeapCount * sizeof (VkMemoryHeap));

  // Get queue family properties.
  uint32_t i = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &i, nullptr);
  std::vector<VkQueueFamilyProperties> qfps;
  qfps.resize(i);
  vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &i, qfps.data());
  if (i == 0) {
    LOG.error("this physical device has no queue, fall back to previous " \
                "selection");
    return false;
  }

  // Find one compute queue and one graphics queue.
  VkQueue comp = VK_NULL_HANDLE, graph = VK_NULL_HANDLE;
  std::array<VkDeviceQueueCreateInfo, 2> dqcis{};
  float default_priority = 0.5;
  auto& comp_dqci = dqcis[0];
  auto& graph_dqci = dqcis[1];
  while (i-- && (comp_dqci.sType == 0 || graph_dqci.sType == 0)) {
    auto qfp = qfps[i];
    if (comp_dqci.sType == 0 &&
        qfp.queueFlags & VK_QUEUE_COMPUTE_BIT &&
        qfp.queueCount > 0) {
      comp_dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      comp_dqci.queueFamilyIndex = i;
      comp_dqci.queueCount = 1;
      comp_dqci.pQueuePriorities = &default_priority;
    }
    if (graph_dqci.sType == 0 &&
        qfp.queueFlags & VK_QUEUE_GRAPHICS_BIT &&
        qfp.queueCount > 0) {
      graph_dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      graph_dqci.queueFamilyIndex = i;
      graph_dqci.queueCount = 1;
      graph_dqci.pQueuePriorities = &default_priority;
    }
  }
  if (comp_dqci.sType == 0 || graph_dqci.sType == 0) {
    LOG.warning("this physical device has insufficient capability, fall back " \
                "to previous selection");
    return false;
  }
  // Create device and queues.
  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 2;
  dci.pQueueCreateInfos = dqcis.data();
  if (L_VK <- vkCreateDevice(phys_dev, &dci, nullptr, &_dev)) {
    LOG.error("unable to create device, fall back to previous selection");
    return false;
  }

  // Get queues.
  vkGetDeviceQueue(_dev, _comp_fam_idx, 0, &_comp);
  vkGetDeviceQueue(_dev, _graph_fam_idx, 0, &_graph);

  // Broadcast changes.
  if (!create_cmd_pools()) {
    LOG.error("unable to create command pools");
    return false;
  }

  for (auto& x : _rg) {
    // On failure, invalidate the context.
    if (!x->context_changed()) {
      LOG.error("unable to apply change in device, the current context is " \
        "invalidated.");
      invalidate();
      return false;
    }
  }
  return true;
}

void Context::invalidate() {
  for (auto& x : _rg) {
    x->context_changing();
    x->_ctxt = nullptr;
  }
  _mem_types.clear();
  _mem_heaps.clear();
  _comp = _graph = VK_NULL_HANDLE;
  if (_comp_pool) {
    vkDestroyCommandPool(_dev, _comp_pool, nullptr);
    _comp_pool = VK_NULL_HANDLE;
  }
  if (_graph_pool) {
    vkDestroyCommandPool(_dev, _graph_pool, nullptr);
    _graph_pool = VK_NULL_HANDLE;
  }
  if (_dev) {
    vkDestroyDevice(_dev, nullptr);
    _dev = VK_NULL_HANDLE;
  }
}

VkDevice Context::dev() const { return _dev; }

const VkPhysicalDeviceLimits& Context::limits() const {
  return _limits;
}

std::string translate_mem_props(VkMemoryPropertyFlags props) {
  std::string out;
  if (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
    if (!out.empty()) out += " | ";
    out += "DEVICE_LOCAL";
  }
  if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    if (!out.empty()) out += " | ";
    out += "HOST_VISIBLE";
  }
  if (props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    if (!out.empty()) out += " | ";
    out += "HOST_COHERENT";
  }
  if (props & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
    if (!out.empty()) out += " | ";
    out += "HOST_CACHED";
  }
  if (props & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) {
    if (!out.empty()) out += " | ";
    out += "LAZY_ALLOCATED";
  }
  if (props & VK_MEMORY_PROPERTY_PROTECTED_BIT) {
    if (!out.empty()) out += " | ";
    out += "PROTECTED";
  }
  return out;
}

uint32_t Context::find_mem_type(VkMemoryPropertyFlags flags) const {
  auto i = static_cast<uint32_t>(_mem_types.size());
  while (i--) {
    const auto& cur = _mem_types[i];
    if (cur.propertyFlags == flags) {
      LOG.debug("found memory type {}", translate_mem_props(flags));
      return i;
    }
  }
  LOG.debug("unable to find requested memory type {}",
    translate_mem_props(flags));
  return VK_MAX_MEMORY_TYPES;
}
const std::vector<VkMemoryHeap>& Context::get_mem_heap() const {
  return _mem_heaps;
}

uint32_t Context::get_queue_fam_idx(ExecType ty) const {
  return ty == ExecType::Compute ? _comp_fam_idx : _graph_fam_idx;
}
VkQueue Context::get_queue(ExecType ty) const {
  return ty == ExecType::Compute ? _comp : _graph;
}
VkCommandPool Context::get_cmd_pool(ExecType ty) const {
  return ty == ExecType::Compute ? _comp_pool : _graph_pool;
}

size_t min_multiple(size_t size, size_t base) {
  return ((size + base - 1) / base) * base;
}
size_t Context::get_aligned_size(size_t size, BufferType ty) const {
  switch (ty)
  {
  case cuvk::BufferType::UniformBuffer:
    return min_multiple(size, _limits.minUniformBufferOffsetAlignment);
  case cuvk::BufferType::StorageBuffer:
    return min_multiple(size, _limits.minStorageBufferOffsetAlignment);
  case cuvk::BufferType::TexelBuffer:
    return min_multiple(size, _limits.minTexelBufferOffsetAlignment);
  }
}

bool Context::create_inst() {
  uint32_t api_ver;
  if (L_VK <- vkEnumerateInstanceVersion(&api_ver)) {
    LOG.error("unable to fetch api version");
    return false;
  }
  LOG.info("api version is '{}.{}.{}'", VK_VERSION_MAJOR(api_ver),
    VK_VERSION_MINOR(api_ver), VK_VERSION_PATCH(api_ver));

  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = APP_NAME;
  ai.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
  ai.apiVersion = api_ver;

  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;

#ifndef NDEBUG
  std::array<const char*, 1> vlayer = { "VK_LAYER_LUNARG_standard_validation" };
  std::array<const char*, 1> vext = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
  ici.enabledLayerCount = vlayer.size();
  ici.ppEnabledLayerNames = vlayer.data();
  ici.enabledExtensionCount = vext.size();
  ici.ppEnabledExtensionNames = vext.data();
#endif // !NDEBUG

  if (L_VK <- vkCreateInstance(&ici, nullptr, &_inst)) {
    LOG.error("unable to create vulkan instance");
    return false;
  }

  _ver = api_ver;

#ifndef NDEBUG
  VkDebugUtilsMessengerCreateInfoEXT dumci = {};
  dumci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  dumci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  dumci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  dumci.pfnUserCallback = validation_cb;
  if (CreateDebugUtilsMessengerEXT(_inst, &dumci, nullptr, &_cb) != VK_SUCCESS) {
    throw std::runtime_error("failed to set up debug callback");
}
#endif // !NDEBUG

  return true;
}

bool Context::create_cmd_pools() {
  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

  cpci.queueFamilyIndex = _comp_fam_idx;
  if (L_VK <- vkCreateCommandPool(_dev, &cpci, nullptr, &_comp_pool)) {
    LOG.error("unable to create command pool for compute pipelines");
    return false;
  }
  cpci.queueFamilyIndex = _graph_fam_idx;
  if (L_VK <- vkCreateCommandPool(_dev, &cpci, nullptr, &_graph_pool)) {
    LOG.error("unable to create command pool for graphics pipelines");
    return false;
  }
  return true;
}



Contextual::Contextual() : _ctxt(nullptr) {
}
Contextual::~Contextual() {
  if (!validate_ctxt()) return;

  if (_ctxt != nullptr) {
    for (auto cx = _ctxt->_rg.begin(); cx != _ctxt->_rg.end(); ++cx) {
      if (*cx == this) {
        _ctxt->_rg.erase(cx);
        return;
      }
    }
  }
}

bool Contextual::validate_ctxt() const {
  return _ctxt == nullptr;
}
const Context& Contextual::ctxt() const {
  return *_ctxt;
}
Context& Contextual::ctxt() {
  return *_ctxt;
}

L_CUVK_END_
