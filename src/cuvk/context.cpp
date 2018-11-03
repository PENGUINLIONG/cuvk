#include "cuvk/context.hpp"
#include "cuvk/logger.hpp"
#include <array>
#include <exception>
#include <map>
#include <mutex>
#include <fmt/format.h>

const char* APP_NAME = "L_CUVK";

L_CUVK_BEGIN_

Context::Context() :
  _inst(VK_NULL_HANDLE),
  _ver(0),
  _min_ver(VK_VERSION_1_0),
  _dev(VK_NULL_HANDLE),
  _comp(VK_NULL_HANDLE),
  _graph(VK_NULL_HANDLE) {
  LOG.info("constructing vulkan context");
  if (!create_inst()) {
    throw std::runtime_error("unable to create instance");
  }
}
Context::~Context() {
  for (auto& x : _rg) {
    x->context_changing();
    x->_ctxt = nullptr;
  }
  _ver = _min_ver = 0;
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
}
std::vector<PhysicalDeviceInfo> Context::enum_phys_dev() const {
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
    PhysicalDeviceInfo pdi {
      phys_dev,
      props.deviceName,
      translate_dev_ty(props.deviceType),
    };
    LOG.info(fmt::format("found '{} ({})'", pdi.name, pdi.ty));
    rv.emplace_back(pdi);
  }
  LOG.info("{} physical devices were found, {} are filtered out",
    count, filtered);
  return rv;
}
bool Context::select_phys_dev(const PhysicalDeviceInfo& pdi) {
  LOG.info(fmt::format("selected '{} ({})'", pdi.name, pdi.ty));
  auto phys_dev = pdi.handle;

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
  VkDevice dev = VK_NULL_HANDLE;
  if (L_VK <- vkCreateDevice(phys_dev, &dci, nullptr, &dev)) {
    LOG.error("unable to create device, fall back to previous selection");
    return false;
  }

  if (!create_cmd_pools()) {
    LOG.error("unable to create command pools");
    return false;
  }

  // Broadcast changes.
  for (auto& x : _rg) {
    x->context_changing();
    x->_ctxt = shared_from_this();
  }
  _dev = dev;
  for (auto& x : _rg) {
    x->context_changed();
  }
  return true;
}

VkDevice Context::dev() const { return _dev; }

uint32_t Context::find_mem_type(VkMemoryPropertyFlags flags) const {
  auto i = _mem_types.size();
  while (i--) {
    const auto& cur = _mem_types[i];
    if (cur.propertyFlags == flags) {
      return i;
    }
  }
  LOG.error("unable to find recqested ");
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

bool Context::create_inst() {
  uint32_t api_ver;
  L_VK <- vkEnumerateInstanceVersion(&api_ver);
  LOG.info(fmt::format("api version is '{}.{}.{}'", VK_VERSION_MAJOR(api_ver),
    VK_VERSION_MINOR(api_ver), VK_VERSION_PATCH(api_ver)));

  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = APP_NAME;
  ai.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
  ai.apiVersion = api_ver;

  VkInstanceCreateInfo ici{};
  ici.pApplicationInfo = &ai;

  L_VK <- vkCreateInstance(&ici, nullptr, &_inst);

  _ver = api_ver;
}

bool Context::create_cmd_pools() {
  VkCommandPoolCreateInfo cpci;
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

  cpci.queueFamilyIndex = _comp_fam_idx;
  if (L_VK <- vkCreateCommandPool(_dev, &cpci, nullptr, &_comp_pool)) {
    return false;
  }
  cpci.queueFamilyIndex = _graph_fam_idx;
  if (L_VK <- vkCreateCommandPool(_dev, &cpci, nullptr, &_graph_pool)) {
    return false;
  }
  return true;
}



Contextual::Contextual() : _ctxt(nullptr) {
}
Contextual::~Contextual() {
  if (!validate_ctxt()) return;

  if (ctxt != nullptr) {
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


L_CUVK_END_
