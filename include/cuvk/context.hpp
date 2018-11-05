#pragma once
#include "cuvk/comdef.hpp"
#include <array>
#include <string>
#include <vector>
#include <type_traits>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_
using DataId = int;
using ProcId = int;

class Context;
struct DispatcherEntry;
class Dispatcher;


struct PhysicalDeviceInfo {
  VkPhysicalDevice phys_dev;
  VkPhysicalDeviceProperties props;
};

enum class BufferType {
  UniformBuffer,
  StorageBuffer,
  TexelBuffer,
};

enum class ExecType {
  Compute, Graphics
};



class Contextual;

class Context : public std::enable_shared_from_this<Context> {
  friend Contextual;
private:
  // Vulkan RT version.
  uint32_t _ver;
  // Minimum version requirement. Vulkan 1.0 by default.
  uint32_t _min_ver;
  VkInstance _inst;
  VkDevice _dev;
  uint32_t _comp_fam_idx, _graph_fam_idx;
  VkQueue _comp, _graph;
  VkCommandPool _comp_pool, _graph_pool;
  std::vector<VkMemoryType> _mem_types;
  std::vector<VkMemoryHeap> _mem_heaps;

  std::vector<Contextual*> _rg;

#ifndef NDEBUG
  VkDebugUtilsMessengerEXT _cb;
#endif // !NDEBUG

  VkPhysicalDeviceLimits _limits;

  bool create_inst();
  bool create_cmd_pools();

public:
  Context();
  ~Context();

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  // Force the context to use a supporting version of Vulkan. Unsupporting
  // physical devices will be filtered out in enumeration.
  bool require_min_ver(uint32_t ver);
  // Enumerate all devices that have queues supporting both computing and
  // graphics.
  const std::vector<PhysicalDeviceInfo> enum_phys_dev() const;
  // Select a specific physical device and create coresponding logical device
  // and queues.
  bool select_phys_dev(const PhysicalDeviceInfo& pdi);

  // Destruct the device bound to vulkan instance and related functionalities.
  void invalidate();

  VkDevice dev() const;

  const VkPhysicalDeviceLimits& limits() const;

  uint32_t find_mem_type(VkMemoryPropertyFlags flags) const;
  const std::vector<VkMemoryHeap>& get_mem_heap() const;

  uint32_t get_queue_fam_idx(ExecType ty) const;
  VkQueue get_queue(ExecType ty) const;
  VkCommandPool get_cmd_pool(ExecType ty) const;

  // The aligned size of a length of memory on the current physical device.
  size_t get_aligned_size(size_t size, BufferType ty) const;

  template<typename T,
           typename _ = std::enable_if<std::is_base_of_v<Contextual, T>>,
           typename ... TArgs>
  std::shared_ptr<T> make_contextual(TArgs&& ... args) {
    auto rv = std::make_shared<T>(std::forward<TArgs>(args) ...);
    rv->_ctxt = shared_from_this();
    // Invalidate the context if the context is unable to allocate resoureces
    // for the new contextual object.
    if (!rv->context_changed()) {
      invalidate();
      return nullptr;
    }
    _rg.push_back(rv.get());

    return rv;
  }
};

// Base class of all CUVK objects that depends on the `Context`.
class Contextual {
  friend class Context;
private:
  std::shared_ptr<Context> _ctxt;

public:
  Contextual();
  ~Contextual();

  Contextual(const Contextual&) = delete;
  Contextual& operator=(const Contextual&) = delete;

  // Validate that the current contextaul is actually registered to an context.
  // If so, `true` is returned.
  bool validate_ctxt() const;
  // Get the context currently bound.
  const Context& ctxt() const;
  Context& ctxt();

  // Update event when context is about to change. The bound device will be
  // destroyed soon after this event is fired. Implementations must ensure this
  // event handler is idempotent, that is, triggering this event for multiple
  // times will have the same effect as firing it once. This method must not
  // fail.
  virtual bool context_changing() = 0;
  // Update event when context has changed, like when the selected device has
  // changed. Return `true` if no error occurred. If any error occurred durring
  // context change, the context will become invalid.
  virtual bool context_changed() = 0;
};

L_CUVK_END_
