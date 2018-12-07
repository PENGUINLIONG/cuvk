#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/config.hpp"
#include "cuvk/storage.hpp"
#include "cuvk/pipeline.hpp"
#include "cuvk/span.hpp"
#include <vector>

L_CUVK_BEGIN_

struct Vulkan;
struct PhysicalDeviceInfo;

struct Vulkan {
  VkInstance inst;
  std::vector<PhysicalDeviceInfo> phys_dev_infos;
  VkDebugUtilsMessengerEXT debug_msgr;

  Vulkan() noexcept;
  bool make() noexcept;
  bool make_debug() noexcept;
  void drop() noexcept;
  ~Vulkan() noexcept;

private:
  bool enum_phys_dev() noexcept;
};



struct PhysicalDeviceInfo {
  VkPhysicalDevice phys_dev;
  VkPhysicalDeviceProperties phys_dev_props;
  std::vector<VkQueueFamilyProperties> queue_fam_props;
};



struct Queue {
  VkQueue queue;
  uint32_t queue_fam_idx;
};



struct ContextRequirements {
  const PhysicalDeviceInfo* phys_dev_info;
  L_STATIC const VkPhysicalDeviceFeatures& phys_dev_feats;
  L_STATIC Span<VkQueueFlags> queue_caps;
};
struct Context {
  const ContextRequirements req;

  VkDevice dev;
  size_t nqueue;
  std::array<Queue, MAX_DEV_QUEUE_COUNT> queues;

  Context(const PhysicalDeviceInfo& phys_dev_info, 
    const VkPhysicalDeviceFeatures& phys_dev_feats,
    L_STATIC Span<VkQueueFlags> queue_caps) noexcept;
  bool make() noexcept;
  void drop() noexcept;
  ~Context() noexcept;

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&&) noexcept;
};

L_CUVK_END_
