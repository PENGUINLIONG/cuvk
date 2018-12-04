#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/config.hpp"
#include "cuvk/span.hpp"
#include <map>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_

struct Context;

// Host memory view.
struct HostInputMemoryView;
struct HostOutputMemoryView;
// Device memory view.
struct DeviceMemorySlice;
// Requirements.
struct BufferAllocationRequirements;
struct ImageAllocationRequirements;
// Allocations.
struct BufferAllocation;
struct BufferSlice;
struct ImageAllocation;
// Views.
struct BufferView;
struct ImageView;
// Heap manager.
struct HeapManager;



enum class MemoryVisibility {
  Invisible, DeviceOnly, HostVisible,
};



struct HostInputMemoryView {
  const void* data;
  size_t size;
};
struct HostOutputMemoryView {
  L_OUT void* data;
  size_t size;
};



struct HeapAllocation {
  const Context* ctxt;
  VkDeviceSize alloc_size;
  VkDeviceMemory dev_mem;
};
struct DeviceMemorySlice {
  const HeapAllocation* heap_alloc;

  VkDeviceSize offset;
  VkDeviceSize size;

  bool send(const void* data, size_t size) const noexcept;
  bool fetch(L_OUT void* data, size_t size) const noexcept;
private:
  void* map(size_t size) const noexcept;
  void unmap() const noexcept;
};



struct BufferSlice {
  const BufferAllocation* buf_alloc;

  VkDeviceSize offset;
  VkDeviceSize size;

  DeviceMemorySlice dev_mem_view() const noexcept;
};
struct ImageSlice {
  const ImageAllocation* img_alloc;

  uint32_t base_layer;
  std::optional<uint32_t> nlayer;

  DeviceMemorySlice dev_mem_view() const noexcept;
};



struct BufferView {
  BufferSlice buf_slice;
  VkFormat format;

  VkBufferView buf_view;

  BufferView(const BufferSlice& slice, VkFormat format) noexcept;
  // Buffer views are a bit different, since buffer data can be used without a
  // `VkBufferView` being created. `VkBufferView` is required when we are using
  // it as texel buffer, that is, it's a pixel buffer.
  bool make() noexcept;
  void drop() noexcept;
  ~BufferView() noexcept;

  operator const BufferSlice&() const noexcept;

  DeviceMemorySlice dev_mem_view() const noexcept;
};
struct ImageView {
  ImageSlice img_slice;

  VkImageView img_view;

  ImageView(const ImageSlice& img_slice) noexcept;
  bool make() noexcept;
  void drop() noexcept;
  ~ImageView() noexcept;

  operator const ImageSlice&() const noexcept;

  DeviceMemorySlice dev_mem_view() const noexcept;
};



struct BufferAllocationRequirements {
  VkDeviceSize size;
  VkBufferUsageFlags usage;
  MemoryVisibility visibility;
};
struct BufferAllocation {
  const Context* ctxt;
  const BufferAllocationRequirements req;
  const HeapAllocation* heap_alloc;

  VkBuffer buf;
  VkDeviceSize offset;

  BufferSlice slice(VkDeviceSize offset, VkDeviceSize size) const noexcept;
  BufferView view(
    VkDeviceSize offset, VkDeviceSize size, VkFormat format) const noexcept;
};



struct ImageAllocationRequirements {
  VkExtent2D extent;
  std::optional<uint32_t> nlayer;
  VkFormat format;
  VkImageUsageFlags usage;
  VkImageTiling tiling;
  MemoryVisibility visibility;
};
struct ImageAllocation {
  const Context* ctxt;
  const ImageAllocationRequirements req;
  const HeapAllocation* heap_alloc;

  VkImage img;
  VkDeviceSize offset;

  ImageSlice slice(
    uint32_t base_layer, std::optional<uint32_t> nlayer) const noexcept;
  ImageView view(
    uint32_t base_layer, std::optional<uint32_t> nlayer) const noexcept;
};



struct HeapManager {
  const Context* ctxt;

  std::vector<VkMemoryType> mem_types;
  std::vector<VkMemoryHeap> mem_heaps;

  // Memory type index to heap allocation mapping.
  std::map<uint32_t, HeapAllocation> heap_allocs;
  std::list<BufferAllocation> buf_allocs;
  std::list<ImageAllocation> img_allocs;

  HeapManager(const Context& ctxt) noexcept;
  bool make() noexcept;
  void drop() noexcept;
  ~HeapManager() noexcept;

  uint32_t find_mem_type(
    uint32_t hint, VkMemoryPropertyFlags flags) const noexcept;
  std::pair<uint32_t, VkMemoryPropertyFlags> find_mem_type(
    uint32_t hint, Span<VkMemoryPropertyFlags> fallbacks) const noexcept;
  uint32_t get_mem_heap_idx(uint32_t mem_type_idx) const noexcept;

  const BufferAllocation& declare_buf(
    size_t size, VkBufferUsageFlags usage,
    MemoryVisibility visibility) noexcept;
  const ImageAllocation& declare_img(
    const VkExtent2D& extent, std::optional<uint32_t> nlayer, VkFormat format,
    VkImageUsageFlags usage, VkImageTiling tiling,
    MemoryVisibility visibility) noexcept;

private:
  bool make_rscs() noexcept;
  bool make_bufs() noexcept;
  bool make_imgs() noexcept;
  bool alloc_mem() noexcept;
  bool bind_rscs() noexcept;
  bool bind_bufs() noexcept;
  bool bind_imgs() noexcept;
};


L_CUVK_END_
