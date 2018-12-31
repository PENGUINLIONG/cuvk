#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/config.hpp"
#include "cuvk/span.hpp"
#include <map>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_

template<typename TSize>
struct RawSlice;
template<typename TSize>
struct Sizer;
using ImageSizer = Sizer<uint32_t>;
using BufferSizer = Sizer<VkDeviceSize>;

enum class MemoryVisibility {
  Invisible, DeviceOnly, HostVisible,
};

struct HostInputMemoryView;
struct HostOutputMemoryView;

struct HeapAllocation;
struct DeviceMemorySlice;

struct BufferAllocationRequirements;
struct BufferAllocation;
struct ImageAllocationRequirements;
struct ImageAllocation;
struct HeapManager;

struct BufferSlice;
struct BufferView;
struct ImageSlice;
struct ImageView;










namespace detail {
  template<typename TSize>
  constexpr TSize align(TSize size, TSize alignment) {
    return (size + alignment - 1) / alignment * alignment;
  }
}

// External dependency.
struct Context;



template<typename TSize>
struct RawSlice {
  TSize offset;
  TSize size;
};
using RawImageSlice = RawSlice<uint32_t>;
using RawBufferSlice = RawSlice<VkDeviceSize>;
template<typename TSize>
struct Sizer {
private:
  TSize _offset = 0;

public:
  // Prepare space for `size` aligned.
  template<typename TElem = uint8_t>
  RawSlice<TSize> allocate(TSize size, TSize alignment = 1) {
    auto offset = _offset;
    size = detail::align<TSize>(size * sizeof(TElem), alignment);
    _offset += size;
    return {
      offset,
      size,
    };
  }
  operator TSize() const {
    return _offset;
  }
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

  // Send data to the device memory.
  bool send(const void* data, size_t size) const noexcept;
  // Fetch data from the device memory.
  bool fetch(L_OUT void* data, size_t size) const noexcept;
  // Wipe out the memory with 0.
  bool wipe() const noexcept;

  void* map(size_t size) const noexcept;
  void unmap() const noexcept;
};



struct BufferSlice {
  const BufferAllocation* buf_alloc;

  VkDeviceSize offset;
  VkDeviceSize size;

  DeviceMemorySlice dev_mem_view() const noexcept;
  BufferSlice slice(VkDeviceSize offset, VkDeviceSize size) const noexcept;
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
  BufferSlice slice(RawBufferSlice slice) const noexcept;
  BufferView view(
    VkDeviceSize offset, VkDeviceSize size, VkFormat format) const noexcept;
  BufferView view(RawBufferSlice slice, VkFormat format) const noexcept;
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
  ImageSlice slice(RawImageSlice raw_slice, bool is_array) const noexcept;
  ImageView view(
    uint32_t base_layer, std::optional<uint32_t> nlayer) const noexcept;
  ImageView view(RawImageSlice raw_slice, bool is_array) const noexcept;
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
