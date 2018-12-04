#include "cuvk/storage.hpp"
#include "cuvk/logger.hpp"
#include "cuvk/context.hpp"
#include <exception>

L_CUVK_BEGIN_

std::array<VkMemoryPropertyFlags, 1> INVISIBLE_FALLBACKS ={
  0,
};
std::array<VkMemoryPropertyFlags, 1> DEVICE_ONLY_FALLBACKS ={
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
};
std::array<VkMemoryPropertyFlags, 5> HOST_VISIBLE_FALLBACKS ={
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT,

  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT,

  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
};

const Span<VkMemoryPropertyFlags> get_mem_prop_fallback(MemoryVisibility vis) {
  switch (vis)
  {
  case cuvk::MemoryVisibility::Invisible:
    return INVISIBLE_FALLBACKS;
  case cuvk::MemoryVisibility::DeviceOnly:
    return DEVICE_ONLY_FALLBACKS;
  case cuvk::MemoryVisibility::HostVisible:
    return HOST_VISIBLE_FALLBACKS;
  default:
    LOG.error("meet impossible branch");
    std::terminate();
  }
}


inline uint32_t get_pixel_size(VkFormat fmt) noexcept {
  switch (fmt) {
  case VK_FORMAT_R32_SINT:
    return sizeof(int32_t);
  case VK_FORMAT_R32_UINT:
    return sizeof(uint32_t);
  case VK_FORMAT_R32_SFLOAT:
    return sizeof(float);
  case VK_FORMAT_R32G32_SFLOAT:
    return 2 * sizeof(float);
  case VK_FORMAT_R32G32B32A32_SFLOAT:
    return 4 * sizeof(float);
  default:
    break;
  }
  LOG.error("unsupported pixel format");
  std::terminate();
}

constexpr VkDeviceSize align_size(
  VkDeviceSize size, VkDeviceSize alignment) noexcept {
  return (size + alignment - 1) / alignment * alignment;
}



bool DeviceMemorySlice::send(const void* data, size_t size) const noexcept {
  auto dev_data = map(size);
  if (dev_data == nullptr) { return false; }
  // TODO: Use better memcpy.
  std::memcpy(dev_data, data, size);
  unmap();
  return true;
}
bool DeviceMemorySlice::fetch(L_OUT void* data, size_t size) const noexcept {
  auto dev_data = map(size);
  if (dev_data == nullptr) { return false; }
  // TODO: Use better memcpy.
  std::memcpy(data, dev_data, size);
  unmap();
  return true;
}
void* DeviceMemorySlice::map(size_t size) const noexcept {
  if (size > this->size) {
    LOG.error("memory write out of range");
    return nullptr;
  }
  auto alignment = heap_alloc->ctxt->req.phys_dev_info->phys_dev_props
    .limits.minMemoryMapAlignment;
  auto map_offset = offset / alignment * alignment;
  auto partial_offset = offset - map_offset;
  auto map_size = align_size(partial_offset + size, alignment);

  void* dev_data;
  if (L_VK <- vkMapMemory(
    heap_alloc->ctxt->dev, heap_alloc->dev_mem,
    map_offset, map_size, 0, &dev_data)) {
    LOG.error("unable to map device data");
    return nullptr;
  }
  return (char*)dev_data + partial_offset;
}
void DeviceMemorySlice::unmap() const noexcept {
  vkUnmapMemory(heap_alloc->ctxt->dev, heap_alloc->dev_mem);
}



DeviceMemorySlice BufferSlice::dev_mem_view() const noexcept {
  return {
    buf_alloc->heap_alloc,
    buf_alloc->offset + offset,
    size
  };
}

DeviceMemorySlice ImageSlice::dev_mem_view() const noexcept {
  auto& req = img_alloc->req;
  auto layer_size =
    req.extent.width * req.extent.height * get_pixel_size(req.format);
  return {
    img_alloc->heap_alloc,
    img_alloc->offset + base_layer * layer_size,
    nlayer.value_or(1) * layer_size,
  };
}



BufferView::BufferView(const BufferSlice& slice, VkFormat format) noexcept :
  buf_slice(slice),
  format(format),
  buf_view(VK_NULL_HANDLE) {}
bool BufferView::make() noexcept {
  if (buf_slice.buf_alloc->req.usage &
    (VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
     VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT)) {
    VkBufferViewCreateInfo bvci {};
    bvci.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    bvci.buffer = buf_slice.buf_alloc->buf;
    bvci.format = format;
    bvci.offset = buf_slice.offset;
    bvci.range = buf_slice.size;

    if (L_VK <- vkCreateBufferView(
      buf_slice.buf_alloc->ctxt->dev, &bvci, nullptr, &buf_view)) {
      LOG.error("unable to create texel buffer view");
      return false;
    }
  }
  return true;
}
void BufferView::drop() noexcept {
  if (buf_view) {
    vkDestroyBufferView(buf_slice.buf_alloc->ctxt->dev, buf_view, nullptr);
    buf_view = VK_NULL_HANDLE;
  }
}
BufferView::~BufferView() noexcept { drop(); }

BufferView::operator const BufferSlice&() const noexcept {
  return buf_slice;
}

DeviceMemorySlice BufferView::dev_mem_view() const noexcept {
  return buf_slice.dev_mem_view();
}

ImageView::ImageView(const ImageSlice& img_slice) noexcept :
  img_slice(img_slice),
  img_view(VK_NULL_HANDLE) {}
bool ImageView::make() noexcept {
  VkImageViewCreateInfo ivci {};
  ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ivci.image = img_slice.img_alloc->img;
  ivci.components = { VK_COMPONENT_SWIZZLE_IDENTITY };
  ivci.format = img_slice.img_alloc->req.format;
  ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  ivci.subresourceRange.baseArrayLayer = img_slice.base_layer;
  ivci.subresourceRange.layerCount = img_slice.nlayer.value_or(1);
  ivci.subresourceRange.baseMipLevel = 0;
  ivci.subresourceRange.levelCount = 1;
  ivci.viewType = img_slice.nlayer.has_value() ?
    VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;

  if (L_VK <- vkCreateImageView(
    img_slice.img_alloc->ctxt->dev, &ivci, nullptr, &img_view)) {
    LOG.error("unable to create image view");
    return false;
  }
  return true;
}
void ImageView::drop() noexcept {
  if (img_view) {
    vkDestroyImageView(img_slice.img_alloc->ctxt->dev, img_view, nullptr);
    img_view = VK_NULL_HANDLE;
  }
}
ImageView::~ImageView() noexcept { drop(); }

ImageView::operator const ImageSlice&() const noexcept {
  return img_slice;
}

DeviceMemorySlice ImageView::dev_mem_view() const noexcept {
  return img_slice.dev_mem_view();
}



VkBuffer create_buf(
  VkDevice dev, const BufferAllocationRequirements& req) noexcept {
  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = req.size;
  bci.usage = req.usage;

  VkBuffer buf;
  if (L_VK <- vkCreateBuffer(dev, &bci, nullptr, &buf)) {
    LOG.error("unable to create image");
    return VK_NULL_HANDLE;
  }
  return buf;
}
VkImage create_img(
  VkDevice dev, const ImageAllocationRequirements& req) noexcept {
  VkImageCreateInfo ici {};
  ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ici.format = req.format;
  ici.extent.width = req.extent.width;
  ici.extent.height = req.extent.height;
  ici.extent.depth = 1;
  ici.arrayLayers = req.nlayer.value_or(1);
  ici.mipLevels = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.tiling = req.tiling;
  ici.usage = req.usage;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImage img;
  if (L_VK <- vkCreateImage(dev, &ici, nullptr, &img)) {
    LOG.error("unable to create image");
    return VK_NULL_HANDLE;
  }
  return img;
}



BufferSlice BufferAllocation::slice(
  VkDeviceSize offset, VkDeviceSize size) const noexcept {
  return { this, offset, size };
}
BufferView BufferAllocation::view(
  VkDeviceSize offset, VkDeviceSize size, VkFormat format) const noexcept {
  return { slice(offset, size), format };
}

ImageSlice ImageAllocation::slice(
  uint32_t base_layer, std::optional<uint32_t> nlayer) const noexcept {
  return { this, base_layer, nlayer };
}
ImageView ImageAllocation::view(
  uint32_t base_layer, std::optional<uint32_t> nlayer) const noexcept {
  return { slice(base_layer, nlayer) };
}



HeapManager::HeapManager(const Context& ctxt) noexcept :
  ctxt(&ctxt),
  mem_types(), 
  mem_heaps(),
  heap_allocs(),
  buf_allocs(),
  img_allocs() {}
std::string translate_mem_props(VkMemoryPropertyFlags props) noexcept {
  std::string out;
  if (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
    if (!out.empty()) out += " + ";
    out += "DeviceOnly";
  }
  if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    if (!out.empty()) out += " + ";
    out += "HostVisible";
  }
  if (props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
    if (!out.empty()) out += " + ";
    out += "HostCoherent";
  }
  if (props & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
    if (!out.empty()) out += " + ";
    out += "HostCached";
  }
  if (props & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) {
    if (!out.empty()) out += " + ";
    out += "LazyAllocated";
  }
  if (props & VK_MEMORY_PROPERTY_PROTECTED_BIT) {
    if (!out.empty()) out += " + ";
    out += "Protected";
  }
  if (out.empty()) {
    out = "(no property)";
  }
  return out;
}
bool HeapManager::make() noexcept {
  LOG.trace("making managed memory dependent resources");
  // Ger memory properties.
  VkPhysicalDeviceMemoryProperties pdmp;
  vkGetPhysicalDeviceMemoryProperties(ctxt->req.phys_dev_info->phys_dev, &pdmp);
  mem_types.resize(pdmp.memoryTypeCount);
  std::memcpy(mem_types.data(), pdmp.memoryTypes,
    pdmp.memoryTypeCount * sizeof(VkMemoryType));
  mem_heaps.resize(pdmp.memoryHeapCount);
  std::memcpy(mem_heaps.data(), pdmp.memoryHeaps,
    pdmp.memoryHeapCount * sizeof (VkMemoryHeap));

  uint32_t i = 0;
  for (auto& mem_type : mem_types) {
    LOG.info("discovered memory type #{}: {}", i++,
      translate_mem_props(mem_type.propertyFlags));
  }

  if (!make_rscs()) {
    LOG.error("unable to create resources");
    return false;
  }
  if (!alloc_mem()) {
    LOG.error("unable to allocate memory for resources");
    return false;
  }
  if (!bind_rscs()) {
    LOG.error("unable to bind resources to memory");
    return false;
  }
  return true;
}
void HeapManager::drop() noexcept {
  LOG.trace("dropping managed memory dependent resources");
  for (auto& buf_alloc : buf_allocs) {
    vkDestroyBuffer(ctxt->dev, buf_alloc.buf, nullptr);
    buf_alloc.buf = VK_NULL_HANDLE;
  }
  for (auto& img_alloc : img_allocs) {
    vkDestroyImage(ctxt->dev, img_alloc.img, nullptr);
    img_alloc.img = VK_NULL_HANDLE;
  }
  for (auto& heap_alloc : heap_allocs) {
    vkFreeMemory(ctxt->dev, heap_alloc.second.dev_mem, nullptr);
    heap_alloc.second.dev_mem = VK_NULL_HANDLE;
  }
}
HeapManager::~HeapManager() noexcept { drop(); }



uint32_t HeapManager::find_mem_type(uint32_t hint,
  VkMemoryPropertyFlags flags) const noexcept {
  auto n = static_cast<uint32_t>(mem_types.size());
  for (uint32_t i = 0; i < n; ++i) {
    if (hint & 1) {
      const auto& cur = mem_types[i];
      if (cur.propertyFlags == flags) {
        return i;
      }
    }
    hint >>= 1;
  }
  return VK_MAX_MEMORY_TYPES;
}
std::pair<uint32_t, VkMemoryPropertyFlags> HeapManager::find_mem_type(
  uint32_t hint, Span<VkMemoryPropertyFlags> fallbacks) const noexcept {
  for (const auto& mem_props : fallbacks) {
    auto mem_type_idx = find_mem_type(hint, mem_props);
    if (mem_type_idx < VK_MAX_MEMORY_TYPES) {
      return std::make_pair(mem_type_idx, mem_props);
    }
  }
  return std::make_pair(VK_MAX_MEMORY_TYPES, 0);
}
uint32_t HeapManager::get_mem_heap_idx(uint32_t mem_type_idx) const noexcept {
  return mem_types[mem_type_idx].heapIndex;
}



bool HeapManager::make_rscs() noexcept {
  return make_bufs() && make_imgs();
}
bool HeapManager::make_bufs() noexcept  {
  uint32_t i = 0;
  // Create buffers.
  for (auto& buf_alloc : buf_allocs) {
    auto buf = create_buf(ctxt->dev, buf_alloc.req);
    if (buf == VK_NULL_HANDLE) {
      return false;
    }
    buf_alloc.buf = buf;

    // Check memory requirements.
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(ctxt->dev, buf, &mem_req);

    auto fallback = get_mem_prop_fallback(buf_alloc.req.visibility);
    auto pair = find_mem_type(mem_req.memoryTypeBits, fallback);
    auto mem_type_idx = pair.first;
    if (mem_type_idx == VK_MAX_MEMORY_TYPES) {
      LOG.error("unable to find memory type for buffer #{}", i);
      return false;
    } else {
      LOG.info("matched memory type #{} ({}) for buffer #{}", mem_type_idx,
        translate_mem_props(pair.second), i);
    }
    auto mem_heap_idx = get_mem_heap_idx(mem_type_idx);

    auto alloc = heap_allocs.find(mem_type_idx);
    if (alloc == heap_allocs.end()) {
      alloc = heap_allocs.emplace_hint(alloc,
        mem_type_idx, HeapAllocation { ctxt });
    }

    auto& alloc_size = alloc->second.alloc_size;
    auto offset_aligned = align_size(alloc_size, mem_req.alignment);
    buf_alloc.offset = offset_aligned;
    buf_alloc.heap_alloc = &alloc->second;
    alloc_size = offset_aligned + mem_req.size;
    ++i;
  }
  return true;
}
bool HeapManager::make_imgs() noexcept {
  uint32_t i = 0;
  // Create images.
  for (auto& img_alloc : img_allocs) {
    auto img = create_img(ctxt->dev, img_alloc.req);
    if (img == VK_NULL_HANDLE) {
      return false;
    }
    img_alloc.img = img;

    // Check memory requirements.
    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(ctxt->dev, img, &mem_req);

    auto fallback = get_mem_prop_fallback(img_alloc.req.visibility);
    auto pair = find_mem_type(mem_req.memoryTypeBits, fallback);
    auto mem_type_idx = pair.first;
    if (mem_type_idx == VK_MAX_MEMORY_TYPES) {
      LOG.error("unable to find memory type for image #{} (0 based)", i);
      return false;
    } else {
      LOG.info("matched memory type #{} ({}) for image #{}", mem_type_idx,
        translate_mem_props(pair.second), i);
    }
    auto mem_heap_idx = get_mem_heap_idx(mem_type_idx);

    auto alloc = heap_allocs.find(mem_type_idx);
    if (alloc == heap_allocs.end()) {
      alloc = heap_allocs.emplace_hint(alloc,
        mem_type_idx, HeapAllocation { ctxt });
    }

    auto& alloc_size = alloc->second.alloc_size;
    auto offset_aligned = align_size(alloc_size, mem_req.alignment);
    img_alloc.offset = offset_aligned;
    img_alloc.heap_alloc = &alloc->second;
    alloc_size = offset_aligned + mem_req.size;
    ++i;
  }
  return true;
}
bool HeapManager::alloc_mem() noexcept {
  // Allocate memory for each type.
  for (auto& pair : heap_allocs) {
    auto& heap_alloc = pair.second;
    if (heap_alloc.alloc_size > 0) {
      VkMemoryAllocateInfo mai {};
      mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      mai.allocationSize = heap_alloc.alloc_size;
      mai.memoryTypeIndex = pair.first;

      VkDeviceMemory dev_mem;
      if (L_VK <- vkAllocateMemory(ctxt->dev, &mai, nullptr, &dev_mem)) {
        LOG.error("unable to allocate memory for resources requiring memory "
          "type {}", pair.first);
        return false;
      }
      LOG.info("allocated memory for resources requiring memory type {}",
        pair.first);
      heap_alloc.dev_mem = dev_mem;
    }
  }
  return true;
}
bool HeapManager::bind_rscs() noexcept {
  return bind_bufs() && bind_imgs();
}
bool HeapManager::bind_bufs() noexcept {
  uint32_t i = 0;
  for (auto& buf_alloc : buf_allocs) {
    if (L_VK <- vkBindBufferMemory(
      ctxt->dev, buf_alloc.buf,
      buf_alloc.heap_alloc->dev_mem, buf_alloc.offset)) {
      LOG.error("unable to bind buffer #{} to its memory allocation", i);
      return false;
    }
    LOG.info("bound image #{} to its memory allocation", i);
  }
  return true;
}
bool HeapManager::bind_imgs() noexcept {
  uint32_t i = 0;
  for (auto& img_alloc : img_allocs) {
    if (L_VK <- vkBindImageMemory(
      ctxt->dev, img_alloc.img,
      img_alloc.heap_alloc->dev_mem, img_alloc.offset)) {
      LOG.error("unable to bind image #{} to its memory allocation", i);
      return false;
    }
    LOG.info("bound image #{} to its memory allocation", i);
  }
  return true;
}

const BufferAllocation& HeapManager::declare_buf(
  size_t size, VkBufferUsageFlags usage,
  MemoryVisibility visibility) noexcept {
  return buf_allocs.emplace_back(BufferAllocation {
    ctxt, { size, usage, visibility },
    VK_NULL_HANDLE, nullptr, 0,
  });
}
const ImageAllocation& HeapManager::declare_img(
  const VkExtent2D& extent, std::optional<uint32_t> nlayer, VkFormat format,
  VkImageUsageFlags usage, VkImageTiling tiling,
  MemoryVisibility visibility) noexcept {
  return img_allocs.emplace_back(ImageAllocation {
    ctxt, { extent, std::move(nlayer), format, usage, tiling, visibility },
    VK_NULL_HANDLE, nullptr, 0,
  });
}

L_CUVK_END_
