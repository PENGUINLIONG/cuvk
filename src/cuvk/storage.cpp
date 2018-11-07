#include "cuvk/storage.hpp"
#include "cuvk/logger.hpp"
#include <exception>

L_CUVK_BEGIN_

std::vector<VkMemoryPropertyFlags> DEVICE_ONLY_FALLBACKS ={
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
};

std::vector<VkMemoryPropertyFlags> SEND_FALLBACKS ={
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT,

  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
};

std::vector<VkMemoryPropertyFlags> FETCH_FALLBACKS ={
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT,

  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT,

  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
};

const std::vector<VkMemoryAllocateFlags>& get_fallbacks(StorageOptimization opt) {
  switch (opt)
  {
  case cuvk::StorageOptimization::Send:
    return SEND_FALLBACKS;
  case cuvk::StorageOptimization::Fetch:
    return FETCH_FALLBACKS;
  case cuvk::StorageOptimization::DeviceOnly:
    return DEVICE_ONLY_FALLBACKS;
  }
  std::terminate(); // Impossible branch.
}



Storage::Storage(size_t size, StorageOptimization opt) :
  _size(size),
  _opt(opt),
  _dev_mem(VK_NULL_HANDLE),
  _flags(0) {
}
Storage::~Storage() { }

VkDeviceMemory Storage::dev_mem() const {
  return _dev_mem;
}
size_t Storage::size() const {
  return _size;
}

bool Storage::context_changing() {
  if (_dev_mem) {
    vkFreeMemory(ctxt().dev(), _dev_mem, nullptr);
    _dev_mem = VK_NULL_HANDLE;
  }
  _flags = 0;

  return true;
}
bool Storage::context_changed() {
  for (const auto& mem_props : get_fallbacks(_opt)) {
    auto mem_type_idx = ctxt().find_mem_type(mem_props);
    if (mem_type_idx < VK_MAX_MEMORY_TYPES) {
      _flags = mem_props;
      break;
    }
  }
  if (!_flags) {
    LOG.error("unable to find desired memory type");
    return false;
  }

  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = size();
  mai.memoryTypeIndex = ctxt().find_mem_type(_flags);
  if (L_VK <- vkAllocateMemory(ctxt().dev(), &mai, nullptr, &_dev_mem)) {
    LOG.error("unable to allocate memory for storage");
    return false;
  }
  return true;
}

bool Storage::send(const void* data, size_t dst_offset, size_t size) {
  auto dev = ctxt().dev();
  void* dst;
  if (L_VK <- vkMapMemory(dev, _dev_mem, dst_offset, size, 0, &dst)) {
    LOG.error("unable to map memory for sending");
    return false;
  }
  std::memcpy(dst, data, size);
  
  vkUnmapMemory(dev, _dev_mem);

  // Flush memory after write, if the memory is not coherent.
  if (!(_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    VkMappedMemoryRange mmr {};
    mmr.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mmr.memory = _dev_mem;
    // FIXME: (penguinliong) The offset also follow the same rule as size.
    mmr.offset = dst_offset;
    mmr.size = size;
    if (L_VK <- vkFlushMappedMemoryRanges(dev, 1, &mmr)) {
      LOG.error("unable to flush data to device");
      return false;
    }
  }
  return true;
}
bool Storage::fetch(L_OUT void* data, size_t src_offset, size_t size) {
  auto dev = ctxt().dev();
  // Invalidate mapped memory before read if the memory is not coherent.
  if ((_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    VkMappedMemoryRange mmr {};
    mmr.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mmr.memory = _dev_mem;
    mmr.offset = src_offset;
    if (size < ctxt().limits().nonCoherentAtomSize) {
      mmr.size = VK_WHOLE_SIZE;
    } else {
      mmr.size = size;
    }
    if (L_VK <- vkInvalidateMappedMemoryRanges(dev, 1, &mmr)) {
      LOG.error("unable to update mapped memory");
      return false;
    }
  }

  void* src;
  if (L_VK <- vkMapMemory(dev, _dev_mem, src_offset, size, 0, &src)) {
    LOG.error("unable to map memory for fetching");
    return false;
  }
  std::memcpy(data, src, size);

  vkUnmapMemory(dev, _dev_mem);
  return true;
}

StorageBufferView::StorageBufferView(std::shared_ptr<const StorageBuffer> buf,
  size_t offset, size_t size) :
  _buf(buf),
  _offset(offset),
  _size(size) {
}
VkBuffer StorageBufferView::buf() const {
  return _buf->buf();
}
size_t StorageBufferView::offset() const {
  return _offset;
}
size_t StorageBufferView::size() const {
  return _size;
}


StorageBuffer::StorageBuffer(size_t size,
  ExecType ty, VkBufferUsageFlags usage) :
  _storage(VK_NULL_HANDLE),
  _offset(0),
  _size(size),
  _buf(VK_NULL_HANDLE),
  _ty(ty),
  _usage(usage) {
}
StorageBuffer::~StorageBuffer() {
  context_changing();
}

Storage& StorageBuffer::storage() {
  return *_storage;
}
const Storage& StorageBuffer::storage() const {
  return *_storage;
}

bool StorageBuffer::context_changing() {
  if (_buf) {
    vkDestroyBuffer(ctxt().dev(), _buf, nullptr);
    _buf = VK_NULL_HANDLE;
  }
  return true;
}
bool StorageBuffer::context_changed() {
  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = _size;
  bci.usage = _usage;
  // We need to collect all the bacteria transform results before evaluation.
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (L_VK <- vkCreateBuffer(ctxt().dev(), &bci, nullptr, &_buf)) {
    LOG.error("unable to create buffer");
    return false;
  }
  return true;
}

size_t StorageBuffer::size() const {
  return _size;
}
VkBuffer StorageBuffer::buf() const {
  return _buf;
}

StorageBufferView StorageBuffer::view() const {
  return StorageBufferView(shared_from_this(),
    0,
    static_cast<uint32_t>(size()));
}
StorageBufferView StorageBuffer::view(size_t offset, size_t size) const {
  return StorageBufferView(shared_from_this(),
    static_cast<uint32_t>(offset),
    static_cast<uint32_t>(size));
}

size_t StorageBuffer::alloc_size() const {
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(ctxt().dev(), _buf, &mr);
  return mr.size + _size;
}

bool StorageBuffer::bind(std::shared_ptr<Storage> storage, size_t offset) {
  if (storage->size() < alloc_size()) {
    LOG.error("storage bound is too small to contain this buffer");
    return false;
  }
  if (L_VK <- vkBindBufferMemory(
    ctxt().dev(), _buf, storage->dev_mem(), offset)) {
    LOG.error("unable to bind buffer to device memory");
    return false;
  }
  _storage = storage;
  _offset = offset;
  return true;
}



StorageImageView::StorageImageView(std::shared_ptr<const StorageImage> img) :
  _img(img),
  _offset{ 0, 0 },
  _extent(img->extent()) {}
StorageImageView::StorageImageView(std::shared_ptr<const StorageImage> img,
  VkOffset2D offset, VkExtent2D extent) :
  _img(img),
  _offset(offset),
  _extent(extent) {}

bool StorageImageView::context_changing() {
  if (_img_view) {
    vkDestroyImageView(ctxt().dev(), _img_view, nullptr);
    _img_view = VK_NULL_HANDLE;
  }
  return true;
}
bool StorageImageView::context_changed() {
  VkImageViewCreateInfo ivci {};
  ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ivci.image = _img->img();
  // FIXME: (penguinliong) Make it adjustable later.
  ivci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  ivci.format = VK_FORMAT_R32_SFLOAT;
  ivci.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  ivci.subresourceRange.layerCount = _img->nlayer();

  if (L_VK <- vkCreateImageView(ctxt().dev(), &ivci, nullptr, &_img_view)) {
    LOG.error("unable to create image view");
    return false;
  }
  return true;
}

VkImage StorageImageView::img() const {
  return _img->img();
}
VkImageView StorageImageView::img_view() const {
  return _img_view;
}
const VkOffset2D& StorageImageView::offset() const {
  return _offset;
}
const VkExtent2D& StorageImageView::extent() const {
  return _extent;
}
uint32_t StorageImageView::nlayer() const {
  return _img->nlayer();
}
VkImageLayout StorageImageView::layout() const {
  return _img->layout();
}

StorageImage::StorageImage(const VkExtent2D& extent, uint32_t nlayer,
  ExecType exec_ty, StorageOptimization opt,
  VkImageType img_ty, VkFormat format,
  VkImageUsageFlags usage, VkImageLayout layout,
  VkImageTiling tiling) :
  _img(VK_NULL_HANDLE),
  _storage(nullptr),
  _offset(0),
  _extent(extent),
  _nlayer(nlayer),
  _exec_ty(exec_ty),
  _opt(opt),
  _img_ty(img_ty),
  _format(format),
  _usage(usage),
  _layout(layout),
  _tiling(tiling){
}
StorageImage::~StorageImage() {
  context_changing();
}

bool StorageImage::context_changing() {
  if (_img) {
    vkDestroyImage(ctxt().dev(), _img, nullptr);
    _img = VK_NULL_HANDLE;
  }
  return true;
}
bool StorageImage::context_changed() {
  VkImageCreateInfo ici {};
  ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ici.imageType = _img_ty;
  ici.format = _format;
  ici.extent.width = _extent.width;
  ici.extent.height = _extent.height;
  ici.extent.depth = 1;
  ici.mipLevels = 1;
  ici.arrayLayers = _nlayer;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = _tiling;
  ici.usage = _usage;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  if (L_VK <- vkCreateImage(ctxt().dev(), &ici, nullptr, &_img)) {
    LOG.error("unable to create image");
    return false;
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  if (!(_usage & (
    VK_IMAGE_USAGE_SAMPLED_BIT |
    VK_IMAGE_USAGE_STORAGE_BIT |
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
    VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT))) {
    return true;
  }

  return true;
}

Storage& StorageImage::storage() {
  return *_storage;
}
const Storage& StorageImage::storage() const {
  return *_storage;
}
const VkExtent2D& StorageImage::extent() const{
  return _extent;
}
VkImage StorageImage::img() const {
  return _img;
}
VkImageLayout StorageImage::layout() const {
  return _layout;
}
uint32_t StorageImage::nlayer() const {
  return _nlayer;
}
size_t StorageImage::size() const {
  // TODO: (penguinliong) Adapt to other color formats.
  return _extent.width * _extent.height * _nlayer * sizeof(float);
}

size_t StorageImage::alloc_size() const {
  // FIXME: (penguinliong) Make it more generalized.
  VkMemoryRequirements mr {};
  vkGetImageMemoryRequirements(ctxt().dev(), _img, &mr);
  return mr.size;
}

bool StorageImage::bind(std::shared_ptr<Storage> storage, size_t offset) {
  if (L_VK <- vkBindImageMemory(ctxt().dev(), _img,
    storage->dev_mem(), offset)) {
    LOG.error("unable to bind image to device memory");
    return false;
  }
  _storage = storage;
  _offset = offset;
}

L_CUVK_END_
