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

std::vector<VkMemoryPropertyFlags> DUPLEX_FALLBACKS ={
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

const std::vector<VkMemoryAllocateFlags>& get_fallbacks(
  StorageOptimization opt) {
  switch (opt)
  {
  case cuvk::StorageOptimization::Send:
    return SEND_FALLBACKS;
  case cuvk::StorageOptimization::Fetch:
    return FETCH_FALLBACKS;
  case cuvk::StorageOptimization::Duplex:
    return DUPLEX_FALLBACKS;
  }
  std::terminate(); // Impossible branch.
}

//
// StorageMeasure --------------------------------------------------------------
//L

StorageMeasure::StorageMeasure(size_t size) noexcept :
  _fn([size]{ return size; }) {}
StorageMeasure::StorageMeasure(const std::function<size_t()> size_fn) noexcept:
  _fn(size_fn) {}

StorageMeasure::operator size_t() const noexcept {
  return _fn();
}

//
// Storage ---------------------------------------------------------------------
//L

Storage::Storage(L_STATIC const std::vector<VkMemoryPropertyFlags>& fallbacks) :

  _size(0),
  _dev_mem(VK_NULL_HANDLE),
  _props(0),
  _fallbacks(fallbacks),
  _mem_type_hint(0xFFFFFFFF),
  _cur_offset(0) {}

VkDeviceMemory Storage::dev_mem() const {
  return _dev_mem;
}
VkMemoryPropertyFlags Storage::props() const {
  return _props;
}
size_t Storage::size() const {
  return _size;
}

bool Storage::context_changing() {
  if (_dev_mem) {
    vkFreeMemory(ctxt().dev(), _dev_mem, nullptr);
    _dev_mem = VK_NULL_HANDLE;
  }
  _mem_type_hint = 0xFFFFFFFF;
  _cur_offset = 0;
  _props = 0;
  _deps.clear();

  return true;
}
bool Storage::context_changed() {
  for (const auto& mem_props : _fallbacks) {
    auto mem_type_idx = ctxt().find_mem_type(_mem_type_hint, mem_props);
    if (mem_type_idx < VK_MAX_MEMORY_TYPES) {
      _props = mem_props;
      break;
    }
  }
  if (!_props) {
    LOG.error("unable to find desired memory type");
    return false;
  }

  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = _size;
  auto idx = ctxt().find_mem_type(_mem_type_hint, _props);
  mai.memoryTypeIndex = idx;
  if (L_VK <- vkAllocateMemory(ctxt().dev(), &mai, nullptr, &_dev_mem)) {
    LOG.error("unable to allocate memory for storage");
    return false;
  }
  LOG.info("allocated {} bytes on heap {}", _size, ctxt().get_mem_heap_idx(idx));

  for (auto& dep : _deps) {
    switch (dep.type)
    {
    case Dependency::Type::Buffer:
      if (L_VK <- vkBindBufferMemory(ctxt().dev(), dep.buf,
        _dev_mem, dep.offset)) {
        LOG.error("unable to bind image to device memory");
        return false;
      }
      return true;
    case Dependency::Type::Image:
      if (L_VK <- vkBindImageMemory(ctxt().dev(), dep.img,
        _dev_mem, dep.offset)) {
        LOG.error("unable to bind image to device memory");
        return false;
      }
      return true;
    default:
      std::terminate();
    }
  }
}

Storage::Dependency::Dependency(VkBuffer buf, VkDeviceSize offset) :
  type(Type::Buffer),
  buf(buf),
  offset(offset) {}
Storage::Dependency::Dependency(VkImage img, VkDeviceSize offset) :
  type(Type::Image),
  img(img),
  offset(offset) {}
Storage::Dependency::~Dependency() {}

Storage::Dependency::Dependency(Dependency&& right) :
  type(right.type) {
  switch (type)
  {
  case cuvk::Storage::Dependency::Type::Buffer:
    buf = std::exchange(right.buf, nullptr);
  case cuvk::Storage::Dependency::Type::Image:
    img = std::exchange(right.img, nullptr);
  default:
    std::terminate();
  }
}


bool Storage::_declare_dependency(uint32_t hint, VkDeviceSize alloc_size) {
  _mem_type_hint &= hint;
  if (_mem_type_hint == 0) {
    LOG.error("no memory type can fulfill the requirements from all storage"
      " dependencies");
    return false;
  }
  _cur_offset += alloc_size;
  _size += alloc_size;
  return true;
}
bool Storage::declare_dependency(VkBuffer buf, uint32_t hint,
  VkDeviceSize alloc_size) {
  _deps.emplace_back(buf, _cur_offset);
  return _declare_dependency(hint, alloc_size);
}
bool Storage::declare_dependency(VkImage img, uint32_t hint,
  VkDeviceSize alloc_size) {
  _deps.emplace_back(img, _cur_offset);
  return _declare_dependency(hint, alloc_size);
}

//
// DeviceOnlyStorage -----------------------------------------------------------
//L

DeviceOnlyStorage::DeviceOnlyStorage() :
  Storage(L_STATIC DEVICE_ONLY_FALLBACKS) {}

//
// StagingStorage --------------------------------------------------------------
//L

StagingStorage::StagingStorage(StorageOptimization opt) :
  Storage(L_STATIC get_fallbacks(opt)) {}

bool StagingStorage::send(const void* src,
  StorageMeasure dst_offset, StorageMeasure size) {
  auto dev = ctxt().dev();
  auto dev_mem = this->dev_mem();
  void* dst;
  if (L_VK <- vkMapMemory(dev, dev_mem, dst_offset, size, 0, &dst)) {
    LOG.error("unable to map memory for sending");
    return false;
  }
  std::memcpy(dst, src, size);
  
  vkUnmapMemory(dev, dev_mem);

  // Flush memory after write, if the memory is not coherent.
  if (!(props() & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    VkMappedMemoryRange mmr {};
    mmr.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mmr.memory = dev_mem;
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
bool StagingStorage::fetch(L_OUT void* dst,
  StorageMeasure src_offset, StorageMeasure size) {
  auto dev = ctxt().dev();
  auto dev_mem = this->dev_mem();
  // Invalidate mapped memory before read if the memory is not coherent.
  if (!((props() & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
    VkMappedMemoryRange mmr {};
    mmr.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mmr.memory = dev_mem;
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
  if (L_VK <- vkMapMemory(dev, dev_mem, src_offset, size, 0, &src)) {
    LOG.error("unable to map memory for fetching");
    return false;
  }
  std::memcpy(dst, src, size);

  vkUnmapMemory(dev, dev_mem);
  return true;
}

//
// StorageBufferView -----------------------------------------------------------
//L

StorageBufferView::StorageBufferView(std::shared_ptr<StorageBuffer> buf,
  StorageMeasure offset, StorageMeasure size) :
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

VkBufferMemoryBarrier StorageBufferView::barrier(
  VkAccessFlags src, VkAccessFlags dst) const {
  VkBufferMemoryBarrier bmb {};
  bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bmb.buffer = buf();
  bmb.offset = _offset;
  bmb.size = _size;
  bmb.srcAccessMask = src;
  bmb.dstAccessMask = dst;
  return bmb;
}

//
// StorageBuffer ---------------------------------------------------------------
//L

StorageBufferView StorageBuffer::view() {
  return { shared_from_this(), 0, _size };
}
StorageBufferView StorageBuffer::view(
  StorageMeasure offset, StorageMeasure size) {
  return { shared_from_this(), offset, size };
}

StorageBuffer::StorageBuffer(StorageMeasure size, VkBufferUsageFlags usage) :
  _storage(nullptr),
  _size(size),
  _buf(VK_NULL_HANDLE),
  _usage(usage) {
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

  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(ctxt().dev(), _buf, &mr);
  if (_storage == nullptr) {
    LOG.error("buffer must bind with a storage to be used");
    return false;
  }
  if (!_storage->declare_dependency(_buf, mr.memoryTypeBits, mr.size)) {
    LOG.error("unable to declare dependency");
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

void StorageBuffer::bind(Storage& storage) {
  _storage = storage.shared_from_this();
}

//
// StorageImageView ------------------------------------------------------------
//L

StorageImageView::StorageImageView(std::shared_ptr<StorageImage> img,
  const VkOffset2D& offset, const VkExtent2D& extent) :
  _img(img),
  _img_view(VK_NULL_HANDLE),
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
  // TODO: (penguinliong) Make it adjustable later.
  ivci.viewType = _img->nlayer() ?
    VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
  ivci.format = _img->format();
  ivci.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  ivci.subresourceRange.levelCount = 1;
  ivci.subresourceRange.layerCount = _img->nlayer().value_or(1);

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
std::optional<uint32_t> StorageImageView::nlayer() const {
  return _img->nlayer();
}
VkImageMemoryBarrier StorageImageView::barrier(
  VkAccessFlags src, VkAccessFlags dst,
  VkImageLayout srcLayout, VkImageLayout dstLayout) const {
  VkImageMemoryBarrier imb {};
  imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imb.image = img();
  imb.oldLayout = srcLayout;
  imb.newLayout = dstLayout;
  imb.srcAccessMask = src;
  imb.dstAccessMask = dst;
  imb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imb.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
  imb.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
  return imb;
}

VkBufferImageCopy StorageImageView::copy_with_buffer(
  const StorageBufferView& buf) const {
  auto layer = nlayer().value_or(1);
  VkBufferImageCopy bic {};
  bic.bufferOffset = buf.offset();
  bic.bufferRowLength = _extent.width;
  bic.bufferImageHeight = _extent.height;
  bic.imageExtent = { _extent.width, _extent.height, 1 };
  bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  bic.imageSubresource.layerCount = layer;
  return bic;
}

//
// StorageImage ----------------------------------------------------------------
//L

StorageImage::StorageImage(
  const VkExtent2D& extent, std::optional<uint32_t> nlayer,
  VkFormat format, VkImageUsageFlags usage, VkImageTiling tiling) :
  _img(VK_NULL_HANDLE),
  _storage(nullptr),
  _extent(extent),
  _nlayer(nlayer),
  _format(format),
  _usage(usage),
  _tiling(tiling) {
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
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = _format;
  ici.extent.width = _extent.width;
  ici.extent.height = _extent.height;
  ici.extent.depth = 1;
  ici.mipLevels = 1;
  ici.arrayLayers = _nlayer.value_or(1);
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = _tiling;
  ici.usage = _usage;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  if (L_VK <- vkCreateImage(ctxt().dev(), &ici, nullptr, &_img)) {
    LOG.error("unable to create image");
    return false;
  }
  
  VkMemoryRequirements mr {};
  vkGetImageMemoryRequirements(ctxt().dev(), _img, &mr);
  if (_storage == nullptr) {
    LOG.error("image must bind with a storage to be used");
    return false;
  }
  if (!_storage->declare_dependency(_img, mr.memoryTypeBits, mr.size)) {
    LOG.error("unable to declare dependency");
    return false;
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
std::optional<uint32_t> StorageImage::nlayer() const {
  return _nlayer;
}
size_t StorageImage::size() const {
  // TODO: (penguinliong) Adapt to other color formats.
  return _extent.width * _extent.height * _nlayer.value_or(1) * sizeof(float);
}
VkFormat StorageImage::format() const {
  return _format;
}

void StorageImage::bind(Storage& storage) {
  _storage = storage.shared_from_this();
}
std::shared_ptr<StorageImageView> StorageImage::view() {
  return ctxt().make_contextual<StorageImageView>(
    shared_from_this(), VkOffset2D{ 0, 0 }, _extent);
}
std::shared_ptr<StorageImageView> StorageImage::view(const VkOffset2D& offset,
  const VkExtent2D& extent) {
  return ctxt().make_contextual<StorageImageView>(
    shared_from_this(), offset, extent);
}

// StagingStorageImage ---------------------------------------------------------

StagingStorageImage::StagingStorageImage(
  const VkExtent2D& extent, std::optional<uint32_t> nlayer, VkFormat format) :
  StorageImage(extent, nlayer, format,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    VK_IMAGE_TILING_LINEAR) {}

StorageImageView StagingStorageImage::view(
  const VkOffset2D& offset, const VkExtent2D& extent) {
  LOG.error("cannot create view for storage image");
  std::terminate();
}

// ColorAttachmentStorageImage -------------------------------------------------

ColorAttachmentStorageImage::ColorAttachmentStorageImage(
  const VkExtent2D& extent, std::optional<uint32_t> nlayer, VkFormat format) :
  StorageImage(extent, nlayer, format,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    VK_IMAGE_TILING_OPTIMAL) {}

// UniformStorageImage ---------------------------------------------------------

UniformStorageImage::UniformStorageImage(
  const VkExtent2D& extent, std::optional<uint32_t> nlayer, VkFormat format) :
  StorageImage(extent, nlayer, format,
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VK_IMAGE_TILING_OPTIMAL) {}

























VkBuffer create_buf(VkDevice dev, const BufferAllocationRequirements& req) {
  VkBufferCreateInfo bci {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = req.size;
  bci.usage = req.usage;

  VkBuffer buf;
  return (L_VK <- vkCreateBuffer(dev, &bci, nullptr, &buf)) ?
    VK_NULL_HANDLE : buf;
}
VkImage create_img(VkDevice dev, const ImageAllocationRequirements& req) {
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

  VkImage img;
  if (L_VK <- vkCreateImage(dev, &ici, nullptr, &img)) {
    return false;
  }
}

constexpr VkDeviceSize align_size(VkDeviceSize size, VkDeviceSize alignment) {
  return (size + alignment - 1) / alignment * alignment;
}

bool HeapManager::create_rscs() {
  auto dev = ctxt().dev();

  // Create buffers.
  for (auto& buf_alloc : _buf_allocs) {
    auto buf = create_buf(dev, buf_alloc.req);
    if (buf == VK_NULL_HANDLE) {
      return false;
    }
    buf_alloc.alloc->buf = buf;

    // Check memory requirements.
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(dev, buf, &mem_req);
    
    auto mem_type_idx = ctxt().find_mem_type(mem_req.memoryTypeBits,
      get_fallbacks(buf_alloc.req.opt));
    buf_alloc.alloc->mem_type_idx = mem_type_idx;

    auto mem_heap_idx = ctxt().get_mem_heap_idx(mem_type_idx);
    auto& alloc_size = _heap_allocs[mem_heap_idx].alloc_size;
    auto offset_aligned = align_size(alloc_size, mem_req.alignment);
    buf_alloc.alloc->offset = offset_aligned;
    alloc_size = offset_aligned + mem_req.size;
  }

  // Create images.
  for (auto& img_alloc : _img_allocs) {
    auto img = create_img(dev, img_alloc.req);
    if (img == VK_NULL_HANDLE) {
      return false;
    }
    img_alloc.alloc->img = img;

    // Check memory requirements.
    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(dev, img, &mem_req);

    auto mem_type_idx = ctxt().find_mem_type(mem_req.memoryTypeBits,
      get_fallbacks(img_alloc.req.opt));
    img_alloc.alloc->mem_type_idx = mem_type_idx;

    auto mem_heap_idx = ctxt().get_mem_heap_idx(mem_type_idx);
    auto& alloc_size = _heap_allocs[mem_heap_idx].alloc_size;
    auto offset_aligned = align_size(alloc_size, mem_req.alignment);
    img_alloc.alloc->offset = offset_aligned;
    alloc_size = offset_aligned + mem_req.size;
  }
  return true;
}

bool HeapManager::alloc_mem() {
  auto dev = ctxt().dev();

  // Allocate memory for each type.
  for (auto i = 0; i < VK_MAX_MEMORY_TYPES; ++i) {
    auto& heap_alloc = _heap_allocs[i];
    if (heap_alloc.alloc_size > 0) {
      VkMemoryAllocateInfo mai {};
      mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      mai.allocationSize = heap_alloc.alloc_size;
      mai.memoryTypeIndex = i;

      VkDeviceMemory dev_mem;
      if (L_VK <- vkAllocateMemory(dev, &mai, nullptr, &dev_mem)) {
        return false;
      }
      heap_alloc.dev_mem = dev_mem;
    }
  }
  return true;
}

bool HeapManager::bind_rscs() {
  auto dev = ctxt().dev();

  for (auto& buf_alloc : _buf_allocs) {
    auto alloc = *buf_alloc.alloc;
    auto dev_mem = _heap_allocs[alloc.mem_type_idx].dev_mem;
    if (L_VK <- vkBindBufferMemory(dev, alloc.buf, dev_mem, alloc.offset)) {
      return false;
    }
  }
  for (auto& img_alloc : _img_allocs) {
    auto alloc = *img_alloc.alloc;
    auto dev_mem = _heap_allocs[alloc.mem_type_idx].dev_mem;
    if (L_VK <- vkBindImageMemory(dev, alloc.img, dev_mem, alloc.offset)) {
      return false;
    }
  }
  return true;
}

bool HeapManager::context_changing() {
  auto dev = ctxt().dev();
  for (auto& buf_alloc : _buf_allocs) {
    vkDestroyBuffer(dev, buf_alloc.alloc->buf, nullptr);
    buf_alloc = {};
  }
  for (auto& img_alloc : _img_allocs) {
    vkDestroyImage(dev, img_alloc.alloc->img, nullptr);
    img_alloc = {};
  }
  for (auto& heap_alloc : _heap_allocs) {
    vkFreeMemory(dev, heap_alloc.dev_mem, nullptr);
    heap_alloc = {};
  }
  return true;
}
bool HeapManager::context_changed() {
  auto dev = ctxt().dev();
  std::vector<BufAlloc*> delayed_allocs;

  if (!create_rscs()) {
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

BufferAllocation HeapManager::declare_buf(size_t size, VkBufferUsageFlags usage,
  StorageOptimization opt) {
  auto alloc = std::make_shared<BufferAllocationInfo>();
  BufAlloc buf_alloc {};
  buf_alloc.alloc = alloc;
  buf_alloc.req.size = size;
  buf_alloc.req.usage = usage;
  buf_alloc.req.opt = opt;
  _buf_allocs.emplace_back(std::move(buf_alloc));
  return alloc;
}
ImageAllocation HeapManager::declare_img(const VkExtent2D& extent,
    std::optional<uint32_t> nlayer, VkFormat format, VkImageUsageFlags usage,
    VkImageTiling tiling, StorageOptimization opt) {
  auto alloc = std::make_shared<ImageAllocationInfo>();
  ImgAlloc img_alloc {};
  img_alloc.alloc = alloc;
  img_alloc.req.extent = extent;
  img_alloc.req.nlayer = std::move(nlayer);
  img_alloc.req.format = format;
  img_alloc.req.usage = usage;
  img_alloc.req.tiling = tiling;
  img_alloc.req.opt = opt;
  _img_allocs.emplace_back(std::move(img_alloc));
  return alloc;
}

L_CUVK_END_
