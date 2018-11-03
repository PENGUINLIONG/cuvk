#include "cuvk/storage.hpp"
#include "cuvk/logger.hpp"

L_CUVK_BEGIN_

Storage::Storage(size_t size) : _size(size) { }
Storage::~Storage() { }

size_t Storage::size() const {
  return _size;
}



DeviceStorage::DeviceStorage(size_t size, VkMemoryPropertyFlags flags) :
  Storage(size),
  _dev_mem(VK_NULL_HANDLE),
  _flags(flags) { }

bool DeviceStorage::context_changing() {
  vkFreeMemory(ctxt().dev(), _dev_mem, nullptr);
  return true;
}
bool DeviceStorage::context_changed() {
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = size();
  mai.memoryTypeIndex = ctxt().find_mem_type(_flags);
  if (L_VK <- vkAllocateMemory(ctxt().dev(), &mai, nullptr, &_dev_mem)) {
    return false;
  }
  return true;
}



DeviceCache::DeviceCache(size_t size) :
  DeviceStorage(size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) { }

bool DeviceCache::cache(const void* data, size_t dst_offset, size_t size) {
  throw std::logic_error("not implemented yet");
}



SendBuffer::SendBuffer(size_t size) :
  DeviceStorage(size, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) { }

bool SendBuffer::send(const void* data, size_t dst_offset, size_t size) {
  void* dst;
  if (L_VK <- vkMapMemory(ctxt().dev(), _dev_mem, dst_offset, size, 0, &dst)) {
    return false;
  }
  std::memcpy(dst, data, size);
  
  vkUnmapMemory(ctxt().dev(), _dev_mem);
  return true;
}



FetchBuffer::FetchBuffer(size_t size) :
  DeviceStorage(size, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                      VK_MEMORY_PROPERTY_HOST_CACHED_BIT) { }

bool FetchBuffer::fetch(L_OUT void* data, size_t src_offset, size_t size) {
  void* src;
  if (L_VK <- vkMapMemory(ctxt().dev(), _dev_mem, src_offset, size, 0, &src)) {
    return false;
  }
  std::memcpy(data, src, size);
}

L_CUVK_END_
