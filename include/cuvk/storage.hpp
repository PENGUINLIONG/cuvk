#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/context.hpp"
#include <vector>
#include <cstddef>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_

// Represent a huge lump of data.
class Storage : public Contextual {
  friend class MappedStorage;
private:
  size_t _size;

public:
  Storage(size_t size);
  ~Storage();

  size_t size() const;
};

class DeviceStorage : public Storage {
protected:
  VkDeviceMemory _dev_mem;
  VkMemoryPropertyFlags _flags;

public:
  DeviceStorage(size_t size, VkMemoryPropertyFlags flags);

  bool context_changing() override;
  bool context_changed() override;
};

// Create a buffer optimal for persistent storage in the devices. Notice that
// such storage cannot be mapped. Caching data requires the use of a send
// buffer, because the cache is only visible to the device, so cache storages
// should be generated first.
//
// TODO: (penguinliong) DeviceCache is WIP.
class DeviceCache : public DeviceStorage {
public:
  DeviceCache(size_t size);

  bool cache(const void* data, size_t dst_offset, size_t size);
};

// Create a buffer optimal for sending data from the host to the devices.
class SendBuffer : public DeviceStorage {
public:
  SendBuffer(size_t size);

  bool send(const void* data, size_t dst_offset, size_t size);
};
// Create a buffer optimal for fetching data from the devices to the host.
class FetchBuffer : public DeviceStorage {
public:
  FetchBuffer(size_t size);

  bool fetch(L_OUT void* data, size_t src_offset, size_t size);
};

L_CUVK_END_
