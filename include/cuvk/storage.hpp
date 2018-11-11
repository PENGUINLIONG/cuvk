#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/context.hpp"
#include <vector>
#include <functional>
#include <optional>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_

// Represent a lazy evaluated allocation size or offset. Measure has the same
// meaning as that in Mathematics.
struct StorageMeasure {
private:
  std::function<size_t()> _fn;

public:
  StorageMeasure(size_t size) noexcept;
  StorageMeasure(const std::function<size_t()> size_fn) noexcept;

  operator size_t() const noexcept;
  template<typename _ = std::enable_if_t<!std::is_same_v<size_t, VkDeviceSize>>>
  operator VkDeviceSize() const noexcept {
    return static_cast<VkDeviceSize>(_fn());
  }
};


enum class StorageOptimization {
  Send, Fetch, Duplex
};

// Represent a huge lump of data on device.
class Storage : public Contextual,
  public std::enable_shared_from_this<Storage> {
private:
  StorageMeasure _size;
  VkDeviceMemory _dev_mem;
  VkMemoryPropertyFlags _props;
  const std::vector<VkMemoryPropertyFlags>& _fallbacks;

protected:
  Storage(StorageMeasure size,
    L_STATIC const std::vector<VkMemoryPropertyFlags>& fallbacks);

public:
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  VkDeviceMemory dev_mem() const;
  VkMemoryPropertyFlags props() const;
  size_t size() const;

  bool context_changing() override;
  bool context_changed() override;
};

class DeviceOnlyStorage : public Storage {
public:
  DeviceOnlyStorage(StorageMeasure size);
};

// Storage that is visible to the host, used to transfer data between the host
// and the device.
class StagingStorage : public Storage {
public:
  StagingStorage(StorageMeasure size, StorageOptimization opt);
  StagingStorage(const StagingStorage&) = delete;
  
  bool send(const void* src, StorageMeasure dst_offset, StorageMeasure size);
  bool fetch(L_OUT void* dst, StorageMeasure src_offset, StorageMeasure size);
};



class StorageBuffer;

class StorageBufferView {
  friend class StorageBuffer;
private:
  std::shared_ptr<StorageBuffer> _buf;
  StorageMeasure _offset;
  StorageMeasure _size;

protected:
  StorageBufferView(std::shared_ptr<StorageBuffer> buf,
    StorageMeasure offset, StorageMeasure size);

public:
  VkBuffer buf() const;
  size_t offset() const;
  size_t size() const;
};

// Wrapped memory chunk for structured data.
class StorageBuffer : public Contextual,
  public std::enable_shared_from_this<StorageBuffer> {
private:
  std::shared_ptr<Storage> _storage;
  StorageMeasure _offset;
  StorageMeasure _size;
  VkBuffer _buf;
  VkBufferUsageFlags _usage;

public:
  StorageBuffer(StorageMeasure size, VkBufferUsageFlags usage);
  
  StorageBuffer(const StorageBuffer&) = delete;
  StorageBuffer& operator=(const StorageBuffer&) = delete;

  bool context_changing() override;
  bool context_changed() override;

  Storage& storage();
  const Storage& storage() const;
  size_t size() const;
  VkBuffer buf() const;
  
  StorageBufferView view();
  StorageBufferView view(
    StorageMeasure offset, StorageMeasure size);

  bool bind(Storage& storage, StorageMeasure offset = 0);

  // There is extra size for metadata required by the driver. So the resultant
  // memory allocation should be `size + meta_size`, which is slightly larger
  // than that we required via the constructor.
  size_t alloc_size() const;
};




class StorageImage;

class StorageImageView : public Contextual {
  friend class StorageImage;
private:
  std::shared_ptr<const StorageImage> _img;
  VkImageView _img_view;
  VkOffset2D _offset;
  VkExtent2D _extent;

protected:
  StorageImageView(std::shared_ptr<StorageImage> img,
    const VkOffset2D& offset, const VkExtent2D& extent);

public:
  bool context_changing() override;
  bool context_changed() override;

  VkImage img() const;
  VkImageView img_view() const;
  const VkOffset2D& offset() const;
  const VkExtent2D& extent() const;
  std::optional<uint32_t> nlayer() const;
  VkImageLayout preferred_layout() const;
};

class StorageImage : public Contextual,
  public std::enable_shared_from_this<StorageImage> {
private:
  VkImage _img;

  std::shared_ptr<Storage> _storage;
  StorageMeasure _offset;

  VkExtent2D _extent;
  VkFormat _format;
  VkImageUsageFlags _usage;
  VkImageLayout _layout;
  VkImageTiling _tiling;
  std::optional<uint32_t> _nlayer;

protected:
  StorageImage(
    const VkExtent2D& extent, std::optional<uint32_t> nlayer,
    VkFormat format, VkImageUsageFlags usage,
    VkImageLayout layout, VkImageTiling tiling);

public:
  bool context_changing() override;
  bool context_changed() override;

  Storage& storage();
  const Storage& storage() const;
  const VkExtent2D& extent() const;
  VkImage img() const;
  VkImageLayout preferred_layout() const;
  std::optional<uint32_t> nlayer() const;
  size_t size() const;

  size_t alloc_size() const;

  bool bind(Storage& storage, size_t offset);

  StorageImageView view(const VkOffset2D& offset,
    const VkExtent2D& extent);
};

class GeneralStorageImage : public StorageImage {
private:
  StorageImageView view(const VkOffset2D& offset, const VkExtent2D& extent);

public:
  GeneralStorageImage(
    const VkExtent2D& extent, std::optional<uint32_t> nlayer, VkFormat format);
};

class ColorAttachmentStorageImage : public StorageImage {
public:
  ColorAttachmentStorageImage(
    const VkExtent2D& extent, std::optional<uint32_t> nlayer, VkFormat format);
};

class UniformStorageImage : public StorageImage {
public:
  UniformStorageImage(
    const VkExtent2D& extent, std::optional<uint32_t> nlayer, VkFormat format);
};

L_CUVK_END_
