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

class StorageBuffer;
class StorageImage;

enum class StorageOptimization {
  Send, Fetch, Duplex
};

// Represent a huge lump of data on device.
class Storage : public Contextual,
  public std::enable_shared_from_this<Storage> {
private:
  struct Dependency {
    Dependency(VkBuffer buf, VkDeviceSize offset);
    Dependency(VkImage img, VkDeviceSize offset);
    Dependency(Dependency&&);
    ~Dependency() noexcept;
    enum class Type { Buffer, Image } type;
    VkDeviceSize offset;
    union {
      VkBuffer buf;
      VkImage img;
    };
  };

  size_t _size;
  VkDeviceMemory _dev_mem;
  VkMemoryPropertyFlags _props;
  const std::vector<VkMemoryPropertyFlags>& _fallbacks;
  std::vector<Dependency> _deps;

  uint32_t _mem_type_hint;
  // The offset that will .
  VkDeviceSize _cur_offset;

  bool _declare_dependency(uint32_t hint, VkDeviceSize alloc_size);

protected:
  Storage(L_STATIC const std::vector<VkMemoryPropertyFlags>& fallbacks);

public:
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  VkDeviceMemory dev_mem() const;
  VkMemoryPropertyFlags props() const;
  size_t size() const;

  bool context_changing() override;
  bool context_changed() override;

  // For use by `StorageBuffer` and `StorageImage` to declare which memory type
  // should be used, how much size they want to take (size of driver metadata
  // included), and the offset to the memory will be allocated for them is
  // returned through reference.
  bool declare_dependency(VkBuffer buf, uint32_t hint, VkDeviceSize alloc_size);
  bool declare_dependency(VkImage img, uint32_t hint, VkDeviceSize alloc_size);
};

class DeviceOnlyStorage : public Storage {
public:
  DeviceOnlyStorage();
};

// Storage that is visible to the host, used to transfer data between the host
// and the device.
class StagingStorage : public Storage {
public:
  StagingStorage(StorageOptimization opt);
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

  VkBufferMemoryBarrier barrier(VkAccessFlags src, VkAccessFlags dst) const;
};

// Wrapped memory chunk for structured data.
class StorageBuffer : public Contextual,
  public std::enable_shared_from_this<StorageBuffer> {
private:
  std::shared_ptr<Storage> _storage;
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

  void bind(Storage& storage);
};



class StorageImage;

class StorageImageView : public Contextual {
  friend class StorageImage;
private:
  std::shared_ptr<const StorageImage> _img;
  VkImageView _img_view;
  VkOffset2D _offset;
  VkExtent2D _extent;

public:
  StorageImageView(std::shared_ptr<StorageImage> img,
    const VkOffset2D& offset, const VkExtent2D& extent);

  bool context_changing() override;
  bool context_changed() override;

  VkImage img() const;
  VkImageView img_view() const;
  const VkOffset2D& offset() const;
  const VkExtent2D& extent() const;
  std::optional<uint32_t> nlayer() const;

  VkImageMemoryBarrier barrier(
    VkAccessFlags src, VkAccessFlags dst,
    VkImageLayout srcLayout, VkImageLayout dstLayout) const;
  VkBufferImageCopy copy_with_buffer(const StorageBufferView& buf) const;
};

class StorageImage : public Contextual,
  public std::enable_shared_from_this<StorageImage> {
private:
  VkImage _img;

  std::shared_ptr<Storage> _storage;

  VkExtent2D _extent;
  VkFormat _format;
  VkImageUsageFlags _usage;
  VkImageTiling _tiling;
  std::optional<uint32_t> _nlayer;

protected:
  StorageImage(
    const VkExtent2D& extent, std::optional<uint32_t> nlayer,
    VkFormat format, VkImageUsageFlags usage, VkImageTiling tiling);

public:
  bool context_changing() override;
  bool context_changed() override;

  Storage& storage();
  const Storage& storage() const;
  const VkExtent2D& extent() const;
  VkImage img() const;
  std::optional<uint32_t> nlayer() const;
  size_t size() const;
  VkFormat format() const;

  void bind(Storage& storage);

  std::shared_ptr<StorageImageView> view();
  std::shared_ptr<StorageImageView> view(const VkOffset2D& offset,
    const VkExtent2D& extent);
};

class StagingStorageImage : public StorageImage {
private:
  StorageImageView view(const VkOffset2D& offset, const VkExtent2D& extent);

public:
  StagingStorageImage(
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





















struct BufferAllocationRequirements {
  VkDeviceSize size;
  VkBufferUsageFlags usage;
  StorageOptimization opt;
};
struct BufferAllocationInfo {
  uint32_t mem_type_idx;
  VkDeviceSize offset;
  VkBuffer buf;
};
using BufferAllocation = std::shared_ptr<BufferAllocationInfo>;



struct ImageAllocationRequirements {
  VkExtent2D extent;
  VkImageUsageFlags usage;
  std::optional<uint32_t> nlayer;
  VkFormat format;
  VkImageTiling tiling;
  StorageOptimization opt;
};
struct ImageAllocationInfo {
  uint32_t mem_type_idx;
  VkDeviceSize offset;
  VkImage img;
};
using ImageAllocation = std::shared_ptr<ImageAllocationInfo>;



class HeapManager : public Contextual {
private:
  bool create_rscs();
  bool alloc_mem();
  bool bind_rscs();

public:
  struct BufAlloc {
    BufferAllocation alloc;
    BufferAllocationRequirements req;
  };
  struct ImgAlloc {
    ImageAllocation alloc;
    ImageAllocationRequirements req;
  };
  struct HeapAlloc {
    VkDeviceSize alloc_size;
    VkDeviceMemory dev_mem;
  };

  std::array<HeapAlloc, VK_MAX_MEMORY_TYPES> _heap_allocs;
  std::vector<BufAlloc> _buf_allocs;
  std::vector<ImgAlloc> _img_allocs;

  bool context_changing() override;
  bool context_changed() override;

  BufferAllocation declare_buf(size_t size, VkBufferUsageFlags usage,
    StorageOptimization opt);
  ImageAllocation declare_img(const VkExtent2D& extent,
    std::optional<uint32_t> nlayer, VkFormat format, VkImageUsageFlags usage,
    VkImageTiling tiling, StorageOptimization opt);
};


L_CUVK_END_
