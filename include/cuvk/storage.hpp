#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/context.hpp"
#include <vector>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_

enum class StorageOptimization {
  Send, Fetch, DeviceOnly
};

// Represent a huge lump of data on device.
class Storage : public Contextual {
private:
  size_t _size;
  StorageOptimization _opt;
  VkDeviceMemory _dev_mem;
  VkMemoryPropertyFlags _flags;

public:
  Storage(size_t size, StorageOptimization opt);
  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  VkDeviceMemory dev_mem() const;

  size_t size() const;

  bool context_changing() override;
  bool context_changed() override;

  bool send(const void* data, size_t dst_offset, size_t size);
  bool fetch(L_OUT void* data, size_t src_offset, size_t size);
};



class StorageBuffer;

class StorageBufferView {
  friend class StorageBuffer;
private:
  std::shared_ptr<const StorageBuffer> _buf;
  size_t _offset;
  size_t _size;

  StorageBufferView(std::shared_ptr<const StorageBuffer> buf,
    size_t offset, size_t size);

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
  size_t _offset;
  size_t _size;
  VkBuffer _buf;
  ExecType _ty;
  VkBufferUsageFlags _usage;

public:
  StorageBuffer(size_t size, ExecType ty, VkBufferUsageFlags usage);
  ~StorageBuffer();
  
  StorageBuffer(const StorageBuffer&) = delete;
  StorageBuffer& operator=(const StorageBuffer&) = delete;

  bool context_changing() override;
  bool context_changed() override;

  Storage& storage();
  const Storage& storage() const;
  size_t size() const;
  VkBuffer buf() const;

  StorageBufferView view() const;
  StorageBufferView view(size_t offset, size_t size) const;

  // There is extra size for metadata required by the driver. So the resultant
  // memory allocation should be `size + meta_size`, which is slightly larger
  // than that we required via the constructor.
  size_t alloc_size() const;

  // FIXME: (penguinliong) Memory bindings will be invalid after device change
  // because the underlying storages are not necessarily created after the
  // buffer objects. The current solution is error-prone. A better device change
  // strategy should be devised later.
  bool bind(std::shared_ptr<Storage> storage, size_t offset);
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
  StorageImageView(std::shared_ptr<const StorageImage> img);
  StorageImageView(std::shared_ptr<const StorageImage> img,
    VkOffset2D offset, VkExtent2D extent);

  bool context_changing() override;
  bool context_changed() override;

  VkImage img() const;
  VkImageView img_view() const;
  const VkOffset2D& offset() const;
  const VkExtent2D& extent() const;
  uint32_t nlayer() const;
  VkImageLayout layout() const;
};

enum class ImageType {
  Image1D, Image2D, Image3D,
  Image1DArray, Image2DArray, Image3DArray,
};

class StorageImage : public Contextual,
  public std::enable_shared_from_this<StorageImage> {
private:
  VkImage _img;

  std::shared_ptr<Storage> _storage;
  size_t _offset;

  VkExtent2D _extent;
  uint32_t _nlayer;
  ExecType _exec_ty;
  StorageOptimization _opt;
  VkImageType _img_ty;
  VkFormat _format;
  VkImageUsageFlags _usage;
  VkImageLayout _layout;
  VkImageTiling _tiling;

public:
  StorageImage(const VkExtent2D& extent, uint32_t nlayer,
    ExecType exec_ty, StorageOptimization opt,
    // FIXME: (penguinliong) Redesign how to pass the following information.
    VkImageType img_ty, VkFormat format,
    VkImageUsageFlags usage, VkImageLayout layout,
    VkImageTiling tiling);

  ~StorageImage();

  bool context_changing() override;
  bool context_changed() override;

  Storage& storage();
  const Storage& storage() const;
  const VkExtent2D& extent() const;
  VkImage img() const;
  VkImageLayout layout() const;
  uint32_t nlayer() const;
  size_t size() const;

  size_t alloc_size() const;

  bool bind(std::shared_ptr<Storage> storage, size_t offset);
};

L_CUVK_END_
