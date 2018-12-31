#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/config.hpp"
#include "cuvk/storage.hpp"
#include "cuvk/pipeline.hpp"
#include <variant>

L_CUVK_BEGIN_

struct Context;
struct Queue;

enum class FenceStatus {
  Ok, Error, Timeout
};

struct Fence {
  const Context* ctxt;

  VkFence fence;

  Fence(const Context& ctxt) noexcept;
  bool make() noexcept;
  void drop() noexcept;
  ~Fence() noexcept;

  Fence(const Fence&) = delete;
  Fence& operator=(const Fence&) = delete;
  Fence(Fence&&) noexcept;

  FenceStatus wait() noexcept;
  FenceStatus wait_for(uint32_t ns) noexcept;
  FenceStatus wait_for(uint32_t ns, bool warn_timeout) noexcept;
};

struct Semaphore {
  const Context* ctxt;

  VkSemaphore sem;

  Semaphore(const Context& ctxt) noexcept;
  bool make() noexcept;
  void drop() noexcept;
  ~Semaphore() noexcept;

  Semaphore(const Semaphore&) = delete;
  Semaphore& operator=(const Semaphore&) = delete;

  Semaphore(Semaphore&&) noexcept;
};

struct Executable;
struct Execution {
  const Executable* exec;

  uint32_t nwait_sem;
  std::array<VkSemaphore, 4> wait_sems;
  std::array<VkPipelineStageFlags, 4> wait_stages;

  uint32_t nsignal_sem;
  std::array<VkSemaphore, 4> signal_sems;

  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;

  Execution& wait(const Semaphore& sem, VkPipelineStageFlags stage) noexcept;
  Execution& signal(const Semaphore& sem) noexcept;
  bool submit(const Fence& fence) noexcept;

private:
  friend struct Executable;
  Execution(const Executable& exec) noexcept;
};

enum class CommandRecorderStatus {
  Ready, OnAir, Barrier, Done
};
struct CommandRecorder {
  const Executable* exec;

  VkPipelineStageFlagBits cur;
  std::array<VkImageMemoryBarrier, 4> imbs;
  uint32_t nimb;
  std::array<VkBufferMemoryBarrier, 4> bmbs;
  uint32_t nbmb;

  CommandRecorderStatus status;

  CommandRecorder(const Executable& exec) noexcept;
  bool begin() noexcept;
  bool end() noexcept;
  ~CommandRecorder() noexcept;

  CommandRecorder& from_stage(VkPipelineStageFlagBits stage) noexcept;
  CommandRecorder& barrier(const ImageSlice& img_slice,
    VkAccessFlags src_access, VkAccessFlags dst_access,
    VkImageLayout old_layout, VkImageLayout new_layout) noexcept;
  CommandRecorder& barrier(const BufferSlice& buf_slice,
    VkAccessFlags src_access, VkAccessFlags dst_access) noexcept;
  CommandRecorder& to_stage(VkPipelineStageFlagBits stage) noexcept;

  CommandRecorder& copy_buf_to_buf(
    const BufferSlice& src, const BufferSlice& dst) noexcept;
  CommandRecorder& copy_buf_to_img(
    const BufferSlice& src, const ImageSlice& dst) noexcept;
  CommandRecorder& copy_img_to_buf(
    const ImageSlice& src, const BufferSlice& dst) noexcept;
  CommandRecorder& copy_img_to_img(
    const ImageSlice& src, const ImageSlice& dst) noexcept;

  CommandRecorder& push_const(
    const ComputePipeline& comp_pipe,
    uint32_t dst_offset, uint32_t size,
    const void* consts) noexcept;
  CommandRecorder& push_const(
    const GraphicsPipeline& graph_pipe, VkPipelineStageFlags stages,
    uint32_t dst_offset, uint32_t size,
    const void* consts) noexcept;

  CommandRecorder& dispatch(
    const ComputePipeline& comp_pipe,
    std::optional<const DescriptorSet*> desc_set,
    uint32_t x, uint32_t y, uint32_t z) noexcept;
  CommandRecorder& draw(
    const GraphicsPipeline& graph_pipe,
    std::optional<const DescriptorSet*> desc_set,
    const BufferSlice& vert_buf, uint32_t nvert,
    const Framebuffer& framebuf) noexcept;
};

struct Executable {
  const Context* ctxt;
  const Queue* queue;

  VkCommandPool cmd_pool;
  VkCommandBuffer cmd_buf;

  Executable(const Context& ctxt, const Queue& queue) noexcept;
  bool make() noexcept;
  void drop() noexcept;
  ~Executable() noexcept;

  Executable(const Executable&) = delete;
  Executable& operator=(const Executable&) = delete;
  Executable(Executable&& right) noexcept;

  CommandRecorder record() noexcept;
  Execution execute() const noexcept;
};

L_CUVK_END_
