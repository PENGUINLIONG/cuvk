#include "cuvk/executor.hpp"
#include "cuvk/context.hpp"
#include "cuvk/logger.hpp"
#include <map>

L_CUVK_BEGIN_

Fence::Fence(const Context& ctxt) noexcept :
  ctxt(&ctxt),
  fence(VK_NULL_HANDLE) {}
bool Fence::make() noexcept {
  if (fence) {
    // Already created, reset to initialized state.
    if (L_VK <- vkResetFences(ctxt->dev, 1, &fence)) {
      LOG.error("unable to reset fence");
      return false;
    }
  } else {
    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (L_VK <- vkCreateFence(ctxt->dev, &fci, nullptr, &fence)) {
      LOG.error("unable to create fence");
      return false;
    }
  }
  return true;
}
void Fence::drop() noexcept {
  if (fence) {
    vkDestroyFence(ctxt->dev, fence, nullptr);
    fence = VK_NULL_HANDLE;
  }
}
Fence::~Fence() noexcept { drop(); }

Fence::Fence(Fence&& right) noexcept :
  ctxt(right.ctxt),
  fence(std::exchange(right.fence, nullptr)) {}

FenceStatus Fence::wait() noexcept {
  // Most tasks are assumed to finish within 100ms.
  FenceStatus status = wait_for(100'000'000, false);
  if (status == FenceStatus::Timeout) {
    uint32_t n = 1;
    LOG.warning("the fence hasn't been signaled within 100ms");
    while ((status = wait_for(100'000'000, false)) == FenceStatus::Timeout) {
      ++n;
    }
    LOG.warning("it took more than {}ms for the device to signal the fence",
      100 * n);
    return status;
  } else {
    return FenceStatus::Ok;
  }
}
FenceStatus Fence::wait_for(uint32_t ns, bool warn_timeout) noexcept {
  auto res = vkWaitForFences(ctxt->dev, 1, &fence, true, ns);
  if (res == VK_TIMEOUT) {
    if (warn_timeout) {
      LOG.warning("waited for fence for longer than {}ns", ns);
    }
    return FenceStatus::Timeout;
  } else if (L_VK <- res) {
    return FenceStatus::Error;
  }
  return FenceStatus::Ok;
}
FenceStatus Fence::wait_for(uint32_t ns) noexcept {
  return wait_for(ns, true);
}



Semaphore::Semaphore(const Context& ctxt) noexcept :
  ctxt(&ctxt),
  sem(VK_NULL_HANDLE) {}
bool Semaphore::make() noexcept {
  if (sem) { return true; }

  VkSemaphoreCreateInfo sci {};
  sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  
  if (L_VK <- vkCreateSemaphore(ctxt->dev, &sci, nullptr, &sem)) {
    LOG.error("unable to create semaphore");
    return false;
  }
  return true;
}
void Semaphore::drop() noexcept {
  if (sem) {
    vkDestroySemaphore(ctxt->dev, sem, nullptr);
    sem = VK_NULL_HANDLE;
  }
}
Semaphore::~Semaphore() noexcept { drop(); }

Semaphore::Semaphore(Semaphore&& right) noexcept :
  ctxt(right.ctxt),
  sem(std::exchange(right.sem, nullptr)) {}



Execution::Execution(const Executable& exec) noexcept:
  exec(&exec),
  nwait_sem(),
  wait_sems(),
  wait_stages(),
  nsignal_sem(),
  signal_sems() {}

Execution& Execution::wait(
  const Semaphore& sem, VkPipelineStageFlags stage) noexcept{
  wait_sems[nwait_sem] = sem.sem;
  wait_stages[nwait_sem] = stage;
  ++nwait_sem;
  return *this;
}
Execution& Execution::signal(const Semaphore& sem) noexcept {
  signal_sems[nsignal_sem++] = sem.sem;
  return *this;
}
bool Execution::submit(const Fence& fence) noexcept {
  VkSubmitInfo si {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &exec->cmd_buf;
  si.waitSemaphoreCount = nwait_sem;
  si.pWaitSemaphores = wait_sems.data();
  si.pWaitDstStageMask = wait_stages.data();
  si.signalSemaphoreCount = nsignal_sem;
  si.pSignalSemaphores = signal_sems.data();

  if (L_VK <- vkQueueWaitIdle(exec->queue->queue)) {
    LOG.error("unable to wait for queue to be idle");
    return false;
  }

  if (L_VK <- vkQueueSubmit(exec->queue->queue, 1, &si, fence.fence)) {
    LOG.error("unable to submit sommand buffer to queue");
    return false;
  }
  return true;
}



CommandRecorder::CommandRecorder(const Executable& exec) noexcept :
  exec(&exec),
  cur((VkPipelineStageFlagBits)0),
  nimb(0),
  nbmb(0) {}
bool CommandRecorder::begin() noexcept {
  VkCommandBufferBeginInfo cbbi {};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  if (L_VK <- vkBeginCommandBuffer(exec->cmd_buf, &cbbi)) {
    LOG.error("unable to record commands");
    return false;
  }
  status = CommandRecorderStatus::OnAir;
  return true;
}
bool CommandRecorder::end() noexcept {
  if (L_VK <- vkEndCommandBuffer(exec->cmd_buf)) {
    LOG.error("unable to finish command recording");
    return false;
  }
  status = CommandRecorderStatus::Done;
  return true;
}
CommandRecorder::~CommandRecorder() noexcept {
  if (status == CommandRecorderStatus::Ready) {
    LOG.warning("command buffer recording is not started");
  } else if (status != CommandRecorderStatus::Done) {
    LOG.warning("command buffer recording is not ended");
  }
}

CommandRecorder& CommandRecorder::from_stage(
  VkPipelineStageFlagBits stage) noexcept {
  if (status != CommandRecorderStatus::OnAir) {
    LOG.warning("command buffer recording is not started");
  }
  cur = stage;
  status = CommandRecorderStatus::Barrier;
  return *this;
}
CommandRecorder& CommandRecorder::barrier(const ImageSlice& img_slice,
  VkAccessFlags src_access, VkAccessFlags dst_access,
  VkImageLayout old_layout, VkImageLayout new_layout) noexcept {
  if (status != CommandRecorderStatus::Barrier) {
    LOG.warning("barrier recording is not started");
  }
  VkImageMemoryBarrier imb {};
  imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imb.srcAccessMask = src_access;
  imb.dstAccessMask = dst_access;
  imb.oldLayout = old_layout;
  imb.newLayout = new_layout;
  imb.image = img_slice.img_alloc->img;
  imb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imb.subresourceRange.baseArrayLayer = img_slice.base_layer;
  imb.subresourceRange.layerCount = img_slice.nlayer.value_or(1);
  imb.subresourceRange.baseMipLevel = 0;
  imb.subresourceRange.levelCount = 1;

  imbs[nimb++] = imb;
  return *this;
}
CommandRecorder& CommandRecorder::barrier(const BufferSlice& buf_slice,
  VkAccessFlags src_access, VkAccessFlags dst_access) noexcept {
  if (status != CommandRecorderStatus::Barrier) {
    LOG.warning("barrier recording is not started");
  }
  VkBufferMemoryBarrier bmb {};
  bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bmb.srcAccessMask = src_access;
  bmb.dstAccessMask = dst_access;
  bmb.buffer = buf_slice.buf_alloc->buf;
  bmb.offset = buf_slice.offset;
  bmb.size = buf_slice.size;

  bmbs[nbmb++] = bmb;
  return *this;
}
CommandRecorder& CommandRecorder::to_stage(
  VkPipelineStageFlagBits stage) noexcept {
  if (status != CommandRecorderStatus::Barrier) {
    LOG.warning("barrier recording is not started");
  }
  if (nimb == 0 && nbmb == 0) {
    return *this;
  }
  vkCmdPipelineBarrier(exec->cmd_buf,
    cur, stage, 0,
    0, nullptr, nbmb, bmbs.data(), nimb, imbs.data());
  status = CommandRecorderStatus::OnAir;
  cur = (VkPipelineStageFlagBits)0;
  nimb = 0;
  nbmb = 0;
  return *this;
}



CommandRecorder& CommandRecorder::copy_buf_to_buf(
  const BufferSlice& src, const BufferSlice& dst) noexcept {
  if (status != CommandRecorderStatus::OnAir) {
    LOG.warning("command buffer recording is not started");
  }
  VkBufferCopy bc {};
  bc.srcOffset = src.offset;
  bc.dstOffset = dst.offset;
  bc.size = src.size;

  vkCmdCopyBuffer(exec->cmd_buf,
    src.buf_alloc->buf, dst.buf_alloc->buf, 1, &bc);
  return *this;
}
CommandRecorder& CommandRecorder::copy_buf_to_img(
  const BufferSlice& src, const ImageSlice& dst) noexcept {
  if (status != CommandRecorderStatus::OnAir) {
    LOG.warning("command buffer recording is not started");
  }
  VkBufferImageCopy bic {};
  bic.bufferRowLength = dst.img_alloc->req.extent.width;
  bic.bufferImageHeight = dst.img_alloc->req.extent.height;
  bic.bufferOffset = src.offset;
  bic.imageExtent = {
    dst.img_alloc->req.extent.width,
    dst.img_alloc->req.extent.height,
    1
  };
  bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  bic.imageSubresource.baseArrayLayer = dst.base_layer;
  bic.imageSubresource.layerCount = dst.nlayer.value_or(1);

  vkCmdCopyBufferToImage(exec->cmd_buf,
    src.buf_alloc->buf,
    dst.img_alloc->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    1, &bic);
  return *this;
}
CommandRecorder& CommandRecorder::copy_img_to_buf(
  const ImageSlice& src, const BufferSlice& dst) noexcept {
  if (status != CommandRecorderStatus::OnAir) {
    LOG.warning("command buffer recording is not started");
  }
  VkBufferImageCopy bic {};
  bic.bufferRowLength = src.img_alloc->req.extent.width;
  bic.bufferImageHeight = src.img_alloc->req.extent.height;
  bic.bufferOffset = dst.offset;
  bic.imageExtent = {
    src.img_alloc->req.extent.width,
    src.img_alloc->req.extent.height,
    1
  };
  bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  bic.imageSubresource.baseArrayLayer = src.base_layer;
  bic.imageSubresource.layerCount = src.nlayer.value_or(1);

  vkCmdCopyImageToBuffer(exec->cmd_buf,
    src.img_alloc->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    dst.buf_alloc->buf,
    1, &bic);
  return *this;
}
CommandRecorder& CommandRecorder::copy_img_to_img(
  const ImageSlice& src, const ImageSlice& dst) noexcept {
  if (status != CommandRecorderStatus::OnAir) {
    LOG.warning("command buffer recording is not started");
  }
  VkImageCopy ic {};
  ic.extent = {
    src.img_alloc->req.extent.width,
    src.img_alloc->req.extent.height,
    1
  };
  ic.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  ic.dstSubresource.layerCount = src.nlayer.value_or(1);
  ic.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

  vkCmdCopyImage(exec->cmd_buf,
    src.img_alloc->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    dst.img_alloc->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    1, &ic);
  return *this;
}

CommandRecorder& CommandRecorder::push_const(
  const ComputePipeline& comp_pipe,
  uint32_t dst_offset, uint32_t size,
  const void* consts) noexcept {
  if (status != CommandRecorderStatus::OnAir) {
    LOG.warning("command buffer recording is not started");
  }
  vkCmdPushConstants(exec->cmd_buf, comp_pipe.pipe_layout,
    VK_SHADER_STAGE_COMPUTE_BIT, dst_offset, size, consts);
  return *this;
}
CommandRecorder& CommandRecorder::push_const(
  const GraphicsPipeline& graph_pipe, VkPipelineStageFlags stages,
  uint32_t dst_offset, uint32_t size,
  const void* consts) noexcept {
  if (status != CommandRecorderStatus::OnAir) {
    LOG.warning("command buffer recording is not started");
  }
  vkCmdPushConstants(exec->cmd_buf, graph_pipe.pipe_layout,
    stages, dst_offset, size, consts);
  return *this;
}

CommandRecorder& CommandRecorder::dispatch(
  const ComputePipeline& comp_pipe,
  std::optional<const DescriptorSet*> desc_set,
  uint32_t x, uint32_t y, uint32_t z) noexcept {
  if (status != CommandRecorderStatus::OnAir) {
    LOG.warning("command buffer recording is not started");
  }
  vkCmdBindPipeline(exec->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
    comp_pipe.pipe);
  if (desc_set.has_value()) {
    vkCmdBindDescriptorSets(exec->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
      comp_pipe.pipe_layout, 0, 1, &(*desc_set)->desc_set, 0, nullptr);
  }
  vkCmdDispatch(exec->cmd_buf, x, y, z);
  return *this;
}
CommandRecorder& CommandRecorder::draw(
  const GraphicsPipeline& graph_pipe,
  std::optional<const DescriptorSet*> desc_set,
  const BufferSlice& vert_buf, uint32_t nvert,
  const Framebuffer& framebuf) noexcept {
  auto viewport = framebuf.req.extent;

  if (status != CommandRecorderStatus::OnAir) {
    LOG.warning("command buffer recording is not started");
  }
  std::array<VkClearValue, 1> cv;
  cv[0].color = { { 0., 0., 0., 1. } };

  VkRenderPassBeginInfo rpbi {};
  rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpbi.renderPass = framebuf.pass->pass;
  rpbi.framebuffer = framebuf.framebuf;
  rpbi.renderArea.extent = viewport;
  rpbi.clearValueCount = static_cast<uint32_t>(cv.size());
  rpbi.pClearValues = cv.data();

  vkCmdBeginRenderPass(exec->cmd_buf, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport v {};
  v.width = (float)viewport.width;
  v.height = (float)viewport.height;
  v.minDepth = 0.;
  v.maxDepth = 1.;

  vkCmdSetViewport(exec->cmd_buf, 0, 1, &v);

  VkRect2D scissor = {};
  scissor.extent = viewport;

  vkCmdSetScissor(exec->cmd_buf, 0, 1, &scissor);

  VkBuffer buf = vert_buf.buf_alloc->buf;
  VkDeviceSize offset = vert_buf.offset;
  vkCmdBindVertexBuffers(exec->cmd_buf, 0, 1, &buf, &offset);

  vkCmdBindPipeline(exec->cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS,
    graph_pipe.pipe);
  if (desc_set.has_value()) {
    vkCmdBindDescriptorSets(exec->cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS,
      graph_pipe.pipe_layout, 0, 1, &(*desc_set)->desc_set, 0, nullptr);
  }
  // TODO: (penguinliong) Push constants.
  vkCmdDraw(exec->cmd_buf, nvert, 1, 0, 0);
  vkCmdEndRenderPass(exec->cmd_buf);
  return *this;
}



Executable::Executable(const Context& ctxt, const Queue& queue) noexcept:
  ctxt(&ctxt),
  queue(&queue),
  cmd_pool(VK_NULL_HANDLE),
  cmd_buf(VK_NULL_HANDLE) {}
bool Executable::make() noexcept {
  if (cmd_pool) {
    // Already created, reset command buffer to initialized state.
    if (L_VK <- vkResetCommandBuffer(cmd_buf, 0)) {
      LOG.error("unable to reset command buffer");
      return false;
    }
  } else {
    // Create everything required.
    VkCommandPoolCreateInfo cpci {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queue->queue_fam_idx;

    if (L_VK <- vkCreateCommandPool(ctxt->dev, &cpci, nullptr, &cmd_pool)) {
      LOG.error("unable to create command pool");
      return false;
    }
    VkCommandBufferAllocateInfo cbai {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool;
    cbai.commandBufferCount = 1;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    if (L_VK <- vkAllocateCommandBuffers(ctxt->dev, &cbai, &cmd_buf)) {
      LOG.error("unable to allocate command buffer");
      return false;
    }
  }
  return true;
}
void Executable::drop() noexcept {
  if (cmd_buf) {
    vkFreeCommandBuffers(ctxt->dev, cmd_pool, 1, &cmd_buf);
    cmd_buf = VK_NULL_HANDLE;
  }
  if (cmd_pool) {
    vkDestroyCommandPool(ctxt->dev, cmd_pool, nullptr);
    cmd_pool = VK_NULL_HANDLE;
  }
}
Executable::~Executable() noexcept { drop(); }

Executable::Executable(Executable&& right) noexcept :
  ctxt(right.ctxt), queue(right.queue),
  cmd_pool(right.cmd_pool), cmd_buf(cmd_buf) {}

CommandRecorder Executable::record() noexcept {
  return { *this };
}
Execution Executable::execute() const noexcept {
  return Execution(*this);
}


L_CUVK_END_
