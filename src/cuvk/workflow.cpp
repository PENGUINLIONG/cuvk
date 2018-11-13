#include "cuvk/workflow.hpp"
#include "cuvk/logger.hpp"
#include "cuvk/shader_interface.hpp"
#include <fstream>

L_CUVK_BEGIN_

Spirv read_spirv(const std::string& path) {
  std::fstream f(path, std::ios::in | std::ios::binary | std::ios::ate);
  if (!f) {
    LOG.error("unable to read spirv: {}", path);
    return {};
  }
  size_t size = f.tellg();
  std::vector<char> buf;
  buf.resize(size);
  f.seekg(std::ios::beg);
  f.read(buf.data(), size);
  f.close();
  auto spirv = Spirv(buf.data(), size);

  return spirv;
}

std::vector<Deformation::LayoutBinding> Deformation::_layout_binds = {
  VkDescriptorSetLayoutBinding
  // DeformSpecs[] deform_specs
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
  // Bac[] bacs
  { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
  // Bac[] bacs
  { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
};

const std::vector<Deformation::LayoutBinding>& Deformation::layout_binds() const {
  return _layout_binds;
}
Spirv Deformation::comp_spirv() const {
  return read_spirv("assets/shaders/deform.comp.spv");
}

bool Deformation::execute(const StorageBufferView& deform_specs,
                          const StorageBufferView& bacs,
                          L_OUT StorageBufferView& bacs_out) {
  auto dev = ctxt().dev();
  auto cmd_pool = ctxt().get_cmd_pool(ExecType::Compute);

  VkCommandBuffer cmd_buf;

  // Calculate number of specs and bacteria provided.
  size_t nspec = deform_specs.size() / sizeof(shader_interface::DeformSpecs);
  size_t nbac = bacs.size() / sizeof(shader_interface::Bacterium);
  
  // Ensure that the output buffer is larger than or as large as the input
  // bacteria buffer.
  if (bacs_out.size() / sizeof(shader_interface::Bacterium) < nbac * nspec) {
    LOG.error("output buffer must be able to contain the result data");
    return false;
  }

  /* Allocate command buffer. */ {
    VkCommandBufferAllocateInfo cbai {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    cbai.commandPool = cmd_pool;

    if (L_VK <- vkAllocateCommandBuffers(dev, &cbai, &cmd_buf)) {
      LOG.error("unable to allocate command buffer for deformation");
      return false;
    }
  }

  /* Start recording commands. */ {
    VkCommandBufferBeginInfo cbbi {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (L_VK <- vkBeginCommandBuffer(cmd_buf, &cbbi)) {
      LOG.error("unable to record commands");
      return false;
    }
  }

  /* Sync to ensure all bacteria data is transferred to the device memory. */ {
    std::array<VkBufferMemoryBarrier, 2> bmbs {};
    auto& bmb0 = bmbs[0];
    auto& bmb1 = bmbs[1];

    VkBufferMemoryBarrier bmb_template {};
    bmb_template.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb_template.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bmb_template.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bmb_template.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb_template.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    bmb0 = bmb_template;
    bmb0.buffer = deform_specs.buf();
    bmb0.offset = deform_specs.offset();
    bmb0.size = deform_specs.size();

    bmb1 = bmb_template;
    bmb1.buffer = bacs.buf();
    bmb1.offset = bacs.offset();
    bmb1.size = bacs.size();

    vkCmdPipelineBarrier(cmd_buf,
      VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, // No dependency flags.
      0, nullptr,
      2, bmbs.data(),
      0, nullptr);
  }

  /* Update buffer descriptor sets. */ {
    std::array<VkWriteDescriptorSet, 3> wdss {};
    auto& wds_specs = wdss[0];
    auto& wds_bacs = wdss[1];
    auto& wds_bacs_out = wdss[2];

    VkWriteDescriptorSet wds_template {};
    wds_template.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds_template.descriptorCount = 1;
    wds_template.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds_template.dstSet = _desc_set;

    VkDescriptorBufferInfo dbi_specs {};
    dbi_specs.buffer = deform_specs.buf();
    dbi_specs.offset = deform_specs.offset();
    dbi_specs.range = deform_specs.size();

    wds_specs = wds_template;
    wds_specs.dstBinding = 0;
    wds_specs.pBufferInfo = &dbi_specs;

    VkDescriptorBufferInfo dbi_bacs {};
    dbi_bacs.buffer = bacs.buf();
    dbi_bacs.offset = bacs.offset();
    dbi_bacs.range = bacs.size();

    wds_bacs = wds_template;
    wds_bacs.dstBinding = 1;
    wds_bacs.pBufferInfo = &dbi_bacs;

    VkDescriptorBufferInfo dbi_bacs_out {};
    dbi_bacs_out.buffer = bacs_out.buf();
    dbi_bacs_out.offset = bacs_out.offset();
    dbi_bacs_out.range = bacs_out.size();

    wds_bacs_out = wds_template;
    wds_bacs_out.dstBinding = 2;
    wds_bacs_out.pBufferInfo = &dbi_bacs_out;

    vkUpdateDescriptorSets(dev, (uint32_t)wdss.size(), wdss.data(), 0, nullptr);
  }

  /* Bind compute pipeline and descriptor sets. */ {
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, _pl);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
      _pl_layout, 0, 1, &_desc_set, 0, nullptr);
  }

  /* Dispatch deformation. */ {
    // TODO: (penguinliong) Push constants.
    vkCmdDispatch(cmd_buf,
      static_cast<uint32_t>(nspec),
      static_cast<uint32_t>(nbac),
      1);
  }

  /* Sync for host to read. */ {
    VkBufferMemoryBarrier bmb {};
    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer = bacs_out.buf();
    bmb.offset = bacs_out.offset();
    bmb.size = bacs_out.size();

    vkCmdPipelineBarrier(cmd_buf,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,
      0,
      0, nullptr,
      1, &bmb,
      0, nullptr);
  }

  /* Finish recording. */ {
    if (L_VK <- vkEndCommandBuffer(cmd_buf)) {
      LOG.error("unable to finish command recording");
    }
  }

  /* Dispatch computing job to devices. */ {

    // TODO: (penguinliong) Reuse the fence and adapt for execution of multiple
    // command buffers, i.e. use of multiple fences.
    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence fence;
    if (L_VK <- vkCreateFence(dev, &fci, nullptr, &fence)) {
      LOG.error("unable to create fence");
    }

    vkResetFences(dev, 1, &fence);

    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf;

    VkQueue queue = ctxt().get_queue(ExecType::Compute);

    if (L_VK <- vkQueueWaitIdle(queue)) {
      LOG.error("unable to wait for target queue");
      return false;
    }

    if (L_VK <- vkQueueSubmit(queue, 1, &si, fence)) {
      LOG.error("unable to submit commands to device");
      return false;
    }
    const uint64_t timeout = 5'000'000;
    if (L_VK <- vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout)) {
      LOG.error("execution timeout ({}ns)", timeout);
      return false;
    }

    vkDestroyFence(dev, fence, nullptr);
  }
  vkFreeCommandBuffers(dev, cmd_pool, 1, &cmd_buf);

  return true;
}

std::vector<Evaluation::LayoutBinding> Evaluation::_layout_binds = {
  VkDescriptorSetLayoutBinding
  // image2D real_univ
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
    VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
};
std::vector<Evaluation::Binding> Evaluation::_in_binds = {
  VkVertexInputBindingDescription
  { 0, 6 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX }, // Bacterium
};
std::vector<Evaluation::Attribute> Evaluation::_in_attrs = {
  VkVertexInputAttributeDescription
  { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0                 }, // pos
  { 1, 0, VK_FORMAT_R32G32_SFLOAT, 2 * sizeof(float) }, // size
  { 2, 0, VK_FORMAT_R32_SFLOAT,    4 * sizeof(float) }, // orient
  { 3, 0, VK_FORMAT_R32_SINT,      5 * sizeof(float) }, // univ
};
std::vector<Evaluation::Blend> Evaluation::_out_blends = {
};
std::vector<Evaluation::Attachment> Evaluation::_out_attaches = {
  VkAttachmentDescription
  {
    0, VK_FORMAT_R32_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
    VK_ATTACHMENT_LOAD_OP_CLEAR,
    VK_ATTACHMENT_STORE_OP_STORE,
    VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    VK_ATTACHMENT_STORE_OP_DONT_CARE,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
  },
};


const std::vector<Evaluation::LayoutBinding>& Evaluation::layout_binds() const {
  return _layout_binds;
}
const std::vector<Evaluation::Binding>& Evaluation::in_binds() const {
  return _in_binds;
}
const std::vector<Evaluation::Attribute>& Evaluation::in_attrs() const {
  return _in_attrs;
}
const std::vector<Evaluation::Blend>& Evaluation::out_blends() const {
  return _out_blends;
}
const std::vector<Evaluation::Attachment>& Evaluation::out_attaches() const {
  return _out_attaches;
}
Spirv Evaluation::vert_spirv() const {
  return read_spirv("assets/shaders/eval.vert.spv");
}
Spirv Evaluation::geom_spirv() const {
  return read_spirv("assets/shaders/eval.geom.spv");
}
Spirv Evaluation::frag_spirv() const {
  return read_spirv("assets/shaders/eval.frag.spv");
}

Evaluation::Evaluation(uint32_t rows, uint32_t cols, uint32_t layers) :
  GraphicsShaderContextual(rows, cols, layers) {
}

bool Evaluation::execute(const StorageBufferView& bacs,
                         const StorageBufferView& real_univ_buf,
                         const StorageImageView& real_univ,
                         L_OUT StorageBufferView& sim_univs_buf,
                         L_OUT StorageImageView& sim_univs,
                         L_OUT StorageBufferView& costs) {
  auto dev = ctxt().dev();
  auto cmd_pool = ctxt().get_cmd_pool(ExecType::Graphics);

  VkCommandBuffer cmd_buf;
  VkFramebuffer f;

  VkExtent2D extent = real_univ.extent();
  uint32_t nlayer = sim_univs.nlayer().value_or(1);

  VkImageView iv = sim_univs.img_view();

  VkFramebufferCreateInfo fci {};
  fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fci.renderPass = render_pass();
  fci.attachmentCount = 1;
  fci.pAttachments = &iv;
  fci.width = extent.width;
  fci.height = extent.height;
  fci.layers = nlayer;

  if (L_VK <- vkCreateFramebuffer(dev, &fci, nullptr, &f)) {
    LOG.error("unable to create framebuffer");
    return false;
  }

  /* Allocate command buffer. */ {
    VkCommandBufferAllocateInfo cbai {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    cbai.commandPool = cmd_pool;

    if (L_VK <- vkAllocateCommandBuffers(dev, &cbai, &cmd_buf)) {
      LOG.error("unable to allocate command buffer for evaluation");
      return false;
    }
  }

  auto nbac = bacs.size() / sizeof(shader_interface::Bacterium);

  /* Start recording commands. */ {
    VkCommandBufferBeginInfo cbbi {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (L_VK <- vkBeginCommandBuffer(cmd_buf, &cbbi)) {
      LOG.error("unable to start recording commands");
      return false;
    }
  }

  /* Sync input bacteria. */ {
    VkBufferMemoryBarrier bmb = bacs.barrier(
      VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    vkCmdPipelineBarrier(cmd_buf,
      VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
      0,
      0, nullptr,
      1, &bmb,
      0, nullptr);
  }

  /* Sync to ensure the real universe are written to device memory. */ {
    VkBufferMemoryBarrier bmb = real_univ_buf.barrier(
      VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    vkCmdPipelineBarrier(cmd_buf,
      VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, // No dependency flags.
      0, nullptr,
      1, &bmb,
      0, nullptr);
  }

  /* Sync again to ensure buffer data is fully written to the image object. */ {
    auto imb = real_univ.barrier(
      0, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdPipelineBarrier(cmd_buf,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0,
      0, nullptr,
      0, nullptr,
      1, &imb);
  }

  /* Transfer real universe's pixel data to image. */ {
    auto bic = real_univ.copy_with_buffer(real_univ_buf);
    vkCmdCopyBufferToImage(cmd_buf, real_univ_buf.buf(), real_univ.img(),
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
  }

  /* Sync again to real universe data is fully written to the image object. */ {
    auto imb = real_univ.barrier(
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdPipelineBarrier(cmd_buf,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      0,
      0, nullptr,
      0, nullptr,
      1, &imb);
  }

  /* Invalidate previous simulation results. */ {
    auto imb = sim_univs.barrier(
      0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdPipelineBarrier(cmd_buf,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      0,
      0, nullptr,
      0, nullptr,
      1, &imb);
  }

  /* Update description. */ {
    VkDescriptorImageInfo dii {};
    dii.imageView = real_univ.img_view();
    dii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet wds {};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = _desc_set;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wds.dstBinding = 0;
    wds.pImageInfo = &dii;

    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);
  }

  /* Record render pass. */ {
    std::array<VkClearValue, 2> cv;
    cv[0].color = { { 0., 0., 0., 1. } };
    cv[1].depthStencil = { 1., 0 };

    VkRenderPassBeginInfo rpbi {};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = render_pass();
    rpbi.renderArea.extent.width = sim_univs.extent().width;
    rpbi.renderArea.extent.height = sim_univs.extent().height;
    rpbi.clearValueCount = cv.size();
    rpbi.pClearValues = cv.data();
    rpbi.framebuffer = f;

    vkCmdBeginRenderPass(cmd_buf, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport {};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.;
    viewport.maxDepth = 1.;

    vkCmdSetViewport(cmd_buf, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.extent.width = extent.width;
    scissor.extent.height = extent.height;

    vkCmdSetScissor(cmd_buf, 0, 1, &scissor);

    VkBuffer buf = bacs.buf();
    VkDeviceSize offset = bacs.offset();
    vkCmdBindVertexBuffers(cmd_buf, 0, 1, &buf, &offset);

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, _pl);

    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS,
      _pl_layout, 0, 1, &_desc_set, 0, nullptr);

    // TODO: (penguinliong) Push constants.
    vkCmdDraw(cmd_buf, nbac, 1, 0, 0);
    vkCmdEndRenderPass(cmd_buf);
  }

  /* Transfer data to output buffer. */ {
    auto bic = sim_univs.copy_with_buffer(sim_univs_buf);
    vkCmdCopyImageToBuffer(cmd_buf, sim_univs.img(),
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sim_univs_buf.buf(), 1, &bic);
  }

  /* Sync for host to read. */ {
    auto bmb = sim_univs_buf.barrier(
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    vkCmdPipelineBarrier(cmd_buf,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,
      0,
      0, nullptr,
      1, &bmb,
      0, nullptr);
  }

  /* End recording commands. */ {
    if (L_VK <- vkEndCommandBuffer(cmd_buf)) {
      LOG.error("unable to finish recording commands");
      return false;
    }
  }

  /* Submit to device for execution. */ {
    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence fence;
    if (L_VK <- vkCreateFence(dev, &fci, nullptr, &fence)) {
      LOG.error("unable to create fence");
      return false;
    }

    vkResetFences(dev, 1, &fence);

    VkQueue queue = ctxt().get_queue(ExecType::Graphics);

    if (L_VK <- vkQueueWaitIdle(queue)) {
      LOG.error("failed to wait graphics queue to be idle");
      return false;
    }

    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf;
    if (L_VK <- vkQueueSubmit(queue, 1, &si, fence)) {
      LOG.error("unable to submit queue");
      return false;
    }
    // TODO: (penguinliong) Use constants.
    if (vkWaitForFences(dev, 1, &fence, VK_TRUE, 5'000'000)) {
      LOG.error("fence timed out");
      return false;
    }

    vkDestroyFence(dev, fence, nullptr);
  }

  vkDestroyFramebuffer(dev, f, nullptr);
  vkFreeCommandBuffers(dev, cmd_pool, 1, &cmd_buf);

  return false;
}

L_CUVK_END_
