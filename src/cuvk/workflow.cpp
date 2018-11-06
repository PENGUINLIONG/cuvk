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

  VkCommandBuffer cmd_buf;
  // Ensure that the output buffer is larger than or as large as the input
  // bacteria buffer.
  if (bacs_out.size() < bacs.size()) {
    LOG.error("output buffer must be able to contain the result data");
    return false;
  }

  // Calculate number of specs and bacteria provided.
  size_t nspec = deform_specs.size() / sizeof(shader_interface::DeformSpecs);
  size_t nbac = bacs.size() / sizeof(shader_interface::Bacterium);

  /* Allocate command buffer. */ {
    VkCommandBufferAllocateInfo cbai {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    cbai.commandPool = ctxt().get_cmd_pool(ExecType::Compute);

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

    VkPipelineStageFlags psf = VK_PIPELINE_STAGE_TRANSFER_BIT;

    VkSubmitInfo si {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf;
    si.pWaitDstStageMask = &psf;

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

  return true;
}

std::vector<Evaluation::LayoutBinding> Evaluation::_layout_binds = {
  VkDescriptorSetLayoutBinding
  // image2D real_univ
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr },
  { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr },
  { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr },
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
  { 3, 0, VK_FORMAT_R32_UINT,      5 * sizeof(float) }, // univ
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
Spirv Evaluation::vert_spirv() const {
  return read_spirv("assets/shaders/eval.vert.spv");
}
Spirv Evaluation::geom_spirv() const {
  return read_spirv("assets/shaders/eval.geom.spv");
}
Spirv Evaluation::frag_spirv() const {
  return read_spirv("assets/shaders/eval.frag.spv");
}

bool Evaluation::execute(const Storage& bacs,     size_t n_bacs,
                         L_OUT Storage& diff_map,
                         L_OUT Storage& costs) {
  auto dev = ctxt().dev();
  VkCommandBuffer cmd_buf;

  /* Allocate command buffer. */ {
    VkCommandBufferAllocateInfo cbai {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    cbai.commandPool = ctxt().get_cmd_pool(ExecType::Graphics);

    if (L_VK <- vkAllocateCommandBuffers(dev, &cbai, &cmd_buf)) {
      LOG.error("unable to allocate command buffer for evaluation");
      return false;
    }
  }

  /* Start recording commands. */ {
    VkCommandBufferBeginInfo cbbi {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (L_VK <- vkBeginCommandBuffer(cmd_buf, &cbbi)) {
      LOG.error("unable to start recording commands");
      return false;
    }
  }

  /* Sync to ensure all bacteria data and the real universe are written to
     device memory. */

  /* Sync to ensure shader has written simulated universes and costs. */

  /*  */


  return false;
}
L_CUVK_END_
