#include "cuvk/workflow.hpp"
#include "cuvk/logger.hpp"
#include <fstream>

L_CUVK_BEGIN_

Spirv read_spirv(const std::string& path) {
  std::fstream f(path, std::fstream::in | std::fstream::binary);
  f.seekg(std::fstream::end);
  size_t size = f.tellg();
  std::vector<char> buf;
  f.seekg(std::fstream::beg);
  f.read(buf.data(), size);
  f.close();
  auto spirv = Spirv(buf.data(), size);
  
  return spirv;
}

bool Deformation::create_com_buf() {
  if (!validate_ctxt()) return false;
  VkCommandBufferAllocateInfo cbai;
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandBufferCount = 1;
  cbai.commandPool = ctxt().get_cmd_pool(ExecType::Compute);

  if (L_VK <- vkAllocateCommandBuffers(ctxt().dev(), &cbai, &_com_buf)) {
    LOG.error("unable to allocate command buffer for deformation");
    return false;
  }

  VkCommandBufferBeginInfo cbbi;
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  if (L_VK <- vkBeginCommandBuffer(_com_buf, &cbbi)) {
    LOG.error("unable to record commands");
    return false;
  }

  //TODO: (penguinliong) Fill command buffer.
  //vkCmdDispatch(_com_buf, 2, , 0);

  return true;
}

std::vector<VkDescriptorSetLayoutBinding> Deformation::layout_bindings() const {
  return {
    VkDescriptorSetLayoutBinding
    // DeformSpecs[] deform_specs
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, VK_SHADER_STAGE_ALL, nullptr },
    // Bac[] bacs
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, VK_SHADER_STAGE_ALL, nullptr },
    // Bac[] bacs_out
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, VK_SHADER_STAGE_ALL, nullptr },
  };
}
Spirv Deformation::comp_spirv() const {
  return read_spirv("assets/shaders/deform.comp.spv");
}

bool Deformation::execute(const   Storage& deform_specs, size_t n_deform_spec,
                          L_INOUT Storage& bacs,         size_t n_bacs) {
  
  return true;
}

std::vector<VkDescriptorSetLayoutBinding> Evaluation::layout_bindings() const {
  return {
    VkDescriptorSetLayoutBinding
    // image2D real_univ
    { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr },
    { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr },
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr },
  };
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
}
L_CUVK_END_
