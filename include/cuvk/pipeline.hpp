#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/span.hpp"
#include <vector>

L_CUVK_BEGIN_

extern struct Context;
extern struct ImageView;



struct ShaderStage {
  const Shader* shader;

  L_STATIC const char* entry;
  VkShaderStageFlagBits stage;

  // Specialization.
  L_STATIC Span<VkSpecializationMapEntry> spec_entries;
  L_STATIC Span<int32_t> spec_data;
  VkSpecializationInfo spec_info;

  operator VkPipelineShaderStageCreateInfo() const noexcept;
};
struct Shader {
  // Source SPIR-V will be wiped out on make.
  std::vector<uint32_t> spv;

  VkShaderModule shader;

  ShaderStage stage(
    L_STATIC const char* entry, VkShaderStageFlagBits stage) const noexcept;
};



struct ShaderManager {
  const Context* ctxt;

  std::list<Shader> shaders;

  const Shader& declare_shader(std::vector<uint32_t> spv) noexcept;

  ShaderManager(const Context& ctxt) noexcept;
  // By default the SPIR-V source is cleared after a successful make, set
  // `keep_spv` true to keep the source. There is no benefit to keep the source
  // except for that it might help diagnostics, or you want to remake a shader
  // manager. (Why you wanna do that?)
  bool make(bool keep_spv) noexcept;
  void drop() noexcept;
  ~ShaderManager() noexcept;

  ShaderManager(const ShaderManager&) = delete;
  ShaderManager& operator=(const ShaderManager&) = delete;
  ShaderManager(ShaderManager&&);
};



struct DescriptorSetLayout {
  L_STATIC Span<VkDescriptorSetLayoutBinding> layout_binds;
  std::vector<VkDescriptorPoolSize> desc_pool_sizes;

  VkDescriptorSetLayout desc_set_layout;
};



struct DescriptorSet {
  const Context* ctxt;
  const DescriptorSetLayout* desc_set_layout;

  VkDescriptorPool desc_pool;
  VkDescriptorSet desc_set;

  DescriptorSet(const Context& ctxt,
    const DescriptorSetLayout& desc_set_layout) noexcept;
  bool make() noexcept;
  void drop() noexcept;
  ~DescriptorSet() noexcept;

  DescriptorSet(DescriptorSet&&) noexcept;
  
  DescriptorSet& write(
    uint32_t bind_pt, const BufferSlice& buf_slice,
    VkDescriptorType desc_type) noexcept;
  DescriptorSet& write(
    uint32_t bind_pt, const ImageView& img_view, VkImageLayout layout,
    VkDescriptorType desc_type) noexcept;
};



struct RenderPass {
  L_STATIC Span<VkAttachmentDescription> attach_descs;
  L_STATIC Span<VkAttachmentReference> attach_refs;

  VkRenderPass pass;
};



struct FramebufferRequirements {
  L_STATIC Span<const ImageView*> attaches;
  VkExtent2D extent;
  uint32_t nlayer;
};
struct Framebuffer {
  const Context* ctxt;
  const RenderPass* pass;

  FramebufferRequirements req;

  VkFramebuffer framebuf;

  Framebuffer(const Context& ctxt, const RenderPass& pass,
    L_STATIC Span<const ImageView*> attaches,
    VkExtent2D extent, uint32_t nlayer) noexcept;
  bool make() noexcept;
  void drop() noexcept;
  ~Framebuffer() noexcept;

  Framebuffer(const Framebuffer&) = delete;
  Framebuffer& operator=(const Framebuffer&) = delete;

  Framebuffer(Framebuffer&& rv) noexcept;
};



// TODO: (penguinliong) The current implementation only allow a single subpass.
// See if multi-subpass is needed, or it just a non-goal.
struct GraphicsPipeline {
  const Context* ctxt;
  const char* name;

  L_STATIC Span<ShaderStage> stages;
  L_STATIC Span<VkPushConstantRange> push_const_rngs;

  L_STATIC Span<VkVertexInputBindingDescription> vert_binds;
  L_STATIC Span<VkVertexInputAttributeDescription> vert_attrs;
  L_STATIC Span<VkPipelineColorBlendAttachmentState> blends;

  VkExtent2D extent;

  DescriptorSetLayout desc_set_layout;
  RenderPass pass;

  VkPipeline pipe;
  VkPipelineLayout pipe_layout;

  std::optional<DescriptorSet> desc_set() noexcept;
  std::optional<Framebuffer> framebuf(
    L_STATIC Span<const ImageView*> attaches,
    VkExtent2D extent, uint32_t nlayer) noexcept;
};



struct ComputePipeline {
  const Context* ctxt;
  const char* name;

  L_STATIC const ShaderStage* stage;
  L_STATIC Span<VkPushConstantRange> push_const_rngs;

  std::optional<std::array<uint32_t, 3>> local_workgrp_size;
  DescriptorSetLayout desc_set_layout;

  VkPipeline pipe;
  VkPipelineLayout pipe_layout;

  std::optional<DescriptorSet> desc_set() noexcept;
};



struct PipelineManager {
  const Context* ctxt;

  std::list<GraphicsPipeline> graph_pipes;
  std::list<ComputePipeline> comp_pipes;

  const GraphicsPipeline& declare_graph_pipe(const char* name,
    VkExtent2D extent,
    L_STATIC Span<ShaderStage> stages,
    L_STATIC Span<VkPushConstantRange> push_const_rngs,
    L_STATIC Span<VkDescriptorSetLayoutBinding> layout_binds,
    L_STATIC Span<VkAttachmentDescription> attach_descs,
    L_STATIC Span<VkAttachmentReference> attach_refs,
    L_STATIC Span<VkVertexInputBindingDescription> vert_binds,
    L_STATIC Span<VkVertexInputAttributeDescription> vert_attrs,
    L_STATIC Span<VkPipelineColorBlendAttachmentState> blends) noexcept;
  const ComputePipeline& declare_comp_pipe(const char* name,
    const ShaderStage* stage,
    L_STATIC Span<VkPushConstantRange> push_const_rngs,
    L_STATIC Span<VkDescriptorSetLayoutBinding> layout_binds,
    std::optional<std::array<uint32_t, 3>> local_workgrp) noexcept;

  PipelineManager(const Context& ctxt) noexcept;
  bool make() noexcept;
  void drop() noexcept;
  ~PipelineManager() noexcept;

  PipelineManager(const PipelineManager&) = delete;
  PipelineManager& operator=(const PipelineManager) = delete;

  PipelineManager(PipelineManager&&) noexcept;

private:
  bool make_layouts(
    L_STATIC Span<VkDescriptorSetLayoutBinding> layout_binds,
    L_STATIC Span<VkPushConstantRange> push_const_rngs,
    L_OUT VkDescriptorSetLayout& desc_set_layout,
    L_OUT VkPipelineLayout& pipe_layout) noexcept;
  bool make_graph_pipes() noexcept;
  bool make_comp_pipes() noexcept;
  void drop_graph_pipes() noexcept;
  void drop_comp_pipes() noexcept;
};

L_CUVK_END_
