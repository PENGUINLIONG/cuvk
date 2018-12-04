#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/span.hpp"
#include "cuvk/storage.hpp"
#include <vector>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_

struct Context; // External dependency.

struct ShaderStage;
struct Shader;

struct ShaderManager;

struct DescriptorSetLayout;
struct DescriptorSet;

struct GraphicsPipeline;
struct ComputePipeline;



struct ShaderStage {
  const Shader* shader;

  L_STATIC const char* entry;
  VkShaderStageFlagBits stage;

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

  DescriptorSet(const Context& ctxt, const GraphicsPipeline& pipe) noexcept;
  DescriptorSet(const Context& ctxt, const ComputePipeline& pipe) noexcept;
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



struct GraphicsIORequirements {
  L_STATIC Span<VkVertexInputBindingDescription> in_binds;
  L_STATIC Span<VkVertexInputAttributeDescription> in_attrs;
  L_STATIC Span<VkPipelineColorBlendAttachmentState> out_blends;
  L_STATIC Span<VkAttachmentDescription> out_attach_descs;
  L_STATIC Span<VkAttachmentReference> out_attach_refs;
  // If `viewport` doesn't have an value, it means that the viewport info is
  // updated on each draw, and a pipeline dynamic state with
  // `VK_DYNAMIC_STATE_VIEWPORT` is created.
};
struct GraphicsPipeline {
  const Context* ctxt;
  const char* name;

  Span<ShaderStage> stages;
  Span<VkPushConstantRange> push_const_rngs;

  GraphicsIORequirements graph_io_req;

  DescriptorSetLayout desc_set_layout;

  VkPipeline pipe;
  VkPipelineLayout pipe_layout;

  VkRenderPass pass;
  VkFramebuffer framebuf;
  const ImageView* attach;

  std::optional<DescriptorSet> desc_set() noexcept;
};

struct ComputePipeline {
  const Context* ctxt;
  const char* name;

  const ShaderStage* stage;
  Span<VkPushConstantRange> push_const_rngs;

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
    L_STATIC Span<ShaderStage> stages,
    L_STATIC Span<VkPushConstantRange> push_const_rngs,
    L_STATIC Span<VkDescriptorSetLayoutBinding> layout_binds,
    L_STATIC Span<VkVertexInputBindingDescription> in_binds,
    L_STATIC Span<VkVertexInputAttributeDescription> in_attrs,
    L_STATIC Span<VkPipelineColorBlendAttachmentState> out_blends,
    L_STATIC Span<VkAttachmentDescription> out_attach_descs,
    L_STATIC Span<VkAttachmentReference> out_attach_refs,
    const ImageView& attachment) noexcept;
  const ComputePipeline& declare_comp_pipe(const char* name,
    const ShaderStage* stage,
    L_STATIC Span<VkPushConstantRange> push_const_rngs,
    L_STATIC Span<VkDescriptorSetLayoutBinding> layout_binds) noexcept;

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
