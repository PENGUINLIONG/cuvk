#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/context.hpp"
#include <vector>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_

struct Spirv {
  std::vector<uint32_t> _code;
public:
  Spirv();
  Spirv(std::vector<uint32_t>&& code);
  Spirv(const char* spirv, size_t len);

  size_t size() const;
  const uint32_t* data() const;
};

class PipelineContextual : public Contextual {
protected:
  VkDescriptorSetLayout _desc_set_layout;
  VkDescriptorPool _desc_pool;
  VkDescriptorSet _desc_set;
  VkPipelineLayout _pl_layout;
  VkPipeline _pl;

  bool create_shader_module(const Spirv& spv, L_OUT VkShaderModule& mod) const;

  virtual bool create_pl() = 0;
  virtual void destroy_pl() = 0;

public:
  using LayoutBinding = VkDescriptorSetLayoutBinding;

  PipelineContextual();

  virtual const std::vector<LayoutBinding>& layout_binds() const = 0;

  bool context_changing() override;
  bool context_changed() override;
};

// Contextual objects that make use of shaders and pipelines.
class ComputeShaderContextual : public PipelineContextual {
private:
  VkShaderModule _mod;

  bool create_pl() override final;
  void destroy_pl() override final;

public:
  ComputeShaderContextual() = default;

  virtual Spirv comp_spirv() const = 0;
};


// Contextual objects that make use of shaders and pipelines.
class GraphicsShaderContextual : public PipelineContextual {
private:
  std::array<VkShaderModule, 3> _mods;
  uint32_t _rows, _cols, _layers;
  VkRenderPass _pass;

  bool create_pl() override final;
  void destroy_pl() override final;

public:
  using Binding = VkVertexInputBindingDescription;
  using Attribute = VkVertexInputAttributeDescription;
  using Blend = VkPipelineColorBlendAttachmentState;
  using Attachment = VkAttachmentDescription;

  GraphicsShaderContextual(uint32_t rows, uint32_t cols, uint32_t layers);

  virtual const std::vector<Binding>& in_binds() const = 0;
  virtual const std::vector<Attribute>& in_attrs() const = 0;
  virtual const std::vector<Blend>& out_blends() const = 0;
  virtual const std::vector<Attachment>& out_attaches() const = 0;

  VkRenderPass render_pass() const;

  virtual Spirv vert_spirv() const = 0;
  virtual Spirv geom_spirv() const = 0;
  virtual Spirv frag_spirv() const = 0;
};

L_CUVK_END_
