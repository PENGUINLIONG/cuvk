#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/context.hpp"
#include <vector>
#include <cstddef>
#include <array>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_

struct Spirv {
public:
  std::vector<uint32_t> code;
  Spirv() : code() { }
  Spirv(const char* spirv, size_t len) : code((len + 3) / 4) {
    // TODO: Optimize `memcpy` later.
    std::memcpy((void*)code.data(), spirv, len);
  }
};

class PipelineContextual : public Contextual {
protected:
  VkDescriptorSetLayout _desc_set_layout;
  VkRenderPass _pass;
  VkPipelineLayout _pl_layout;
  VkPipeline _pl;

  bool create_shader_module(const Spirv& spv, L_OUT VkShaderModule& mod) const;

  virtual std::vector<VkDescriptorSetLayoutBinding> layout_bindings() const = 0;

  virtual bool create_pl() = 0;
  virtual void destroy_pl() = 0;

public:
  PipelineContextual(const Context& ctxt);

  bool context_changing() override final;
  bool context_changed() override final;
};

// Contextual objects that make use of shaders and pipelines.
class ComputeShaderContextual : public PipelineContextual {
private:
  VkShaderModule _mod;

  bool create_pl() override final;
  void destroy_pl() override final;

public:
  ComputeShaderContextual(const Context& ctxt);

  virtual Spirv comp_spirv() const = 0;
};


// Contextual objects that make use of shaders and pipelines.
class GraphicsShaderContextual : public PipelineContextual {
private:
  std::array<VkShaderModule, 3> _mods;
  uint32_t _rows, _cols, _layers;

  bool create_pass();

  bool create_pl() override final;
  void destroy_pl() override final;

public:
  GraphicsShaderContextual(const Context& ctxt,
                           uint32_t rows, uint32_t cols, uint32_t layers);

  std::vector<VkVertexInputBindingDescription> in_binds() const;
  std::vector<VkVertexInputAttributeDescription> in_descs() const;

  virtual Spirv vert_spirv() const = 0;
  virtual Spirv geom_spirv() const = 0;
  virtual Spirv frag_spirv() const = 0;
};

L_CUVK_END_
