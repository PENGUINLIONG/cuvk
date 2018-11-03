#include "cuvk/pipeline.hpp"
#include "cuvk/logger.hpp"

L_CUVK_BEGIN_

PipelineContextual::PipelineContextual(const Context& ctxt) { }

bool PipelineContextual::create_shader_module(
    const Spirv& spv, L_OUT VkShaderModule& mod) const {
  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spv.code.size();
  smci.pCode = spv.code.data();

  if (L_VK <- vkCreateShaderModule(ctxt().dev(), &smci, nullptr, &mod)) {
    return false;
  }
  return true;
}

bool PipelineContextual::create_pl() {
  auto dev = ctxt().dev();

  auto dslbs = layout_bindings();
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.pBindings = dslbs.data();
  dslci.bindingCount = (uint32_t)dslbs.size();

  if (L_VK <- vkCreateDescriptorSetLayout(
      dev, &dslci, nullptr, &_desc_set_layout)) {
    return false;
  }

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &_desc_set_layout;

  if (L_VK <- vkCreatePipelineLayout(dev, &plci, nullptr, &_pl_layout)) {
    return false;
  }
}
void PipelineContextual::destroy_pl() {
  auto dev = ctxt().dev();

  if (_pl_layout) {
    vkDestroyPipelineLayout(dev, _pl_layout, nullptr);
    _pl_layout = VK_NULL_HANDLE;
  }
  if (_desc_set_layout) {
    vkDestroyDescriptorSetLayout(dev, _desc_set_layout, nullptr);
    _desc_set_layout = VK_NULL_HANDLE;
  }
}

bool PipelineContextual::context_changing() {
  destroy_pl();
  return true;
}
bool PipelineContextual::context_changed() {
  return create_pl();
}



ComputeShaderContextual::ComputeShaderContextual(const Context& ctxt) :
  PipelineContextual(ctxt) { }

bool ComputeShaderContextual::create_pl() {
  auto dev = ctxt().dev();

  if (!PipelineContextual::create_pl()) {
    return false;
  }

  if (!create_shader_module(comp_spirv(), _mod)) {
    return false;
  }

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.layout = _pl_layout;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = _mod;
  cpci.stage.pName = "main";

  // TODO:: (penguinliong) Use cache.
  VkPipelineCache pl_cache = VK_NULL_HANDLE;
  L_VK <- vkCreateComputePipelines(dev, pl_cache, 1, &cpci, nullptr, &_pl);
}
void ComputeShaderContextual::destroy_pl() {
  auto dev = ctxt().dev();

  if (_pl) {
    vkDestroyPipeline(dev, _pl, nullptr);
    _pl = VK_NULL_HANDLE;
  }
  if (_mod) {
    vkDestroyShaderModule(dev, _mod, nullptr);
    _mod = VK_NULL_HANDLE;
  }
  PipelineContextual::destroy_pl();
}



GraphicsShaderContextual::GraphicsShaderContextual(const Context& ctxt,
    uint32_t rows, uint32_t cols, uint32_t layers) :
  PipelineContextual(ctxt), _rows(rows), _cols(cols), _layers(layers) { }

bool GraphicsShaderContextual::create_pass() {
  // Create render pass with attachment (output image) info. We have no
  // attachment here.
  VkSubpassDescription sd{};
  sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

  VkRenderPassCreateInfo rpci{};
  rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  rpci.subpassCount = 1;
  rpci.pSubpasses = &sd;

  if (L_VK <- vkCreateRenderPass(ctxt().dev, &rpci, nullptr, &_pass)) {
    return false;
  }
}

bool GraphicsShaderContextual::create_pl() {
  auto dev = ctxt().dev();

  if (!PipelineContextual::create_pl()) {
    return false;
  }

  // Load shader stages.
  if (!create_shader_module(vert_spirv(), _mods[0]) ||
      !create_shader_module(geom_spirv(), _mods[1]) ||
      !create_shader_module(frag_spirv(), _mods[2])) {
    return false;
  }

  std::array<VkPipelineShaderStageCreateInfo, 3> pssci{};
  pssci[0].sType = pssci[1].sType = pssci[2].sType =
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pssci[0].pName = pssci[1].pName = pssci[2].pName = "main";
  pssci[0].module = _mods[0];
  pssci[1].module = _mods[1];
  pssci[2].module = _mods[2];
  pssci[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  pssci[1].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
  pssci[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT;

  // Create render pass.
  if (!create_pass()) { return false; }

  // Vertex inputs.
  std::array<VkVertexInputBindingDescription, 1> vibds {
    VkVertexInputBindingDescription
    { 0, 6 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX },
  };
  std::array<VkVertexInputAttributeDescription, 4> viads {
    VkVertexInputAttributeDescription
    { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0                 }, // pos
    { 1, 0, VK_FORMAT_R32G32_SFLOAT, 2 * sizeof(float) }, // size
    { 2, 0, VK_FORMAT_R32_SFLOAT,    4 * sizeof(float) }, // orient
    { 3, 0, VK_FORMAT_R32_UINT,      5 * sizeof(float) }, // univ
  };

  VkPipelineVertexInputStateCreateInfo pvisci{};
  pvisci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  pvisci.vertexAttributeDescriptionCount = (uint32_t)viads.size();
  pvisci.pVertexAttributeDescriptions = viads.data();
  pvisci.vertexBindingDescriptionCount = (uint32_t)vibds.size();
  pvisci.pVertexBindingDescriptions = vibds.data();

  // Input assembly.
  VkPipelineInputAssemblyStateCreateInfo piasci{};
  piasci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  piasci.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

  // Viewport and scissors.
  VkViewport viewport { 0, 0, _cols, _rows, 0.0, 0.0 };
  VkRect2D scissor { { 0, 0 }, { _cols, _rows } };

  VkPipelineViewportStateCreateInfo pvsci{};
  pvsci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  pvsci.viewportCount = 1;
  pvsci.pViewports = &viewport;
  pvsci.scissorCount = 1;
  pvsci.pScissors = &scissor;

  // Resterization requirements.
  VkPipelineRasterizationStateCreateInfo prsci{};
  prsci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  prsci.cullMode = VK_BACK;
  prsci.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

  // Multisampling which we don't need.
  VkPipelineMultisampleStateCreateInfo pmsci{};
  pmsci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  pmsci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  pmsci.minSampleShading = 1.0;

  // Blending which we don't need.
  VkPipelineColorBlendStateCreateInfo pcbsci{};
  pcbsci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  pcbsci.blendConstants[0] = 1.0;

  // Create graphics pipeline.
  VkGraphicsPipelineCreateInfo gpci{};
  gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  gpci.layout = _pl_layout;
  gpci.stageCount = 3;
  gpci.pStages = pssci.data();
  gpci.pVertexInputState = &pvisci;
  gpci.pInputAssemblyState = &piasci;
  gpci.pViewportState = &pvsci;
  gpci.pRasterizationState = &prsci;
  gpci.pMultisampleState = &pmsci;
  gpci.pColorBlendState = &pcbsci;
  gpci.renderPass = _pass;
  gpci.subpass = 0;

  // TODO: (penguinliong) Do we need to use cache?
  if (L_VK <-vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1,
      &gpci, nullptr, &_pl)) {
    return false;
  }
}
void GraphicsShaderContextual::destroy_pl() {
  auto dev = ctxt().dev();

  if (_pl) {
    vkDestroyPipeline(dev, _pl, nullptr);
    _pl = VK_NULL_HANDLE;
  }

  for (auto mod = _mods.begin(); mod != _mods.end(); ++mod) {
    if (*mod) {
      vkDestroyShaderModule(dev, *mod, nullptr);
      *mod = VK_NULL_HANDLE;
    }
  }
  PipelineContextual::destroy_pl();
}

L_CUVK_END_
